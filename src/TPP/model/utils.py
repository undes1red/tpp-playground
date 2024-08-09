from einops import rearrange, reduce, repeat
import torch
import numpy as np

from src.toolbox.misc import move_from_tensor_to_ndarray

from sklearn.metrics import f1_score, top_k_accuracy_score, accuracy_score


'''
This function returns a list consisting of the step size of each operation.
For example:
(total_rate: 40, step_size: 15) -> [15, 15, 10] 
'''
def step_split(total_rate, step_size):
    substep_rate_list = []
    while total_rate > 0:
        substep_rate_list.append(step_size)
        total_rate -= step_size
    substep_rate_list[-1] += total_rate

    return substep_rate_list


'''
Bisection Method.
'''
def median_prediction(max_step, bisect_early_stop_threshold, bisect_func, probability_threshold, 
                      *args, l_val = 0.0001, r_val = 1e6, **kwargs):
    l = l_val*torch.ones_like(probability_threshold)
    r = r_val*torch.ones_like(probability_threshold)

    for _ in range(max_step):
        c = (l + r)/2
        v = bisect_func(c, probability_threshold, *args, **kwargs)
        l = torch.where(v < 0, c, l)
        r = torch.where(v >= 0, c, r)
        if torch.allclose(r, l, atol = bisect_early_stop_threshold):
            break
    
    return (l + r)/2


'''
Thinning algorithm.
'''
def thinning_sampling(maximum_thinning_loops, max_sample_time_limit, sample_output_shape, device, intensity_func, \
                      find_maximum_intensity_values_in_one_interval, *args, **kwargs):
    sample_rate, batch_size, seq_len = sample_output_shape
    thinning_unit_interval_length = max_sample_time_limit / maximum_thinning_loops

    predicted_time = torch.zeros(sample_rate, batch_size, seq_len, dtype = torch.int32, device = device)
                                                                           # [sample_rate, batch_size, seq_len]
    # The initial mask tensor contains only zero.
    # Zero means we have got a valid time sample.
    # One means we need a resample
    rejected_mask = torch.ones(sample_rate, batch_size, seq_len, dtype = torch.int32, device = device)
                                                                           # [sample_rate, batch_size, seq_len]
    thinning_loops = 0
    while(rejected_mask.sum() > 0):
        thinning_loops += 1
        if thinning_loops > maximum_thinning_loops:
            break

        sampling_interval_left_side = torch.ones_like(rejected_mask) * thinning_unit_interval_length * (thinning_loops - 1)
                                                                               # [sample_rate, batch_size, seq_len]
        sampling_interval_right_side = torch.ones_like(rejected_mask) * thinning_unit_interval_length * thinning_loops
                                                                               # [sample_rate, batch_size, seq_len]
        intensity_values_for_thinning_upper_bound = find_maximum_intensity_values_in_one_interval(sampling_interval_left_side, sampling_interval_right_side, *args, **kwargs)
                                                                               # [sample_rate, batch_size, seq_len]
        # Exponential distribution: F(x) = 1 - exp(-\\lambda x) => x = ln(1 - F(x)) / (-\\lambda)
        probability_threshold_for_exp = torch.zeros_like(intensity_values_for_thinning_upper_bound)
                                                                               # [sample_rate, batch_size, seq_len]
        torch.nn.init.uniform_(probability_threshold_for_exp)                  # [sample_rate, batch_size, seq_len]
        probability_threshold_for_thinning = torch.zeros_like(intensity_values_for_thinning_upper_bound)
                                                                               # [sample_rate, batch_size, seq_len]
        torch.nn.init.uniform_(probability_threshold_for_thinning)             # [sample_rate, batch_size, seq_len]
        sampled_time = - torch.log(1 - probability_threshold_for_exp) / intensity_values_for_thinning_upper_bound
                                                                               # [sample_rate, batch_size, seq_len]
        # Part 1: exclude time exceeding the limit.
        sampled_time_exceeding_limit = sampled_time > thinning_unit_interval_length
                                                                               # [sample_rate, batch_size, seq_len]
        # Part 2: exclude time rejected by the learned MTPP.
        intensity_values_at_sampled_time = intensity_func(sampled_time, *args, **kwargs)
                                                                               # [sample_rate, batch_size, seq_len]
        sampled_time_rejected = probability_threshold_for_thinning > intensity_values_at_sampled_time / intensity_values_for_thinning_upper_bound
                                                                               # [sample_rate, batch_size, seq_len]
        rejected_in_this_loop = sampled_time_rejected | sampled_time_exceeding_limit
                                                                               # [sample_rate, batch_size, seq_len]
        accept_mask = rejected_mask & (~rejected_in_this_loop)                 # [sample_rate, batch_size, seq_len]
        rejected_mask = rejected_mask & rejected_in_this_loop                  # [sample_rate, batch_size, seq_len]
        predicted_time = predicted_time + accept_mask * sampled_time + rejected_mask * thinning_unit_interval_length
                                                                               # [sample_rate, batch_size, seq_len]
    return predicted_time


'''
Sample event from a (unnormalized) probability distribution.
'''
def predict_event(probability, sample = False):
    # The shape of the input probability is [..., num_events].
    if sample:
        distribution_of_marks = torch.distributions.categorical.Categorical(probability)
        sampled_marks = distribution_of_marks.sample()                         # [...]
    else:
        sampled_marks = torch.argmax(probability, dim = -1)                    # [...]
    
    return sampled_marks


'''
resolution_inf and resolution_between_events.
'''
def decide_resolution_inf_and_resolution_between_events(time_next, memory_ceiling, num_events, mean, std):
    '''
    Suggested batch_size: 1
    '''

    if mean == 0 and std == 1:
        max_ = time_next.mean() + 10 * time_next.std()
    else:
        max_ = mean + 10 * std

    if mean == 0:
        resolution_between_events = max(min(int(time_next.mean().item() // 0.005), 500), 10)
    else:
        resolution_between_events = max(min(int(mean // 0.005), 500), 10)
        
    max_ = min(1e6, max_)
    resolution_inf = max(int(max_ // 0.005), 100)

    batch_size, seq_len = time_next.shape
    if batch_size * seq_len * resolution_inf * num_events > memory_ceiling:
        resolution_inf = int(memory_ceiling // (seq_len * num_events * batch_size))
    
    if batch_size * seq_len * resolution_between_events * num_events * num_events > memory_ceiling:
        resolution_between_events = int(memory_ceiling // (seq_len * num_events * num_events * batch_size))

    return max_, resolution_inf, resolution_between_events


'''
custom metrics
'''
def get_f1_and_top_k_acc_in_mae_e(events_true, p_m, input_mask, num_events):
    f1 = []
    top_k_acc = []
    for (events_true_per_seq, probability_integral_per_seq, input_mask_per_seq) in zip(events_true, p_m, input_mask):
        events_true_per_seq, probability_integral_per_seq, input_mask_per_seq \
            = move_from_tensor_to_ndarray(events_true_per_seq, probability_integral_per_seq, input_mask_per_seq)
        y_pred = np.argmax(probability_integral_per_seq, axis = -1)

        selected_events_true_per_seq = events_true_per_seq[input_mask_per_seq == 1]
        selected_y_pred = y_pred[input_mask_per_seq == 1]
        selected_probability_integral_per_seq = probability_integral_per_seq[input_mask_per_seq == 1]

        f1.append(f1_score(y_true = selected_events_true_per_seq, y_pred = selected_y_pred, average = 'macro'))
        top_k_acc_single_event_seq = []
        if num_events > 2:
            for k in range(1, num_events):
                top_k_acc_single_event_seq.append(
                    top_k_accuracy_score(y_true = selected_events_true_per_seq,
                                         y_score = selected_probability_integral_per_seq,
                                         k = k,
                                         labels = np.arange(num_events))
                )
        else:
            top_k_acc_single_event_seq.append(
                accuracy_score(
                    y_true = selected_events_true_per_seq, y_pred = selected_y_pred
                )
            )
        top_k_acc.append(top_k_acc_single_event_seq)
    
    return f1, top_k_acc


'''
For EHD.
'''
def pick_log_probability(log_probability, last_index, seq_len_x):
    device = last_index.device
    batch_size = last_index.shape[0]

    start_idx = torch.clamp(last_index - 1 - seq_len_x, min = 0)
                                                                           # [batch_size]
    index_indices = torch.arange(seq_len_x, device = device)               # [seq_len_x]
    index_indices = repeat(index_indices, '... -> b ...', b = batch_size) + start_idx.unsqueeze(dim = -1)
                                                                           # [batch_size, seq_len_x]
    log_probability_x = log_probability.gather(-1, index_indices)          # [batch_size, seq_len_x]

    return log_probability_x