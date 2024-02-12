from einops import rearrange, reduce, repeat
import torch
import numpy as np

from sklearn.metrics import f1_score, top_k_accuracy_score, accuracy_score


def move_from_tensor_to_ndarray(*kwargs):
    '''
    This function converts an arbitrary number of torch.tensor to np.array.
    This function can automaticly move cuda tensor to cpu.
    '''
    def move_tensor(x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        else:
            return x

    if len(kwargs) == 1:
        tmp_results = move_tensor(kwargs[0])
    else:
        tmp_results = []
        for object in kwargs:
            tmp_results.append(move_tensor(object))

    return tmp_results


def check_tensor(x, positive = True, inf = True, nan = True):
    '''
    Ensure that the input tensor does not contain: negative numbers, inf, and nan.
    
    Args:
    * x  type: torch.tensor shape: any shape
         the input tensor.

    Outputs:
      No outputs available.
    '''
    if positive:
        assert (x < 0).any() == False, 'Negative numbers detected!'

    if inf:
        assert torch.isfinite(x).all() == True, 'inf detected in input!'

    if nan:
        assert torch.isnan(x).any() == False, 'Nan detected in input!'


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
        if (l - r).abs().max() < bisect_early_stop_threshold:
            break
    
    return (l + r)/2

'''
resolution_inf and resolution_between_events.
'''
def decide_resolution_inf_and_resolution_between_events(time_next, memory_ceiling, num_events, mean, var):
    '''
    Suggested batch_size: 1
    '''

    if mean == 0 and var == 1:
        max_ = time_next.mean() + 10 * time_next.var()
    else:
        max_ = mean + 10 * var

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
Approximate an integral based on its definition.
dim refers to the dimension index of expanded_func_value where the integration should be performed.
'''
def approximate_integration(expanded_func_value, expanded_x, dim, only_last_result = False, same_dim_on_expanded_x = False):
    # tensor check
    func_val_number_of_dim = len(expanded_func_value.shape)
    integration_sample_rate = expanded_func_value.shape[dim]

    if same_dim_on_expanded_x:
        assert expanded_x.shape == expanded_func_value.shape
        dim_expanded_x = dim
    else:
        dim_expanded_x = -1

    assert expanded_func_value.device == expanded_x.device
    assert expanded_x.shape[dim_expanded_x] == integration_sample_rate
    device = expanded_func_value.device
    
    expanded_func_value_1 = expanded_func_value.index_select(dim, torch.arange(integration_sample_rate - 1, device = device))
                                                                               # [..., integration_sample_rate - 1, ...]
    expanded_func_value_2 = expanded_func_value.index_select(dim, torch.arange(1, integration_sample_rate, device = device))
                                                                               # [..., integration_sample_rate - 1, ...]
    width_of_rectangle = expanded_x.diff(dim = dim_expanded_x)                 # [..., integration_sample_rate - 1, ...]

    if not same_dim_on_expanded_x:
        the_number_of_dimensions_after_integration_dim = abs(dim) - 1 if dim < 0 else func_val_number_of_dim - dim - 1
        einop = f'... -> ... {"() " * the_number_of_dimensions_after_integration_dim}'
        width_of_rectangle = rearrange(width_of_rectangle, einop)              # [..., integration_sample_rate - 1, ...]

    # \int_{a}{b}{f(x)dx} \approx \sum_{i = 0}^{N - 2}{f(\frac{(b - a)i}{N - 1}) * \frac{(b - a)}{N - 1}}
    integral_of_all_events_1 = (expanded_func_value_1 * width_of_rectangle).cumsum(dim = dim)
                                                                               # [..., integration_sample_rate - 1, ...]
    # \int_{a}{b}{f(x)dx} \approx \sum_{i = 0}^{N - 2}{f(\frac{(b - a)(i + 1)}{N - 1}) * \frac{(b - a)}{N - 1}}
    integral_of_all_events_2 = (expanded_func_value_2 * width_of_rectangle).cumsum(dim = dim)
                                                                               # [..., integration_sample_rate - 1, ...]
    # Effectively increase the precision.
    integral_of_all_events = (integral_of_all_events_1 + integral_of_all_events_2) / 2
                                                                               # [..., integration_sample_rate - 1, ...]
    
    # Prepend 0 to integral_of_all_events because \int_{t_l}^{t_l}{\lambda^*(\tau)d\tau} = 0
    # We have to check the shape.
    integral_start_from_zero = torch.zeros(
        ( *(integral_of_all_events.shape[:dim]), 1, *(integral_of_all_events.shape[dim + 1:] if dim != -1 else []) ), 
        device = device)                                                       # [..., 1, ...]
    integral_of_all_events = torch.concat((integral_start_from_zero, integral_of_all_events), dim = dim)
                                                                               # [..., integration_sample_rate, ...]
    
    if only_last_result:
        integral_of_all_events = torch.select(integral_of_all_events, dim, -1) # [...]

    return integral_of_all_events


'''
custom metrics
'''
def get_f1_and_top_k_acc_in_mae_e(events_true, num_events, p_m):
    f1 = []
    top_k_acc = []
    for (events_true_per_seq, probability_integral_per_seq) in zip(events_true, p_m):
        events_true_per_seq, probability_integral_per_seq = \
            move_from_tensor_to_ndarray(events_true_per_seq, probability_integral_per_seq)
        y_pred = np.argmax(probability_integral_per_seq, axis = -1)

        f1.append(f1_score(y_true = events_true_per_seq, y_pred = y_pred, average = 'macro'))
        top_k_acc_single_event_seq = []
        if num_events > 2:
            for k in range(1, num_events):
                top_k_acc_single_event_seq.append(
                    top_k_accuracy_score(y_true = events_true_per_seq,
                                         y_score = probability_integral_per_seq,
                                         k = k,
                                         labels = np.arange(num_events))
                )
        else:
            top_k_acc_single_event_seq.append(
                accuracy_score(
                    y_true = events_true_per_seq, y_pred = y_pred
                )
            )
        top_k_acc.append(top_k_acc_single_event_seq)
    
    return f1, top_k_acc


def L1_distance_across_events(input, resolution, num_events, time_next):
    '''
    This function calculates the L^1 distance between two functions in scattered form.
    Input:
    1. input:      function values
                   [seq_len * resolution, num_events]
    2. resolution: int
                   the number of points from [t_{i - 1}, t_i]
    3. num_events: int
                   the number of event types
    4. time_next:  [seq_len, num_events]
                   the length of all intervals with interpolations.
    '''

    input = rearrange(input, '(s r) ne -> ne s r', r = resolution)             # [num_events, seq_len, resolution]
    intensity_1 = repeat(input, 'ne s r -> ne new_d s r', new_d = num_events)  # [num_events, num_events, seq_len, resolution]
    intensity_2 = repeat(input, 'ne s r -> new_d ne s r', new_d = num_events)  # [num_events, num_events, seq_len, resolution]
    delta_intensity = np.abs(intensity_1 - intensity_2)                        # [num_events, num_events, seq_len, resolution]

    gap = time_next.detach().cpu().numpy() / (resolution - 1)                  # [seq_len]
    gap = rearrange(gap, 's -> 1 1 s 1')                                       # [num_events, num_events, seq_len, 1]

    L1 = reduce((delta_intensity * gap)[:, :, :, :-1], 'ne1 ne2 s r -> ne1 ne2', 'sum')
                                                                               # [num_events, num_events]
    # round off the value smaller than 1e-6
    L1[L1 < 1e-6] = 0

    return L1


def L1_distance_between_two_funcs(x, y, timestamp, resolution):
    '''
    This function calculates the L^1 distance between two functions.
    Input:
    1. x:          function values
                   [seq_len * resolution, num_events]
    2. y:          function values
                   the number of points from [t_{i - 1}, t_i]
    3. time:       \Delta t
                   the number of event types
    '''

    function_interval = np.abs(x - y).reshape(-1, resolution)[:, :-1]          # [batch_size * seq_len, resolution - 1]
    timestamp = timestamp.reshape(-1, resolution)[:, 1:]                       # [batch_size * seq_len, resolution - 1]

    L1 = (function_interval * timestamp).sum()

    # round up the value smaller than 1e-6
    if L1 < 1e-6:
        L1 = 0

    return L1


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