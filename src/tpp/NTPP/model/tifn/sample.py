import torch
from einops import rearrange, repeat, reduce, pack

from src.tpp.tpp_models.utils import predict_event

from src.tpp.tpp_models.basic_tpp_model import its_lower_bound, its_upper_bound
from src.tpp.tpp_models.utils import step_split, median_prediction


def sample_time(self, sampling_approach = 'its', task = 'mt', *args, **kwargs):
    '''
    Sample time from the learned MTPP model using p(t) or p(t|m) using different methods.
       
    ### Args
        * ```str``` sampling_approach
          Use which method to sample from a distribution.
          its      -> Inverse Transform Sampling.
          thinning -> Thinning algorithm.
        * ```str``` task
          Use which distribution to sample time, p(t) or p(t|m)?
          mt -> p(t|m)
          tm -> p(t)
        * ```bool``` autoregressive
          If true, we autoregressively generate a sequence using the learned MTPP model.
          If false, we sample one next event given a history sequence.
    
    ### Args required when sampling from p(t|m) using its.
        * ```torch.tensor``` events_history
          shape: ```[batch_size, seq_len]```
          Historical event sequences. Commonly, this sequence is a slice of the original event sequence from 0 to seq_len - 1(included).
        * ```torch.tensor``` time_history
          shape: ```[batch_size, seq_len]```
          Historical time sequences. Similar to events_history, we always generate this sequence as a slice of the original time sequence from 0 to seq_len - 1(included).
        * ```torch.tensor``` p_m
          shape: ```[batch_size, seq_len, num_events]```
          The value of p(m) over the different mark m.
        * ```int``` resolution
          The number of interpolated points in a time interval between two adjoint events for integration estimation.
          The number of interpolated points counts the start and end point of the interval.
        * ```int``` number_of_total_samples
          This tells how many time samples are generated from the time distribution.
        * ```int``` step
          This parameter controls how many samples are generated in one shot when sampling from p(t|m).
        * ```float``` inf_val
          the upper limit of the bisection method.
        * ```float``` mean
        * ```float``` std
          Used for input time scaling.

    ### Args required when sampling from p(t) using its.
        * ```torch.tensor``` events_history
          shape: ```[batch_size, seq_len]```
          Historical event sequences. Commonly, this sequence is a slice of the original event sequence from 0 to seq_len - 1(included).
        * ```torch.tensor``` time_history
          shape: ```[batch_size, seq_len]```
          Historical time sequences. Similar to events_history, we always generate this sequence as a slice of the original time sequence from 0 to seq_len - 1(included).
        * ```int``` number_of_total_samples
          This tells how many time samples are generated from the time distribution.
        * ```int``` step
          This parameter controls how many samples are generated in one shot when sampling from p(t|m).
        * ```float``` inf_val
          the upper limit of the bisection method.
        * ```float``` mean
        * ```float``` std
          Used for input time scaling.

    ### Args required when sampling from p(t|m) using thinning.
        Do not exist since it is impossible for now to sample from p(t|m) using thinning.
    
    ### Args required when sampling from p(t) using thinning.
        * ```torch.tensor``` events_history
          shape: ```[batch_size, seq_len]```
          Historical event sequences. Commonly, this sequence is a slice of the original event sequence from 0 to seq_len - 1(included).
        * ```torch.tensor``` time_history
          shape: ```[batch_size, seq_len]```
          Historical time sequences. Similar to events_history, we always generate this sequence as a slice of the original time sequence from 0 to seq_len - 1(included).
        * ```int``` number_of_total_samples
          This tells how many time samples are generated from the time distribution.
        * ```int``` step
          This parameter controls how many samples are generated in one shot when sampling from p(t|m).
        * ```float``` inf_val
          the upper limit of the bisection method.
        * ```float``` mean
        * ```float``` std
          Used for input time scaling.
    '''
    dict_sampling_apparoch = {
        'its': sampling_by_its,
        'thinning': sampling_by_thinning
    }

    return dict_sampling_apparoch[sampling_approach](self, task, *args, **kwargs)


def sampling_by_its(self, task, *args, **kwargs):
    dict_apparoch_for_tasks = {
        'mt': sampling_by_its_for_mt,
        'tm': sampling_by_its_for_tm
    }

    return dict_apparoch_for_tasks[task](self, *args, **kwargs)


def sampling_by_its_for_mt(self, events_history, time_history, mask_history, p_m,
                           number_of_total_samples, step, inf_val, mean, std, autoregressive = False):
    # Preprocess
    sample_rate_list = step_split(number_of_total_samples, step)

    def bisect_target(taus, probability_threshold):
        # \\int_{tau}^{+\\inf}{p(m, \\tau|\\mathcal{H})d\\tau}
        if autoregressive:
            probability_integral_from_t_to_infinite = self.model('sample', events_history, time_history, taus, mean = mean, std = std)
                                                                            # [sample_rate, num_events]
        else:
            probability_integral_from_t_to_infinite = self.model('default_forward', events_history, time_history, taus, mask_history, mean = mean, std = std)
                                                                            # [sample_rate, batch_size, seq_len, num_events]
        # \\int_{0}^{tau}{p(m, \\tau|\\mathcal{H})d\\tau}
        p_mt = p_m - probability_integral_from_t_to_infinite               # [sample_rate, batch_size, seq_len, num_events] if not autoregressive else [sample_rate, num_events]
        p_t_m = p_mt / p_m                                                 # [sample_rate, batch_size, seq_len, num_events] if not autoregressive else [sample_rate, num_events]
        p_gap = p_t_m - probability_threshold                              # [sample_rate, batch_size, seq_len, num_events] if not autoregressive else [sample_rate, num_events]

        return p_gap

    # Preprocess
    tau_pred = []
    batch_size, seq_len = time_history.shape
    if not autoregressive:
        p_m = p_m.unsqueeze(dim = 0)                                       # [1, batch_size, seq_len, num_events] if not autoregressive else [sample_rate, 1, num_events]

    for sub_sample_rate in sample_rate_list:
        if autoregressive:
            probability_threshold = torch.zeros((sub_sample_rate, self.num_events), device = self.device)
                                                                            # [sub_sample_rate, num_events]
        else:
            probability_threshold = torch.zeros((sub_sample_rate, batch_size, seq_len, self.num_events), device = self.device)
                                                                            # [sub_sample_rate, batch_size, seq_len, num_events]
        torch.nn.init.uniform_(probability_threshold, a = its_lower_bound, b = its_upper_bound)
                                                                            # [sub_sample_rate, batch_size, seq_len, num_events] if not autoregressive else [sub_sample_rate, num_events]
        tau_pred.append(median_prediction(self.max_step, self.bisect_early_stop_threshold, \
                                            bisect_target, probability_threshold, r_val = inf_val))
                                                                            # [sub_sample_rate, batch_size, seq_len, num_events] if not autoregressive else [sub_sample_rate, num_events]

    tau_pred = torch.cat(tau_pred, dim = 0)                                # [sample_rate, batch_size, seq_len, num_events] if not autoregressive else [sample_rate, num_events]

    return tau_pred


def sampling_by_its_for_tm(self, events_history, time_history, mask_history,
                           number_of_total_samples, step, mean, std, 
                           autoregressive = False):
    # Preprocess
    sample_rate_list = step_split(number_of_total_samples, step)

    def evaluate(taus, probability_threshold, integral_from_zero_to_inf):
        taus = repeat(taus, '... -> ... ne', ne = self.num_events)         # [..., num_events]
        if autoregressive:
            probability_integral_from_t_to_inf = self.model('sample', events_history, time_history, taus, mean, std)
                                                                            # [sample_rate, num_events]
        else:
            probability_integral_from_t_to_inf = self.model('default_forward', events_history, time_history, taus, mask_history, mean, std)
                                                                            # [sample_rate, batch_size, seq_len, num_events]
        # P_m(t) = \\int_{0}^{t}{p(t|m, \\mathcal{H})}
        probability_integral = integral_from_zero_to_inf - probability_integral_from_t_to_inf
                                                                            # [sample_rate, batch_size, seq_len, num_events] if not autoregressive else [sample_rate, num_events]
        probability_integral = torch.sum(probability_integral, dim = -1)   # [sample_rate, batch_size, seq_len] if not autoregressive else [sample_rate]
        
        return probability_integral - probability_threshold

    tau_pred = []
    batch_size, seq_len = time_history.shape

    for sub_sample_rate in sample_rate_list:
        if autoregressive:
            probability_threshold = torch.zeros(sub_sample_rate, device = self.device)
                                                                            # [sub_sample_rate]
        else:
            probability_threshold = torch.zeros((sub_sample_rate, batch_size, seq_len), device = self.device)
                                                                            # [sub_sample_rate, batch_size, seq_len]
        torch.nn.init.uniform_(probability_threshold, a = its_lower_bound, b = its_upper_bound)
                                                                            # [sub_sample_rate, batch_size, seq_len] if not autoregressive else [sub_sample_rate]

        time_next_zero = torch.zeros_like(probability_threshold)           # [sub_sample_rate, batch_size, seq_len] if not autoregressive else [sub_sample_rate]
        time_next_zero = repeat(time_next_zero, '... -> ... ne', ne = self.num_events)
                                                                            # [sub_sample_rate, batch_size, seq_len, num_events] if not autoregressive else [sub_sample_rate, num_events]
        if autoregressive:
            integral_from_zero_to_inf = self.model('sample', events_history, time_history, time_next_zero, mean = mean, std = std)
                                                                            # [sub_sample_rate, num_events]
        else:
            integral_from_zero_to_inf = self.model('default_forward', events_history, time_history, time_next_zero, mask_history, mean = mean, std = std)
                                                                            # [sub_sample_rate, batch_size, seq_len, num_events]

        tau_pred.append(median_prediction(self.max_step, self.bisect_early_stop_threshold, \
                                            evaluate, probability_threshold, integral_from_zero_to_inf))
                                                                            # [sub_sample_rate, batch_size, seq_len] if not autoregressive else [sub_sample_rate]
    tau_pred = torch.cat(tau_pred, dim = 0)                                # [sample_rate, batch_size, seq_len] if not autoregressive else [sample_rate]

    return tau_pred


def sampling_by_thinning(self, task, *args, **kwargs):
    dict_apparoch_for_tasks = {
        'mt': self.sampling_by_thinning_for_mt,
        'tm': self.sampling_by_thinning_for_tm
    }

    return dict_apparoch_for_tasks[task](*args, **kwargs)


def sampling_by_thinning_for_mt(self, *args, **kwargs):
    raise Exception('Thinning algorithm can not solve task MT. Please use ITS by setting sampling_approach = its.')


def sampling_by_thinning_for_tm(self, events_history, time_history, mask_history, number_of_total_samples, step, mean, std):
    raise Exception('IFIB does not know intensity functions, which thinning algorithm requires. Please use ITS by setting sampling_approach = its.')

def sample_time_event(self, number_of_sampled_sequences, end_time, mean, std):
    '''
    This function will sample x sequences by the learned probability distribution following the time-event prediction procedure.
    Steps:
    1. Sample a time \\(t_s\\) from p^*(t) = \\sum{n \\in M}{p^*(m, t)} referring to existing history
    2. Judge the mark of this event by comparing \\(\\lambda^*(m, t_s)\\).
    '''
    time_history_for_sampling = torch.zeros(number_of_sampled_sequences, 1, device = self.device)
                                                                           # [number_of_sampled_sequences, 1]
    events_history_for_sampling = torch.ones(number_of_sampled_sequences, 1, device = self.device, dtype = torch.int32) * self.num_events
                                                                           # [number_of_sampled_sequences, 1]
    tmp_sum_of_sampled_time = time_history_for_sampling.sum(dim = -1)      # [number_of_sampled_sequences]

    MAX_sampled_seq = 250
    seq_length = 1

    while seq_length < MAX_sampled_seq:
        mask_history_for_sampling = torch.ones_like(time_history_for_sampling)
        sampled_time = self.sample_time('its', 'tm',
                                        events_history_for_sampling, time_history_for_sampling, mask_history_for_sampling, 
                                        number_of_sampled_sequences, number_of_sampled_sequences, mean, std,
                                        autoregressive = True)             # [number_of_sampled_sequences]
        repeated_sampled_time = repeat(sampled_time, '... -> ... ne', ne = self.num_events)
                                                                           # [number_of_sampled_sequences, num_events]
        repeated_sampled_time.requires_grad = True
        integral_from_sampled_time_to_inf = self.model('sample', events_history_for_sampling, time_history_for_sampling, repeated_sampled_time, 
                                                       mean = mean, std = std)
                                                                           # [number_of_sampled_sequences, num_events]
        probability_for_each_event_at_pred_time = - torch.autograd.grad(
            outputs = integral_from_sampled_time_to_inf,
            inputs = repeated_sampled_time,
            grad_outputs = torch.ones_like(integral_from_sampled_time_to_inf)
        )[0]                                                               # [number_of_sampled_sequences, num_events]
        repeated_sampled_time.requires_grad = False

        sampled_marks = predict_event(probability_for_each_event_at_pred_time, sample = True)
                                                                           # [number_of_sampled_sequences]

        tmp_time_history_for_sampling, _ = pack([time_history_for_sampling, sampled_time], 'nss *')
                                                                           # [number_of_sampled_sequences, history_length + 1]
        tmp_events_history_for_sampling, _ = pack([events_history_for_sampling, sampled_marks], 'nss *')
                                                                           # [number_of_sampled_sequences, history_length + 1]
        tmp_sum_of_sampled_time = tmp_time_history_for_sampling.sum(dim = -1)
                                                                           # [number_of_sampled_sequences]
        seq_length += 1

        if tmp_sum_of_sampled_time.min() >= end_time:
            break
        else:
            events_history_for_sampling = tmp_events_history_for_sampling  # [number_of_sampled_sequences, new_length]
            time_history_for_sampling = tmp_time_history_for_sampling      # [number_of_sampled_sequences, new_length]

    sampled_mask = (time_history_for_sampling.cumsum(dim = -1) < end_time).int()
                                                                           # [number_of_sampled_sequences, sampled_sequences_length]

    return time_history_for_sampling, events_history_for_sampling, sampled_mask


def sample_event_time(self, number_of_sampled_sequences, end_time, mean, std):
    '''
    These two functions will sample a event sequence from the learned p^*(m, t) following the event-time prediction procedure.
    Steps:
    1. Sample the mark \\(m_p\\) from p^*(m) = \\int_{t_l}^{+\\infty}{p^*(m, \\tau)d\\tau}.
    2. Sample when a new \\(m_p\\) event would happen in the future time by \\(p^*(t|m_p)\\).
    '''
    time_history_for_sampling = torch.zeros((number_of_sampled_sequences, 1), device = self.device)
                                                                           # [number_of_sampled_sequences, 1]
    events_history_for_sampling = torch.ones((number_of_sampled_sequences, 1), device = self.device, dtype = torch.int32) * self.num_events
                                                                           # [number_of_sampled_sequences, 1]
    tmp_sum_of_sampled_time = time_history_for_sampling.sum(dim = -1)      # [number_of_sampled_sequences]

    MAX_sampled_seq = 250
    seq_length = 1

    while seq_length < MAX_sampled_seq:
        time_next_zero = torch.zeros(number_of_sampled_sequences, self.num_events, device = self.device)
                                                                           # [number_of_sampled_sequences, num_events]
        integral_from_zero_to_inf = self.model('sample', events_history_for_sampling, time_history_for_sampling, time_next_zero, mean = mean, std = std)
                                                                           # [number_of_sampled_sequences, num_events]
        sampled_marks = predict_event(integral_from_zero_to_inf, sample = True)
                                                                           # [number_of_sampled_sequences]

        mask_history_for_sampling = torch.ones_like(time_history_for_sampling)
        all_sampled_time = self.sample_time('its', 'mt', 
                                            events_history_for_sampling, time_history_for_sampling, mask_history_for_sampling, 
                                            integral_from_zero_to_inf, number_of_sampled_sequences, number_of_sampled_sequences, 1e6, mean, std, 
                                            autoregressive = True)         # [number_of_sampled_sequences, num_events]
        one_hot_mask_of_sampled_marks = torch.nn.functional.one_hot(sampled_marks, num_classes = self.num_events)
                                                                           # [number_of_sampled_sequences, num_events]
        sampled_time = torch.sum(all_sampled_time * one_hot_mask_of_sampled_marks, dim = -1)
                                                                           # [number_of_sampled_sequences, 1]
        
        tmp_events_history_for_sampling, _ = pack([events_history_for_sampling, sampled_marks], 'nss *')
                                                                           # [number_of_sampled_sequences, history_length + 1]
        tmp_time_history_for_sampling, _ = pack([time_history_for_sampling, sampled_time], 'nss *')
                                                                           # [number_of_sampled_sequences, history_length + 1]
        tmp_sum_of_sampled_time = tmp_time_history_for_sampling.sum(dim = -1)
                                                                           # [number_of_sampled_sequences, 1]
        seq_length += 1

        if tmp_sum_of_sampled_time.min() >= end_time:
            break
        else:
            events_history_for_sampling = tmp_events_history_for_sampling  # [number_of_sampled_sequences, new_length]
            time_history_for_sampling = tmp_time_history_for_sampling      # [number_of_sampled_sequences, new_length]

    sampled_mask = (time_history_for_sampling.cumsum(dim = -1) < end_time).int()
                                                                           # [number_of_sampled_sequences, new_length]

    return time_history_for_sampling, events_history_for_sampling, sampled_mask