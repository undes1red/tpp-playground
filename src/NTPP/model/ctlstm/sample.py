import torch
from einops import rearrange, repeat, reduce, pack

from src.NTPP.model.basic_tpp_model import its_lower_bound, its_upper_bound
from src.NTPP.model.utils import step_split, median_prediction, thinning_sampling, predict_event

from src.toolbox.integration import approximate_integration
from src.toolbox.misc import check_should_we_stop_sampling, conditional_compile_class_method


@torch.inference_mode()
@conditional_compile_class_method
def sample_time(self, sampling_approach = 'its', task = 'mt', autoregressive = False, *args, **kwargs):
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
    if autoregressive:
        dict_sampling_apparoch = {
            'its': autoregressive_sampling_by_its,
            'thinning': autoregressive_sampling_by_thinning
        }
    else:
        dict_sampling_apparoch = {
            'its': sampling_by_its,
            'thinning': sampling_by_thinning
        }

    return dict_sampling_apparoch[sampling_approach](self, task = task, *args, **kwargs)


# Sample events from p^*(m, t) using inversed transform sampling in a autoregressive manner.
def autoregressive_sampling_by_its(self, task, *args, **kwargs):
    dict_apparoch_for_tasks = {
        'mt': autoregressive_sampling_by_its_for_mt,
        'tm': autoregressive_sampling_by_its_for_tm
    }

    return dict_apparoch_for_tasks[task](self, *args, **kwargs)


def autoregressive_sampling_by_its_for_mt(self, events_history, time_history, p_m, resolution,
                                          number_of_total_samples, step, inf_val, mean, std):
    # Preprocess
    sample_rate_list = step_split(number_of_total_samples, step)

    def evaluate_all_event(taus):
        expanded_integral_across_events, expanded_intensity_across_events, timestamp = \
            self.model.integral_intensity_time_next_3d(events_history, time_history, taus, resolution, num_dimension_prior_batch = 1)
                                                                            # 2 * [sample_rate, batch_size, seq_len, num_events, resolution, num_events] + [sample_rate, batch_size, seq_len, num_events, resolution]
        expanded_integral_sum_across_events = expanded_integral_across_events.sum(dim = -1)
                                                                            # [sample_rate, batch_size, seq_len, num_events, resolution]
        intensity_event_mask = torch.diag(torch.ones(self.num_events, device = self.device))
                                                                            # [num_events, num_events]
        intensity_event_mask = rearrange(intensity_event_mask, f'ne ne1 -> {"() " * (len(expanded_intensity_across_events.shape) - 3)}ne () ne1')
                                                                            # [sample_rate, batch_size, seq_len, num_events, resolution, num_events]
        expanded_intensity_per_event = (expanded_intensity_across_events * intensity_event_mask).sum(dim = -1)
                                                                            # [sample_rate, batch_size, seq_len, num_events, resolution]
        expanded_probability_per_event = expanded_intensity_per_event * torch.exp(-expanded_integral_sum_across_events)
                                                                            # [sample_rate, batch_size, seq_len, num_events, resolution]
        probability = approximate_integration(expanded_probability_per_event, timestamp, dim = -1, only_integral = True)
                                                                            # [sample_rate, batch_size, seq_len, num_events]
        return probability

    def bisect_target(taus, probability_threshold):
        p_mt = evaluate_all_event(taus)                                    # [sample_rate, batch_size, seq_len, num_events]
        p_t_m = p_mt / p_m                                                 # [sample_rate, batch_size, seq_len, num_events]
        p_gap = p_t_m - probability_threshold                              # [sample_rate, batch_size, seq_len, num_events]

        return p_gap

    tau_pred = []
    batch_size, seq_len = time_history.shape
    p_m = p_m.unsqueeze(dim = 0)                                           # [1, batch_size, seq_len, num_events]
    for sub_sample_rate in sample_rate_list:
        probability_threshold = torch.zeros((sub_sample_rate, batch_size, seq_len, self.num_events), device = self.device)
                                                                            # [sample_rate, batch_size, seq_len, num_events]
        torch.nn.init.uniform_(probability_threshold, a = its_lower_bound, b = its_upper_bound)
                                                                            # [sample_rate, batch_size, seq_len, num_events]
        tau_pred.append(median_prediction(self.max_step, self.bisect_early_stop_threshold, \
                                            bisect_target, probability_threshold, r_val = inf_val))
                                                                            # [sample_rate, batch_size, seq_len, num_events]
    tau_pred = torch.cat(tau_pred, dim = 0)                                # [sample_rate, batch_size, seq_len, num_events]
    
    return tau_pred


def autoregressive_sampling_by_its_for_tm(self, events_history, time_history,
                                          number_of_total_samples, step, mean, std):
    sample_rate_list = step_split(number_of_total_samples, step)

    def bisect_target(taus, probability_threshold):
        '''
        Args:
        1. time: the sequence containing events' timestamps. shape: [batch_size, seq_len + 1]
        2. events: the sequence containing information about events. shape: [batch_size, seq_len + 1]
        3. mask: the padding mask introduced by the dataloader. shape: [batch_size, seq_len + 1]
        '''
        expanded_integral_all_events, _, = \
            self.model.sample_for_tm(time_history, taus, events_history)   # [number_of_sampled_sequences, num_events]
        expanded_integral = expanded_integral_all_events.sum(dim = -1)     # [number_of_sampled_sequences]

        return expanded_integral + torch.log(1 - probability_threshold)

    tau_pred = []
    for sub_sample_rate in sample_rate_list:
        probability_threshold = torch.zeros((sub_sample_rate), device = self.device)
                                                                            # [sub_sample_rate]
        torch.nn.init.uniform_(probability_threshold, a = its_lower_bound, b = its_upper_bound)
                                                                            # [sub_sample_rate]
        tau_pred.append(median_prediction(self.max_step, self.bisect_early_stop_threshold, \
                                            bisect_target, probability_threshold))
                                                                            # [sub_sample_rate]
    tau_pred = torch.cat(tau_pred, dim = 0)                                 # [sample_rate]

    return tau_pred


def sampling_by_its(self, task, *args, **kwargs):
    dict_apparoch_for_tasks = {
        'mt': sampling_by_its_for_mt,
        'tm': sampling_by_its_for_tm
    }

    return dict_apparoch_for_tasks[task](self, *args, **kwargs)


def sampling_by_its_for_mt(self, events_history, time_history, mask_history, p_m, resolution,
                           number_of_total_samples, step, inf_val, mean, std, note_embedding_history = None, note_embedding_next = None):
    # Preprocess
    sample_rate_list = step_split(number_of_total_samples, step)

    def evaluate_all_event(taus):
        expanded_integral_across_events, expanded_intensity_across_events, timestamp = \
            self.model.integral_intensity_time_next_3d(events_history, time_history, taus, mask_history, resolution, num_dimension_prior_batch = 1, \
                                                       note_embedding_history = note_embedding_history, note_embedding_next = note_embedding_next)
                                                                            # 2 * [sample_rate, batch_size, seq_len, num_events, resolution, num_events] + [sample_rate, batch_size, seq_len, num_events, resolution]
        expanded_integral_sum_across_events = expanded_integral_across_events.sum(dim = -1)
                                                                            # [sample_rate, batch_size, seq_len, num_events, resolution]
        intensity_event_mask = torch.diag(torch.ones(self.num_events, device = self.device))
                                                                            # [num_events, num_events]
        intensity_event_mask = rearrange(intensity_event_mask, f'ne ne1 -> {"() " * (len(expanded_intensity_across_events.shape) - 3)}ne () ne1')
                                                                            # [sample_rate, batch_size, seq_len, num_events, resolution, num_events]
        expanded_intensity_per_event = (expanded_intensity_across_events * intensity_event_mask).sum(dim = -1)
                                                                            # [sample_rate, batch_size, seq_len, num_events, resolution]
        expanded_probability_per_event = expanded_intensity_per_event * torch.exp(-expanded_integral_sum_across_events)
                                                                            # [sample_rate, batch_size, seq_len, num_events, resolution]
        probability = approximate_integration(expanded_probability_per_event, timestamp, dim = -1, only_integral = True)
                                                                            # [sample_rate, batch_size, seq_len, num_events]
        return probability

    def bisect_target(taus, probability_threshold):
        p_mt = evaluate_all_event(taus)                                    # [sample_rate, batch_size, seq_len, num_events]
        p_t_m = p_mt / p_m                                                 # [sample_rate, batch_size, seq_len, num_events]
        p_gap = p_t_m - probability_threshold                              # [sample_rate, batch_size, seq_len, num_events]

        return p_gap

    tau_pred = []
    batch_size, seq_len = time_history.shape
    p_m = p_m.unsqueeze(dim = 0)                                           # [1, batch_size, seq_len, num_events]
    for sub_sample_rate in sample_rate_list:
        probability_threshold = torch.zeros((sub_sample_rate, batch_size, seq_len, self.num_events), device = self.device)
                                                                            # [sample_rate, batch_size, seq_len, num_events]
        torch.nn.init.uniform_(probability_threshold, a = its_lower_bound, b = its_upper_bound)
                                                                            # [sample_rate, batch_size, seq_len, num_events]
        tau_pred.append(median_prediction(self.max_step, self.bisect_early_stop_threshold, \
                                            bisect_target, probability_threshold, r_val = inf_val))
                                                                            # [sample_rate, batch_size, seq_len, num_events]
    tau_pred = torch.cat(tau_pred, dim = 0)                                # [sample_rate, batch_size, seq_len, num_events]
    
    return tau_pred


def sampling_by_its_for_tm(self, events_history, time_history, mask_history, 
                           number_of_total_samples, step, mean, std, note_embedding_history = None, note_embedding_next = None):
    sample_rate_list = step_split(number_of_total_samples, step)

    def bisect_target(taus, probability_threshold):
        '''
        Args:
        1. time: the sequence containing events' timestamps. shape: [batch_size, seq_len + 1]
        2. events: the sequence containing information about events. shape: [batch_size, seq_len + 1]
        3. mask: the padding mask introduced by the dataloader. shape: [batch_size, seq_len + 1]
        '''
        expanded_integral_all_events, _, = \
            self.model(time_history, taus, events_history, mask_history, num_dimension_prior_batch = 1, \
                       note_embedding_history = note_embedding_history, note_embedding_next = note_embedding_next)
                                                                               # [sample_rate, batch_size, seq_len, num_events]
        expanded_integral = expanded_integral_all_events.sum(dim = -1)         # [sample_rate, batch_size, seq_len]

        return expanded_integral + torch.log(1 - probability_threshold)

    tau_pred = []
    for sub_sample_rate in sample_rate_list:
        probability_threshold = torch.zeros((sub_sample_rate, *time_history.shape), device = self.device)
                                                                               # [sample_rate, batch_size, seq_len]
        torch.nn.init.uniform_(probability_threshold, a = its_lower_bound, b = its_upper_bound)
                                                                               # [sample_rate, batch_size, seq_len]
        tau_pred.append(median_prediction(self.max_step, self.bisect_early_stop_threshold, \
                                            bisect_target, probability_threshold))
                                                                               # [sample_rate, batch_size, seq_len]
    tau_pred = torch.cat(tau_pred, dim = 0)                                    # [sample_rate, batch_size, seq_len]

    return tau_pred


# Sample events from p^*(m, t) using thinning algorithm in a autoregressive manner.
def autoregressive_sampling_by_thinning(self, task, *args, **kwargs):
    dict_apparoch_for_tasks = {
        'mt': autoregressive_sampling_by_thinning_for_mt,
        'tm': autoregressive_sampling_by_thinning_for_tm
    }

    return dict_apparoch_for_tasks[task](self, *args, **kwargs)


def autoregressive_sampling_by_thinning_for_mt(self):
    pass


def autoregressive_sampling_by_thinning_for_tm(self):
    pass


def sampling_by_thinning(self, task, *args, **kwargs):
    dict_apparoch_for_tasks = {
        'mt': sampling_by_thinning_for_mt,
        'tm': sampling_by_thinning_for_tm
    }

    return dict_apparoch_for_tasks[task](self, *args, **kwargs)


def sampling_by_thinning_for_mt(self, *args, **kwargs):
    raise Exception('Thinning algorithm can not solve task MT. Please use ITS by setting sampling_approach = its.')


def sampling_by_thinning_for_tm(self, events_history, time_history, number_of_total_samples, step, mean, std):
    sample_rate_list = step_split(number_of_total_samples, step)
    batch_size, seq_len = time_history.shape
    maximum_thinning_loops = 50
    max_sample_time_limit = mean + 10 * std

    def get_intensity(tau, time_history, events_history):
        return self.model(time_history, tau, events_history, num_dimension_prior_batch = 1)[-1].sum(dim = -1)
    
    def find_maximum_intensity_values_in_one_interval(interval_left, interval_right, time_history, events_history):
        _, intensity_between_interval_left_and_right, _ \
            = self.model.integral_intensity_time_next_2d(events_history, time_history, interval_right, \
                                                            self.integration_sample_rate, num_dimension_prior_batch = 1, time_next_start = interval_left)
                                                                            # [sample_rate, batch_size, seq_len, integration_sample_rate, num_events]
        intensity_between_interval_left_and_right = intensity_between_interval_left_and_right.sum(dim = -1)
                                                                            # [sample_rate, batch_size, seq_len, integration_sample_rate]

        return intensity_between_interval_left_and_right.max(dim = -1)[0]
    
    sampled_time = []
    for each_step in sample_rate_list:
        sampled_time.append(thinning_sampling(maximum_thinning_loops, max_sample_time_limit, (each_step, batch_size, seq_len), self.device, \
                                                get_intensity, find_maximum_intensity_values_in_one_interval, time_history, events_history))
                                                                            # [sample_rate, batch_size, seq_len]
    
    sampled_time = torch.cat(sampled_time, dim = 0)
    return sampled_time


def sample_time_event(self, time_history_for_sampling, events_history_for_sampling, mean, std, \
                      end_sampling_requirement = 'time', **kwargs):
    '''
    This function will sample x sequences by the learned probability distribution following the time-event prediction procedure.
    Steps:
    1. Sample a time \\(t_s\\) from p^*(t) = \\sum{n \\in M}{p^*(m, t)} referring to existing history
    2. Judge the mark of this event by comparing \\(\\lambda^*(m, t_s)\\).
    '''
    if time_history_for_sampling is None and events_history_for_sampling is None:
        number_of_sampled_sequences = kwargs['number_of_sampled_sequences']
        time_history_for_sampling = torch.zeros((number_of_sampled_sequences, 1), device = self.device)
                                                                               # [number_of_sampled_sequences, 1]
        events_history_for_sampling = torch.ones((number_of_sampled_sequences, 1), device = self.device, dtype = torch.int32) * self.num_events
                                                                               # [number_of_sampled_sequences, 1]
    else:
        assert time_history_for_sampling is not None and events_history_for_sampling is not None, 'How is it possible that one input history is not None while another one is?'
        assert events_history_for_sampling.shape[0] == time_history_for_sampling.shape[0], f'time_history_for_sampling says we will sample {time_history_for_sampling.shape[0]} sequences, while events_history_for_sampling suggests {events_history_for_sampling.shape[0]}. So, how many sequences should we sample?'
        number_of_sampled_sequences = events_history_for_sampling.shape[0]
        
    sampled_mask = None
    
    while True:
        should_we_stop, sampled_mask = \
            check_should_we_stop_sampling(time_history_for_sampling, end_sampling_requirement, **kwargs)
        
        if should_we_stop:
            break
        
        sampled_time = self.sample_time(sampling_approach = 'its', task = 'tm', autoregressive = True,
                                        events_history = events_history_for_sampling,
                                        time_history = time_history_for_sampling,
                                        number_of_total_samples = number_of_sampled_sequences,
                                        step = number_of_sampled_sequences, mean = mean, std = std)
                                                                               # [number_of_sampled_sequences]
        _, intensity_all_events = \
            self.model.sample_for_tm(time_history_for_sampling, sampled_time, events_history_for_sampling)
                                                                               # [number_of_sampled_sequences]
        sampled_marks = predict_event(intensity_all_events, sample = True)     # [number_of_sampled_sequences]

        time_history_for_sampling, _ = pack([time_history_for_sampling, sampled_time], 'nss *')
                                                                               # [number_of_sampled_sequences, history_length + 1]
        events_history_for_sampling, _ = pack([events_history_for_sampling, sampled_marks], 'nss *')
                                                                               # [number_of_sampled_sequences, history_length + 1]


    return time_history_for_sampling, events_history_for_sampling, sampled_mask


def sample_event_time(self, time_history_for_sampling, events_history_for_sampling, mean, std, \
                      end_sampling_requirement = 'time', **kwargs):
    '''
    These two functions will sample a event sequence from the learned p^*(m, t) following the event-time prediction procedure.
    Steps:
    1. Sample the mark \\(m_p\\) from p^*(m) = \\int_{t_l}^{+\\infty}{p^*(m, \\tau)d\\tau}.
    2. Sample when a new \\(m_p\\) event would happen in the future time by \\(p^*(t|m_p)\\).
    '''
    if time_history_for_sampling is None and events_history_for_sampling is None:
        number_of_sampled_sequences = kwargs['number_of_sampled_sequences']
        time_history_for_sampling = torch.zeros((number_of_sampled_sequences, 1), device = self.device)
                                                                            # [number_of_sampled_sequences, 1]
        events_history_for_sampling = torch.ones((number_of_sampled_sequences, 1), device = self.device, dtype = torch.int32) * self.num_events
                                                                            # [number_of_sampled_sequences, 1]
    else:
        assert time_history_for_sampling is not None and events_history_for_sampling is not None, 'How is it possible that one history is not None while another one is?'
        assert events_history_for_sampling.shape[0] == time_history_for_sampling.shape[0], f'time_history_for_sampling says we will sample {time_history_for_sampling.shape[0]} sequences, while events_history_for_sampling suggests {events_history_for_sampling.shape[0]}. So, how many sequences should we sample?'
        number_of_sampled_sequences = events_history_for_sampling.shape[0]

    sampled_mask = None

    while True:
        should_we_stop, sampled_mask = \
            check_should_we_stop_sampling(time_history_for_sampling, end_sampling_requirement, **kwargs)

        if should_we_stop:
            break

        time_next_zero = torch.zeros(number_of_sampled_sequences, self.num_events, device = self.device)
                                                                            # [number_of_sampled_sequences, num_events]
        integral_from_zero_to_inf = self.model('sample', events_history_for_sampling, time_history_for_sampling, time_next_zero, mean = mean, std = std)
                                                                            # [number_of_sampled_sequences, num_events]
        sampled_marks = predict_event(integral_from_zero_to_inf, sample = True)
                                                                            # [number_of_sampled_sequences]
        all_sampled_time = self.sample_time('its', 'mt', True,
                                            events_history_for_sampling, time_history_for_sampling, integral_from_zero_to_inf,
                                            number_of_sampled_sequences, number_of_sampled_sequences, 1e6, mean, std)
                                                                            # [number_of_sampled_sequences, num_events]
        one_hot_mask_of_sampled_marks = torch.nn.functional.one_hot(sampled_marks, num_classes = self.num_events)
                                                                            # [number_of_sampled_sequences, num_events]
        sampled_time = torch.sum(all_sampled_time * one_hot_mask_of_sampled_marks, dim = -1)
                                                                            # [number_of_sampled_sequences, 1]

        events_history_for_sampling, _ = pack([events_history_for_sampling, sampled_marks], 'nss *')
                                                                            # [number_of_sampled_sequences, history_length + 1]
        time_history_for_sampling, _ = pack([time_history_for_sampling, sampled_time], 'nss *')
                                                                            # [number_of_sampled_sequences, history_length + 1]
                                                                            
    return time_history_for_sampling, events_history_for_sampling, sampled_mask