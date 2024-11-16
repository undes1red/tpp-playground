import torch
from einops import rearrange, repeat, reduce, pack

from src.toolbox.misc import check_should_we_stop_sampling

from src.TPP.model.basic_tpp_model import its_lower_bound, its_upper_bound
from src.TPP.model.utils import *


def sample_time(self, sampling_approach = 'its', task = 'mt', autoregressive = False, *args, **kwargs):
    '''
    number_of_total_samples: how many samples do we need to predict one next event.
    step: we output "step" samples to reduce memory comsumption during inference.
    sampling_approach: 'its' for invert transform sampling and 'thinning' for thinning algorithm.
    task: 'mt' for mark first time second, 'tm' for time first mark second.
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

    return dict_sampling_apparoch[sampling_approach](self, task, *args, **kwargs)

# Sample events from p^*(m, t) using inversed transform sampling in a autoregressive manner.
def autoregressive_sampling_by_its(self, task, *args, **kwargs):
    dict_apparoch_for_tasks = {
        'mt': autoregressive_sampling_by_its_for_mt,
        'tm': autoregressive_sampling_by_its_for_tm
    }

    return dict_apparoch_for_tasks[task](self, *args, **kwargs)

# Sample events from p^*(m, t) using thinning algorithm in a autoregressive manner.
def autoregressive_sampling_by_thinning(self, task, *args, **kwargs):
    dict_apparoch_for_tasks = {
        'mt': autoregressive_sampling_by_thinning_for_mt,
        'tm': autoregressive_sampling_by_thinning_for_tm
    }

    return dict_apparoch_for_tasks[task](self, *args, **kwargs)

# Sample events from p^*(m, t) using inversed transform sampling.
def sampling_by_its(self, task, *args, **kwargs):
    dict_apparoch_for_tasks = {
        'mt': sampling_by_its_for_mt,
        'tm': sampling_by_its_for_tm
    }

    return dict_apparoch_for_tasks[task](self, *args, **kwargs)

# Sample events from p^*(m, t) using thinning algorithm.
def sampling_by_thinning(self, task, *args, **kwargs):
    dict_apparoch_for_tasks = {
        'mt': sampling_by_thinning_for_mt,
        'tm': sampling_by_thinning_for_tm
    }

    return dict_apparoch_for_tasks[task](self, *args, **kwargs)


# For autoregressive_sampling_by_its.
def autoregressive_sampling_by_its_for_mt(self, events_history, time_history, p_m,
                                          number_of_total_samples, step, inf_val, mean, std):
    # Preprocess
    sample_rate_list = step_split(number_of_total_samples, step)

    def bisect_target(taus, probability_threshold):
        # \\int_{tau}^{+\\inf}{p(m, \\tau|\\mathcal{H})d\\tau}
        probability_integral_from_t_to_infinite = self.model('sample', events_history, time_history, taus, mean = mean, std = std)
                                                                               # [sample_rate, num_events]
        # \\int_{0}^{tau}{p(m, \\tau|\\mathcal{H})d\\tau}
        p_mt = p_m - probability_integral_from_t_to_infinite                   # [sample_rate, num_events]
        p_t_m = p_mt / p_m                                                     # [sample_rate, num_events]
        p_gap = p_t_m - probability_threshold                                  # [sample_rate, num_events]

        return p_gap
        
    # Preprocess
    tau_pred = []
    for sub_sample_rate in sample_rate_list:
        probability_threshold = torch.zeros((sub_sample_rate, self.num_events), device = self.device)
                                                                               # [sub_sample_rate, num_events]
        torch.nn.init.uniform_(probability_threshold, a = its_lower_bound, b = its_upper_bound)
                                                                               # [sub_sample_rate, num_events]
        tau_pred.append(median_prediction(self.max_step, self.bisect_early_stop_threshold, \
                                          bisect_target, probability_threshold, r_val = inf_val))
                                                                               # [sub_sample_rate, num_events]

    tau_pred = torch.cat(tau_pred, dim = 0)                                    # [sample_rate, num_events]

    return tau_pred


def autoregressive_sampling_by_its_for_tm(self, events_history, time_history,
                                          number_of_total_samples, step, mean, std):
    # Preprocess
    sample_rate_list = step_split(number_of_total_samples, step)

    def evaluate(taus, probability_threshold, integral_from_zero_to_inf):
        taus = repeat(taus, '... -> ... ne', ne = self.num_events)             # [..., num_events]
        probability_integral_from_t_to_inf = self.model('sample', events_history, time_history, taus, mean, std)
                                                                               # [sample_rate, num_events]
        # P_m(t) = \\int_{0}^{t}{p(t|m, \\mathcal{H})}
        probability_integral = integral_from_zero_to_inf - probability_integral_from_t_to_inf
                                                                               # [sample_rate, num_events]
        probability_integral = torch.sum(probability_integral, dim = -1)       # [sample_rate]
        
        return probability_integral - probability_threshold

    tau_pred = []
    for sub_sample_rate in sample_rate_list:
        probability_threshold = torch.zeros(sub_sample_rate, device = self.device)
                                                                               # [sub_sample_rate]
        torch.nn.init.uniform_(probability_threshold, a = its_lower_bound, b = its_upper_bound)
                                                                               # [sub_sample_rate]

        time_next_zero = torch.zeros_like(probability_threshold)               # [sub_sample_rate]
        time_next_zero = repeat(time_next_zero, '... -> ... ne', ne = self.num_events)
                                                                               # [sub_sample_rate, num_events]
        integral_from_zero_to_inf = self.model('sample', events_history, time_history, time_next_zero, mean = mean, std = std)
                                                                               # [sub_sample_rate, num_events]

        tau_pred.append(median_prediction(self.max_step, self.bisect_early_stop_threshold, \
                                          evaluate, probability_threshold, integral_from_zero_to_inf))
                                                                               # [sub_sample_rate]
    tau_pred = torch.cat(tau_pred, dim = 0)                                    # [sample_rate]

    return tau_pred


# For autoregressive_sampling_by_thinning.
def autoregressive_sampling_by_thinning_for_mt(self, *args, **kwargs):
    raise Exception('Thinning algorithm can not solve task MT. Please use ITS by setting sampling_approach = its.')


def autoregressive_sampling_by_thinning_for_tm(self, events_history, time_history, mask_history, number_of_total_samples, step, mean, std):
    raise Exception('IFIB does not know intensity functions, which thinning algorithm requires. Please use ITS by setting sampling_approach = its.')


# For sampling_by_its.
def sampling_by_its_for_mt(self, events_history, time_history, p_m,
                           number_of_total_samples, step, inf_val, mean, std):
    # Preprocess
    sample_rate_list = step_split(number_of_total_samples, step)

    def bisect_target(taus, probability_threshold):
        # \\int_{tau}^{+\\inf}{p(m, \\tau|\\mathcal{H})d\\tau}
        probability_integral_from_t_to_infinite = self.model('default_forward', events_history, time_history, taus, mean = mean, std = std)
                                                                               # [sample_rate, batch_size, seq_len, num_events]
        # \\int_{0}^{tau}{p(m, \\tau|\\mathcal{H})d\\tau}
        p_mt = p_m - probability_integral_from_t_to_infinite                   # [sample_rate, batch_size, seq_len, num_events]
        p_t_m = p_mt / p_m                                                     # [sample_rate, batch_size, seq_len, num_events]
        p_gap = p_t_m - probability_threshold                                  # [sample_rate, batch_size, seq_len, num_events]

        return p_gap
        
    # Preprocess
    tau_pred = []
    batch_size, seq_len = time_history.shape
    for sub_sample_rate in sample_rate_list:
        probability_threshold = torch.zeros((sub_sample_rate, batch_size, seq_len, self.num_events), device = self.device)
                                                                               # [sub_sample_rate, batch_size, seq_len, num_events]
        torch.nn.init.uniform_(probability_threshold, a = its_lower_bound, b = its_upper_bound)
                                                                               # [sub_sample_rate, batch_size, seq_len, num_events]
        tau_pred.append(median_prediction(self.max_step, self.bisect_early_stop_threshold, \
                                          bisect_target, probability_threshold, r_val = inf_val))
                                                                               # [sub_sample_rate, batch_size, seq_len, num_events]

    tau_pred = torch.cat(tau_pred, dim = 0)                                    # [sample_rate, batch_size, seq_len, num_events]

    return tau_pred


def sampling_by_its_for_tm(self, events_history, time_history,
                           number_of_total_samples, step, mean, std):
    # Preprocess
    sample_rate_list = step_split(number_of_total_samples, step)

    def evaluate(taus, probability_threshold, integral_from_zero_to_inf):
        taus = repeat(taus, '... -> ... ne', ne = self.num_events)             # [..., num_events]
        probability_integral_from_t_to_inf = self.model('default_forward', events_history, time_history, taus, mean, std)
                                                                               # [sample_rate, batch_size, seq_len, num_events]
        # P_m(t) = \\int_{0}^{t}{p(t|m, \\mathcal{H})}
        probability_integral = integral_from_zero_to_inf - probability_integral_from_t_to_inf
                                                                               # [sample_rate, batch_size, seq_len, num_events]
        probability_integral = torch.sum(probability_integral, dim = -1)       # [sample_rate, batch_size, seq_len]
        
        return probability_integral - probability_threshold

    tau_pred = []
    batch_size, seq_len = time_history.shape
    for sub_sample_rate in sample_rate_list:
        probability_threshold = torch.zeros((sub_sample_rate, batch_size, seq_len), device = self.device)
                                                                               # [sub_sample_rate, batch_size, seq_len]
        torch.nn.init.uniform_(probability_threshold, a = its_lower_bound, b = its_upper_bound)
                                                                               # [sub_sample_rate, batch_size, seq_len]

        time_next_zero = torch.zeros_like(probability_threshold)               # [sub_sample_rate, batch_size, seq_len]
        time_next_zero = repeat(time_next_zero, '... -> ... ne', ne = self.num_events)
                                                                               # [sub_sample_rate, batch_size, seq_len, num_events]
        integral_from_zero_to_inf = self.model('default_forward', events_history, time_history, time_next_zero, mean = mean, std = std)
                                                                               # [sub_sample_rate, batch_size, seq_len, num_events]

        tau_pred.append(median_prediction(self.max_step, self.bisect_early_stop_threshold, \
                                              evaluate, probability_threshold, integral_from_zero_to_inf))
                                                                               # [sub_sample_rate, batch_size, seq_len]
    tau_pred = torch.cat(tau_pred, dim = 0)                                    # [sample_rate, batch_size, seq_len]

    return tau_pred


# For sampling_by_thinning.
def sampling_by_thinning_for_mt(self, *args, **kwargs):
    raise Exception('Thinning algorithm can not solve task MT. Please use ITS by setting sampling_approach = its.')


def sampling_by_thinning_for_tm(self, events_history, time_history, mask_history, number_of_total_samples, step, mean, std):
    raise Exception('IFIB does not know intensity functions, which thinning algorithm requires. Please use ITS by setting sampling_approach = its.')


# For autoregressive sampling.
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
        sampled_time = self.sample_time('its', 'tm', True,
                                        events_history_for_sampling, time_history_for_sampling,
                                        number_of_sampled_sequences, number_of_sampled_sequences, mean, std)
                                                                               # [number_of_sampled_sequences]
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
        )[0]                                                                   # [number_of_sampled_sequences, num_events]
        repeated_sampled_time.requires_grad = False

        sampled_marks = predict_event(probability_for_each_event_at_pred_time, sample = True)
                                                                               # [number_of_sampled_sequences]

        tmp_events_history_for_sampling, _ = pack([events_history_for_sampling, sampled_marks], 'nss *')
                                                                               # [number_of_sampled_sequences, history_length + 1]
        tmp_time_history_for_sampling, _ = pack([time_history_for_sampling, sampled_time], 'nss *')
                                                                               # [number_of_sampled_sequences, history_length + 1]

        should_we_stop, sampled_mask = \
            check_should_we_stop_sampling(tmp_time_history_for_sampling, end_sampling_requirement, **kwargs)
        
        if should_we_stop:
            # Remove the mask of the temporarily added event.
            sampled_mask = sampled_mask[..., :-1]
            break

        events_history_for_sampling = tmp_events_history_for_sampling          # [number_of_sampled_sequences, history_length + 1]
        time_history_for_sampling = tmp_time_history_for_sampling              # [number_of_sampled_sequences, history_length + 1]


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

        tmp_events_history_for_sampling, _ = pack([events_history_for_sampling, sampled_marks], 'nss *')
                                                                               # [number_of_sampled_sequences, history_length + 1]
        tmp_time_history_for_sampling, _ = pack([time_history_for_sampling, sampled_time], 'nss *')
                                                                               # [number_of_sampled_sequences, history_length + 1]

        should_we_stop, sampled_mask = \
            check_should_we_stop_sampling(tmp_time_history_for_sampling, end_sampling_requirement, **kwargs)

        if should_we_stop:
            # Remove the mask of the temporarily added event.
            sampled_mask = sampled_mask[..., :-1]
            break

        events_history_for_sampling = tmp_events_history_for_sampling          # [number_of_sampled_sequences, history_length + 1]
        time_history_for_sampling = tmp_time_history_for_sampling              # [number_of_sampled_sequences, history_length + 1]
                                                                            
    return time_history_for_sampling, events_history_for_sampling, sampled_mask