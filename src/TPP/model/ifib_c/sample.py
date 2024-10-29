import torch
from einops import rearrange, repeat, reduce, pack

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


# Sample events from p^*(m, t) using inversed transform sampling.
def sampling_by_its(self, task, *args, **kwargs):
    dict_apparoch_for_tasks = {
        'mt': sampling_by_its_for_mt,
        'tm': sampling_by_its_for_tm
    }

    return dict_apparoch_for_tasks[task](self, *args, **kwargs)


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


# Sample events from p^*(m, t) using thinning algorithm in a autoregressive manner.
def autoregressive_sampling_by_thinning(self, task, *args, **kwargs):
    dict_apparoch_for_tasks = {
        'mt': autoregressive_sampling_by_thinning_for_mt,
        'tm': autoregressive_sampling_by_thinning_for_tm
    }

    return dict_apparoch_for_tasks[task](self, *args, **kwargs)


def autoregressive_sampling_by_thinning_for_mt(self, *args, **kwargs):
    raise Exception('Thinning algorithm can not solve task MT. Please use ITS by setting sampling_approach = its.')


def autoregressive_sampling_by_thinning_for_tm(self, events_history, time_history, mask_history, number_of_total_samples, step, mean, std):
    raise Exception('IFIB does not know intensity functions, which thinning algorithm requires. Please use ITS by setting sampling_approach = its.')


# Sample events from p^*(m, t) using thinning algorithm.
def sampling_by_thinning(self, task, *args, **kwargs):
    dict_apparoch_for_tasks = {
        'mt': sampling_by_thinning_for_mt,
        'tm': sampling_by_thinning_for_tm
    }

    return dict_apparoch_for_tasks[task](self, *args, **kwargs)
    

def sampling_by_thinning_for_mt(self, *args, **kwargs):
    raise Exception('Thinning algorithm can not solve task MT. Please use ITS by setting sampling_approach = its.')


def sampling_by_thinning_for_tm(self, events_history, time_history, mask_history, number_of_total_samples, step, mean, std):
    raise Exception('IFIB does not know intensity functions, which thinning algorithm requires. Please use ITS by setting sampling_approach = its.')