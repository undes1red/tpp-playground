import torch

from src.tpp.tpp_models.basic_tpp_model import its_lower_bound, its_upper_bound
from src.tpp.tpp_models.utils import *

@torch.inference_mode()
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


def sampling_by_its_for_tm(self, input_events, input_time, input_mask, number_of_total_samples, step, mean, std):
    # Preprocess
    sample_rate_list = step_split(number_of_total_samples, step)

    def bisect_target(taus, probability_threshold):
        taus = repeat(taus, '... -> ... ne', ne = self.num_events + 1)         # [sub_sample_rate, batch_size, seq_len + 1, num_events + 1]
        probability_sum, _ = self.model.probe_sum_of_cdf(input_events, input_time, input_mask, taus, mean, std)
                                                                               # [sample_rate, batch_size, seq_len + 1]
        return probability_sum - probability_threshold
    
    batch_size, seq_len_plus_1 = input_events.shape
    tau_pred = []
    for sub_sample_rate in sample_rate_list:
        probability_threshold = torch.zeros((sub_sample_rate, batch_size, seq_len_plus_1), device = self.device)
                                                                               # [sub_sample_rate, batch_size, seq_len + 1]
        torch.nn.init.uniform_(probability_threshold, a = its_lower_bound, b = its_upper_bound)
                                                                               # [sub_sample_rate, batch_size, seq_len + 1]
        tau_pred.append(median_prediction(self.max_step, self.bisect_early_stop_threshold, \
                                         bisect_target, probability_threshold))# [sub_sample_rate, batch_size, seq_len + 1]

    tau_pred = torch.cat(tau_pred, dim = 0)                                    # [sample_rate, batch_size, seq_len + 1]
    
    return tau_pred


def sampling_by_its_for_mt(self, input_events, input_time, input_mask, p_m, number_of_total_samples, step, mean, std):
    # Preprocess
    sample_rate_list = step_split(number_of_total_samples, step)
    
    def bisect_target(taus, probability_threshold, p_m):
        p_mt, _ = self.model.probe_cdf(input_events, input_time, input_mask, taus, mean, std)
                                                                               # [sample_rate, batch_size, seq_len, num_events]
        p_t_m = p_mt / p_m                                                     # [sample_rate, batch_size, seq_len, num_events]
        p_gap = p_t_m - probability_threshold                                  # [sample_rate, batch_size, seq_len, num_events]

        return p_gap

    batch_size, seq_len = input_events.shape
    
    tau_pred = []
    for sub_sample_rate in sample_rate_list:
        probability_threshold = torch.zeros((sub_sample_rate, batch_size, seq_len, self.num_events + 1), device = self.device)
                                                                               # [sub_sample_rate, batch_size, seq_len + 1, num_events + 1]
        torch.nn.init.uniform_(probability_threshold, a = its_lower_bound, b = its_upper_bound)
        tau_pred.append(median_prediction(self.max_step, self.bisect_early_stop_threshold, \
                                          bisect_target, probability_threshold, p_m.unsqueeze(dim = 0)))
                                                                               # [sub_sample_rate, batch_size, seq_len + 1, num_events + 1]

    tau_pred = torch.cat(tau_pred, dim = 0)                                    # [sample_rate]

    return tau_pred


def sampling_by_thinning(self, task, *args, **kwargs):
    dict_apparoch_for_tasks = {
        'mt': sampling_by_thinning_for_mt,
        'tm': sampling_by_thinning_for_tm
    }

    return dict_apparoch_for_tasks[task](self, *args, **kwargs)


def sampling_by_thinning_for_mt(self, *args, **kwargs):
    raise Exception('Thinning algorithm can not solve task MT. Please use ITS by setting sampling_approach = its.')


def sampling_by_thinning_for_tm(self, events_history, time_history, mask_history, number_of_total_samples, step, mean, std):
    raise Exception('Marked LogNormMix does not know intensity functions, which thinning algorithm requires. Please use ITS by setting sampling_approach = its.')