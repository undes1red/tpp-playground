import torch

from einops import rearrange

from src.TPP.model.utils import step_split, median_prediction, thinning_sampling
from src.toolbox.integration import approximate_integration
from src.TPP.model.basic_tpp_model import its_lower_bound, its_upper_bound


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

    return dict_sampling_apparoch[sampling_approach](self, task = task, *args, **kwargs)


def sampling_by_its(self, task, *args, **kwargs):
    dict_apparoch_for_tasks = {
        'mt': sampling_by_its_for_mt,
        'tm': sampling_by_its_for_tm
    }

    return dict_apparoch_for_tasks[task](self, *args, **kwargs)


def sampling_by_its_for_mt(self, events_history, time_history, p_m, resolution, max_val, number_of_total_samples, step, mean, std, autoregressive = False):
    sample_rate_list = step_split(number_of_total_samples, step)

    def evaluate_all_event(taus):
        expanded_integral_across_events, expanded_intensity_across_events, timestamp = \
            self.model.integral_intensity_time_next_3d(events_history, time_history, taus, resolution, mean, std)
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
        p_mt = evaluate_all_event(taus)                                        # [sample_rate, batch_size, seq_len, num_events]
        p_t_m = p_mt / p_m                                                     # [sample_rate, batch_size, seq_len, num_events]
        p_gap = p_t_m - probability_threshold                                  # [sample_rate, batch_size, seq_len, num_events]

        return p_gap

    tau_pred = []
    batch_size, seq_len = time_history.shape
    p_m = p_m.unsqueeze(dim = 0)                                               # [1, batch_size, seq_len, num_events]        
    for sub_sample_rate in sample_rate_list:
        probability_threshold = torch.zeros((sub_sample_rate, batch_size, seq_len, self.num_events), device = self.device)
                                                                               # [sample_rate, batch_size, seq_len, num_events]
        torch.nn.init.uniform_(probability_threshold, a = its_lower_bound, b = its_upper_bound)
                                                                               # [sample_rate, batch_size, seq_len, num_events]
        tau_pred.append(median_prediction(self.max_step, self.bisect_early_stop_threshold, bisect_target, probability_threshold, r_val = max_val))
                                                                               # [sample_rate, batch_size, seq_len, num_events]
    tau_pred = torch.cat(tau_pred, dim = 0)                                    # [sample_rate, batch_size, seq_len, num_events]
    
    return tau_pred


def sampling_by_its_for_tm(self, events_history, time_history, number_of_total_samples, step, mean, std, autoregressive = False):
    sample_rate_list = step_split(number_of_total_samples, step)

    def bisect_target(taus, probability_threshold):
        integral, _, _ = self.model(events_history, time_history, taus, mean, std)
                                                                               # [sample_rate, batch_size, seq_len, num_events]
        integral = integral.sum(dim = -1)                                      # [sample_rate, batch_size, seq_len]
        return integral + torch.log(1 - probability_threshold)                 # [sample_rate, batch_size, seq_len]

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
        _, intensity, _ = self.model(events_history, time_history, tau, mean, std)
                                                                               # [sample_rate, batch_size, seq_len, num_events]
        return intensity.sum(dim = -1)

    def find_maximum_intensity_values_in_one_interval(interval_left, interval_right, time_history, events_history):
        intensity_values_at_left_side = get_intensity(interval_left, time_history, events_history)
                                                                               # [sample_rate, batch_size, seq_len]
        intensity_values_at_right_side = get_intensity(interval_right, time_history, events_history)
                                                                               # [sample_rate, batch_size, seq_len]        
        intensity_values_at_t_l_higher = (intensity_values_at_left_side > intensity_values_at_right_side).int()
                                                                               # [sample_rate, batch_size, seq_len]
        # We slightly lift the upper bound here to ensure this upper bound definitely higher than all intensity values in this interval.
        intensity_values_for_thinning_upper_bound = (intensity_values_at_left_side * intensity_values_at_t_l_higher + intensity_values_at_right_side * (1 - intensity_values_at_t_l_higher)) * 1.05
                                                                               # [sample_rate, batch_size, seq_len]
        return intensity_values_for_thinning_upper_bound
    
    sampled_time = []
    for each_step in sample_rate_list:
        sampled_time.append(thinning_sampling(maximum_thinning_loops, max_sample_time_limit, (each_step, batch_size, seq_len), self.device, \
                                                get_intensity, find_maximum_intensity_values_in_one_interval, time_history, events_history))
                                                                               # [sample_rate, batch_size, seq_len]
    sampled_time = torch.cat(sampled_time, dim = 0)
    return sampled_time
