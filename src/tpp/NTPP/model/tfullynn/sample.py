import torch

from einops import rearrange, reduce, repeat

from src.toolbox.integration import approximate_integration
from src.toolbox.misc import conditional_compile_class_method

from src.tpp.tpp_models.utils import step_split, median_prediction, thinning_sampling
from src.tpp.tpp_models.basic_tpp_model import its_lower_bound, its_upper_bound


@torch.no_grad
@conditional_compile_class_method
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


def sampling_by_its_for_mt(self, events_history, time_history, mask_history, p_m, resolution,
                            number_of_total_samples, step, inf_val, mean, std, 
                            autoregressive = False):
    sample_rate_list = step_split(number_of_total_samples, step)

    def evaluate_all_event(taus):
        '''
        placeholder
        '''
        # Train k FullyNN models for k different event types.
        integral_all_events, intensity_all_events, time_interval \
                = self.model.integral_intensity_time_next_3d(events_history, time_history, taus, mask_history, resolution, mean, std)
                                                                            # 2 * [sample_rate, batch_size, seq_len, resolution, num_events, num_events] + [sample_rate, batch_size, seq_len, resolution, num_events]
        event_mask = torch.diag(torch.ones(self.num_events, device = self.device))
                                                                            # [num_events, num_events]
        event_mask = rearrange(event_mask, f'ne ne1 -> {"() " * (len(intensity_all_events.shape) - 2)}ne ne1')
                                                                            # [sample_rate, batch_size, seq_len, resolution, num_events, num_events]
        intensity_all_events = reduce(intensity_all_events * event_mask, '... ne -> ...', 'sum')
                                                                            # [sample_rate, batch_size, seq_len, resolution, num_events]
        integral_all_events = reduce(integral_all_events, '... ne -> ...', 'sum')
                                                                            # [sample_rate, batch_size, seq_len, resolution, num_events]
        
        p_dist = intensity_all_events * torch.exp(-integral_all_events)    # [sample_rate, batch_size, seq_len, resolution, num_events]
        probability = approximate_integration(p_dist, time_interval, dim = -2, only_integral = True)
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
                            number_of_total_samples, step, mean, std, 
                            autoregressive = False):
    # Preprocess
    sample_rate_list = step_split(number_of_total_samples, step)

    def bisect_target(taus, probability_threshold):
        '''
        Retrieve the sum of all $ \\Lambda^*(m, t) $ over all $ m $ at $ \\tau $.

        Outputs:
        * integral    type: torch.tensor shape: [batch_size, seq_len]
                        $ \\sum_{n \\in M}{\\Lambda^*(n, \\tau)} $
        '''
        taus = repeat(taus, '... -> ... ne', ne = self.num_events)         # [sample_rate, batch_size, seq_len, num_events]
        integral = self.model(events_history, time_history, taus, mask_history, mean, std)
                                                                            # [sample_rate, batch_size, seq_len, num_events]
        integral = integral.sum(dim = -1)                                  # [sample_rate, batch_size, seq_len]
        
        return integral + torch.log(1 - probability_threshold)
    
    tau_pred = []
    for sub_sample_rate in sample_rate_list:
        probability_threshold = torch.zeros((sub_sample_rate, *time_history.shape), device = self.device)
                                                                            # [sample_rate, batch_size, seq_len]
        torch.nn.init.uniform_(probability_threshold, a = its_lower_bound, b = its_upper_bound)
                                                                            # [sample_rate, batch_size, seq_len]
        tau_pred.append(median_prediction(self.max_step, self.bisect_early_stop_threshold, \
                                            bisect_target, probability_threshold))
                                                                            # [sample_rate, batch_size, seq_len]
    tau_pred = torch.cat(tau_pred, dim = 0)                                # [sample_rate, batch_size, seq_len]

    return tau_pred


def sampling_by_thinning(self, task, *args, **kwargs):
    dict_apparoch_for_tasks = {
        'mt': sampling_by_thinning_for_mt,
        'tm': sampling_by_thinning_for_tm
    }

    return dict_apparoch_for_tasks[task](self, *args, **kwargs)


def sampling_by_thinning_for_mt(self, *args, **kwargs):
    raise Exception('WIP. Please use ITS by setting sampling_approach = its.')


def sampling_by_thinning_for_tm(self, events_history, time_history, mask_history, number_of_total_samples, step, mean, std):
    raise Exception('WIP. Please use ITS by setting sampling_approach = its.')

