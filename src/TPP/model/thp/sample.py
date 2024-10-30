import torch

from einops import rearrange

from src.TPP.model.utils import step_split, median_prediction, thinning_sampling
from src.toolbox.integration import approximate_integration
from src.TPP.model.basic_tpp_model import its_lower_bound, its_upper_bound


@torch.no_grad()
def sample_time(self, sampling_approach = 'its', task = 'mt', *args, **kwargs):
    '''
    number_of_total_samples: how many samples do we need to predict one next event.
    step: we output "step" samples to reduce memory comsumption during inference.
    sampling_approach: 'its' for invert transform sampling and 'thinning' for thinning algorithm.
    task: 'mt' for mark first time second, 'tm' for time first mark second.
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


def sampling_by_its_for_mt(self, events_history, time_history, mask_history, p_m, resolution,
                            number_of_total_samples, step, inf_val, mean, std, autoregressive = False):
    sample_rate_list = step_split(number_of_total_samples, step)

    def evaluate_all_event(taus):
        expanded_integral_across_events, expanded_intensity_across_events, timestamp = \
            self.model.integral_intensity_time_next_3d(events_history, time_history, taus, mask_history, resolution)
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
                            number_of_total_samples, step, mean, std, autoregressive = False):
    sample_rate_list = step_split(number_of_total_samples, step)

    def bisect_target(taus, probability_threshold):
        '''
        MTPP loss function
        '''
        integral_all_events, _ = self.model(time_history, taus, events_history, mask_history)
                                                                            # [sample_rate, batch_size, seq_len, num_events]
        gap = integral_all_events.sum(dim = -1) + torch.log(1 - probability_threshold)
                                                                            # [sample_rate, batch_size, seq_len]
        return gap

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
    raise Exception('Thinning algorithm can not solve task MT. Please use ITS by setting sampling_approach = its.')


def sampling_by_thinning_for_tm(self, events_history, time_history, mask_history, number_of_total_samples, step,
                                mean, std, autoregressive = False):
    sample_rate_list = step_split(number_of_total_samples, step)
    batch_size, seq_len = time_history.shape
    maximum_thinning_loops = 50
    max_sample_time_limit = mean + 10 * std

    def get_intensity(tau, time_history, events_history, mask_history):
        return self.model(time_history, tau, events_history, mask_history)[-1].sum(dim = -1)
    
    def find_maximum_intensity_values_in_one_interval(interval_left, interval_right, time_history, events_history, mask_history):
        _, intensity_between_interval_left_and_right, _ \
            = self.model.integral_intensity_time_next_2d(events_history, time_history, interval_right, mask_history, \
                                                            self.integration_sample_rate, time_next_start = interval_left)
                                                                            # [sample_rate, batch_size, seq_len, integration_sample_rate, num_events]
        intensity_between_interval_left_and_right = intensity_between_interval_left_and_right.sum(dim = -1)
                                                                            # [sample_rate, batch_size, seq_len, integration_sample_rate]

        return intensity_between_interval_left_and_right.max(dim = -1)[0]
    
    sampled_time = []
    for each_step in sample_rate_list:
        sampled_time.append(thinning_sampling(maximum_thinning_loops, max_sample_time_limit, (each_step, batch_size, seq_len), self.device, \
                                                get_intensity, find_maximum_intensity_values_in_one_interval, time_history, events_history, mask_history))
                                                                            # [sample_rate, batch_size, seq_len]
    
    sampled_time = torch.cat(sampled_time, dim = 0)
    return sampled_time