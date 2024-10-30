import torch

from src.TPP.model.utils import step_split, median_prediction, thinning_sampling
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


def sampling_by_its_for_mt(self, *args, **kwargs):
    raise Exception("Vanilla RMTPP does not support task MT as its intensity function is not mark-aware.")


def sampling_by_its_for_tm(self, events_history, time_history, number_of_total_samples, step, mean, std, autoregressive = False):
    sample_rate_list = step_split(number_of_total_samples, step)

    def bisect_target(taus, probability_threshold):
        '''
        MTPP loss function
        '''
        integral, _, _, _ = self.model(events_history, time_history, taus, mean, std)
                                                                            # [sample_rate, batch_size, seq_len]
        return integral + torch.log(1 - probability_threshold)             # [sample_rate, batch_size, seq_len]

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


def sampling_by_thinning_for_tm(self, events_history, time_history, number_of_total_samples, step, mean, std):
    sample_rate_list = step_split(number_of_total_samples, step)
    batch_size, seq_len = time_history.shape
    maximum_thinning_loops = 50
    max_sample_time_limit = mean + 10 * std

    def get_intensity(tau, time_history, events_history):
        _, intensity, _, _ = self.model(events_history, time_history, tau, mean, std)
        return intensity

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
