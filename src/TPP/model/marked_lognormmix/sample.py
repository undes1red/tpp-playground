import torch

from src.TPP.model.utils import median_prediction
from src.TPP.model.basic_tpp_model import its_lower_bound, its_upper_bound


@torch.inference_mode()
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

    return dict_sampling_apparoch[sampling_approach](self, task, *args, **kwargs)


def sampling_by_its(self, task, *args, **kwargs):
    dict_apparoch_for_tasks = {
        'mt': sampling_by_its_for_mt,
        'tm': sampling_by_its_for_tm
    }

    return dict_apparoch_for_tasks[task](self, *args, **kwargs)


def sampling_by_its_for_tm(self, input_events, input_time, input_mask, mean, std):
    def bisect_target(taus, probability_threshold):
        probability_sum, _ = self.model.probe_sum_of_cdf(input_events, input_time, input_mask, taus, mean, std)
                                                                            # [sample_rate, batch_size, seq_len + 1]
        return probability_sum - probability_threshold
    
    probability_threshold = torch.zeros((self.sample_rate, *input_time.shape), device = self.device)
                                                                            # [sample_rate, batch_size, seq_len]
    torch.nn.init.uniform_(probability_threshold, a = its_lower_bound, b = its_upper_bound)
                                                                            # [sample_rate, batch_size, seq_len]
    tau_pred = median_prediction(self.max_step, self.bisect_early_stop_threshold, \
                                    bisect_target, probability_threshold)     # [sample_rate, batch_size, seq_len + 1]

    return tau_pred


def sampling_by_its_for_mt(self, input_events, input_time, input_mask, p_m, mean, std):
    '''
    The input should be the original minibatch
    MAE evaluation part, dwg and fullynn exclusive
    '''
    def bisect_target(taus, probability_threshold, p_m):
        p_mt, _ = self.model.probe_cdf(input_events, input_time, input_mask, taus, mean, std)
                                                                            # [sample_rate, batch_size, seq_len, num_events]
        p_t_m = p_mt / p_m                                                 # [sample_rate, batch_size, seq_len, num_events]
        p_gap = p_t_m - probability_threshold                              # [sample_rate, batch_size, seq_len, num_events]

        return p_gap

    batch_size, seq_len = input_events.shape
    probability_threshold = torch.zeros((self.sample_rate, batch_size, seq_len, self.num_events + 1), device = self.device)
                                                                            # [sample_rate, batch_size, seq_len + 1, num_events + 1]
    torch.nn.init.uniform_(probability_threshold, a = its_lower_bound, b = its_upper_bound)
    p_m = p_m.unsqueeze(dim = 0)                                           # [1, batch_size, seq_len, num_events]
    tau_pred = median_prediction(self.max_step, self.bisect_early_stop_threshold, \
                                    bisect_target, probability_threshold, p_m)# [sample_rate, batch_size, seq_len + 1, num_events + 1]

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