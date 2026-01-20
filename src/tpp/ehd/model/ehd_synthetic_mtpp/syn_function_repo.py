import torch
import torch.nn as nn
from einops import repeat, rearrange
from src.tpp.tpp_models.utils import approximate_integration

class hawkes_1(nn.Module):
    def __init__(self, a, b, mu, t_end, device):
        super(hawkes_1, self).__init__()
        self.device = device
        self.a = a
        self.b = b
        self.mu = mu
        self.t_end = t_end


    def forward(self, time_history, resolution, selection_mask):
        batch_size, dim_x = time_history.shape[-2:]

        expanded_time_multiplier = torch.linspace(0, 1, resolution, device = self.device)
                                                                               # [resolution]
        expanded_time = expanded_time_multiplier * self.t_end                  # [resolution]
        timestamp = rearrange(expanded_time, f'r -> {"() " * (len(time_history.shape) - 1)} r')
                                                                               # [..., batch_size, resolution]
        timestamp_gap = timestamp.diff(dim = -1, prepend = torch.zeros((*timestamp.shape[:-1], 1), device = self.device))
                                                                               # [..., batch_size, resolution]
        # Calculate the probability.
        # step 1: calculate the intensity.
        expand_true_intensity = torch.ones((*time_history.shape[:-1], resolution), device = self.device) * self.mu
                                                                               # [..., batch_size, resolution]    
        padded_history_length = time_history.shape[-1]
        reversed_time_sum = time_history.max(dim = -1)[0].unsqueeze(dim = -1) - time_history
                                                                               # [..., batch_size, padded_history_length]
        for history_event_idx in range(1, padded_history_length):
            expand_batch_time = reversed_time_sum[..., history_event_idx].unsqueeze(dim = -1) + timestamp
                                                                               # [..., batch_size, resolution]
            expand_intensity_add = self.a * self.b * torch.exp(-self.b * expand_batch_time)
                                                                               # [..., batch_size, resolution]
            expand_true_intensity += expand_intensity_add * selection_mask[..., history_event_idx].unsqueeze(dim = -1).squeeze(dim = 0)
                                                                               # [..., batch_size, resolution]
        
        expand_true_intensity_integral = approximate_integration(expand_true_intensity, timestamp, dim = -1)
                                                                               # [..., batch_size, resolution]
        expanded_probability = expand_true_intensity * torch.exp(-expand_true_intensity_integral)
                                                                               # [..., batch_size, resolution]
        return expanded_probability, timestamp_gap


class poisson(nn.Module):
    def __init__(self, lam, t_end, device):
        super(poisson, self).__init__()
        self.device = device
        self.lam = lam
        self.t_end = t_end


    def forward(self, time_history, resolution, selection_mask):
        batch_size, dim_x = time_history.shape[-2:]

        expanded_time_multiplier = torch.linspace(0, 1, resolution, device = self.device)
                                                                               # [resolution]
        expanded_time = expanded_time_multiplier * self.t_end                  # [resolution]
        einop = f"r -> {'() ' * (len(time_history.shape) - len(expanded_time.shape))}r"
        timestamp = repeat(expanded_time, einop)                               # [..., batch_size, resolution]
        timestamp_gap = timestamp.diff(dim = -1, prepend = torch.zeros((*timestamp.shape[:-1], 1), device = self.device))
                                                                               # [..., batch_size, resolution]
        expanded_integral = timestamp * self.lam                               # [..., batch_size, resolution]
        expanded_probability = self.lam * torch.exp(-expanded_integral)        # [..., batch_size, resolution]
        return expanded_probability, timestamp_gap


class self_correct(nn.Module):
    def __init__(self, lam, t_end, device):
        super(self_correct, self).__init__()
        self.device = device
        self.lam = lam
        self.t_end = t_end


    def forward(self, time_history, resolution, selection_mask):
        batch_size, dim_x = time_history.shape[-2:]

        expanded_time_multiplier = torch.linspace(0, 1, resolution, device = self.device)
                                                                               # [resolution]
        expanded_time = expanded_time_multiplier * self.t_end                  # [resolution]
        einop = f"r -> {'() ' * (len(time_history.shape) - len(expanded_time.shape))}r"
        timestamp = repeat(expanded_time, einop)                               # [..., batch_size, resolution]
        timestamp_gap = timestamp.diff(dim = -1, prepend = torch.zeros((*timestamp.shape[:-1], 1), device = self.device))
                                                                               # [..., batch_size, resolution]
        expanded_integral = timestamp * self.lam                               # [..., batch_size, resolution]
        expanded_probability = self.lam * torch.exp(-expanded_integral)        # [..., batch_size, resolution]
        return expanded_probability, timestamp_gap


func_dict = {
    'hawkes_1': hawkes_1,
    'poisson': poisson,
    'self_correct': self_correct
}


class syn(nn.Module):
    def __init__(self, func, minimal_L1, device, **kwargs):
        super(syn, self).__init__()
        self.device = device
        self.minimal_L1 = minimal_L1
        self.func = func_dict[func](device = self.device, **kwargs)


    def forward(self, time_history, true_probability, selection_mask):
        batch_size, resolution = true_probability.shape
        result, timestamp = self.func(time_history, resolution, selection_mask)
        L1 = L1_distance_between_two_funcs(result, true_probability, timestamp, resolution, self.minimal_L1)
        return L1


def L1_distance_between_two_funcs(x, y, timestamp, resolution, minimal_L1):
    '''
    This function calculates the L^1 distance between two functions.
    Input:
    1. x:          function values
                   [seq_len * resolution, num_events]
    2. y:          function values
                   the number of points from [t_{i - 1}, t_i]
    3. time:       \Delta t
                   the number of event types
    '''

    function_interval = torch.abs(x - y).reshape(-1, resolution)[:, :-1]       # [batch_size * seq_len, resolution - 1]
    timestamp = timestamp.reshape(-1, resolution)[:, 1:]                       # [batch_size * seq_len, resolution - 1]

    L1 = (function_interval * timestamp).sum()

    # round up the L1 value.
    if L1 < minimal_L1:
        L1 = torch.tensor(0.)

    return L1