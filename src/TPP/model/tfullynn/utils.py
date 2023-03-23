import torch
import numpy as np
from einops import rearrange, reduce, repeat


def get_subsequent_mask(seq):
    """ For masking out the subsequent info, i.e., masked self-attention. """

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls
    return subsequent_mask


def check_tensor(x):
    '''
    Ensure that the input tensor does not contain: negative numbers, inf, and nan.
    
    Args:
    * x  type: torch.tensor shape: any shape
         the input tensor.

    Outputs:
      No outputs available.
    '''
    assert (x < 0).any() == False, 'Negative numbers detected!'
    assert torch.isfinite(x).all() == True, 'inf detected in input!'
    assert torch.isnan(x).any() == False, 'Nan detected in input!'


def move_from_tensor_to_ndarray(*kwargs):
    tmp_results = []
    for tensor in kwargs:
        tmp_results.append(tensor.detach().cpu().numpy())
    
    return tmp_results


'''
custom metrics
'''
def L1_distance_across_events(input, resolution, num_events, time_next):
    '''
    This function calculates the L^1 distance between two functions in scattered form.
    Input:
    1. input:      function values
                   [seq_len * resolution, num_events]
    2. resolution: int
                   the number of points from [t_{i - 1}, t_i]
    3. num_events: int
                   the number of event types
    4. time_next:  [seq_len, num_events]
                   the length of all intervals with interpolations.
    '''

    input = rearrange(input, '(s r) ne -> ne s r', r = resolution)             # [num_events, seq_len, resolution]
    intensity_1 = repeat(input, 'ne s r -> ne new_d s r', new_d = num_events)  # [num_events, num_events, seq_len, resolution]
    intensity_2 = repeat(input, 'ne s r -> new_d ne s r', new_d = num_events)  # [num_events, num_events, seq_len, resolution]
    delta_intensity = np.abs(intensity_1 - intensity_2)                        # [num_events, num_events, seq_len, resolution]

    gap = time_next.detach().cpu().numpy() / (resolution - 1)                  # [seq_len]
    gap = rearrange(gap, 's -> 1 1 s 1')                                       # [num_events, num_events, seq_len, 1]

    L1 = reduce((delta_intensity * gap)[:, :, :, :-1], 'ne1 ne2 s r -> ne1 ne2', 'sum')
                                                                               # [num_events, num_events]
    # round off the value smaller than 1e-6
    L1[L1 < 1e-6] = 0

    return L1


def L1_distance_between_two_funcs(x, y, timestamp, resolution):
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

    function_interval = np.abs(x - y).reshape(-1, resolution)[:, :-1]          # [batch_size * seq_len, resolution - 1]
    timestamp = timestamp.reshape(-1, resolution)[:, 1:]                       # [batch_size * seq_len, resolution - 1]

    L1 = (function_interval * timestamp).sum()

    # round up the value smaller than 1e-6
    if L1 < 1e-6:
        L1 = 0

    return L1