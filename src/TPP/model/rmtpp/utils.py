import torch
from torch.autograd import Function
from scipy.special import expi
from einops import rearrange, reduce, repeat
import numpy as np


class EI(Function):
    '''
    Special function: Exponential Integral function: expi(x) = \integral_{-x}^{\inft}{\frac{e^{-t}}{t}}
    '''
    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        data = input.cpu().detach().numpy()
        result = expi(data)
        return input.new(result)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.new(grad_output * torch.exp(ctx.input)/ctx.input)


def check_tensor(x):
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

# if __name__ == '__main__':
#     from torch.autograd import gradcheck
# 
#     # gradcheck takes a tuple of tensors as input, check if your gradient
#     # evaluated with these tensors are close enough to numerical
#     # approximations and returns True if they all verify this condition.
#     input = (torch.randn(3,dtype=torch.double,requires_grad=True))
#     test = gradcheck(EI.apply, input, eps=1e-6, atol=1e-4)
#     print(test)