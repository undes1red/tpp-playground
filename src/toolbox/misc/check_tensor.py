import torch
import math
import numbers


def check_tensor(x, positive = True, inf = True, nan = True, break_out = True):
    '''
    Ensure that the input tensor does not contain: negative numbers, inf, and nan.
    
    Args:
    * x  type: torch.tensor shape: any shape
         the input tensor.

    Outputs:
      No outputs available.
    '''
    assert torch.is_tensor(x), 'Input value is not a torch.tensor!'

    if_positive = True
    if positive:
        if_positive = (x >= 0).all().item()

    if_inf = False
    if inf:
        if_inf = not torch.isfinite(x).any().item()

    if_nan = False
    if nan:
        if_nan = torch.isnan(x).any().item()
    
    if break_out:
        assert if_positive and not if_inf and not if_nan, f'Input Check failed! Input: {x}.'
    else:
        return if_positive and not if_inf and not if_nan


def check_number(x, positive = True, inf = True, nan = True, break_out = True):
    assert isinstance(x, numbers.Number), 'Input value is not a number!'

    if_positive = True
    if positive:
        if_positive = x >= 0

    if_inf = False
    if inf:
        if_inf = math.isinf(x)
    
    if_nan = False
    if nan:
        if_nan = math.isnan(x)

    if break_out:
        assert if_positive and not if_inf and not if_nan, f'Input Check failed! Input: {x}.'
    else:
        return if_positive and not if_inf and not if_nan