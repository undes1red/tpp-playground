import torch

def check_tensor(x, positive = True, inf = True, nan = True):
    '''
    Ensure that the input tensor does not contain: negative numbers, inf, and nan.
    
    Args:
    * x  type: torch.tensor shape: any shape
         the input tensor.

    Outputs:
      No outputs available.
    '''
    if positive:
        assert (x < 0).any() == False, 'Negative numbers detected!'

    if inf:
        assert torch.isfinite(x).all() == True, 'inf detected in input!'

    if nan:
        assert torch.isnan(x).any() == False, 'Nan detected in input!'