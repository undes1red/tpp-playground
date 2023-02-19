import torch


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