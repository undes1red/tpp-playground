import torch
import torch.nn.functional as F
from src.TPP.model.utils import check_tensor


def get_subsequent_mask(seq):
    """ For masking out the subsequent info, i.e., masked self-attention. """

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls
    return subsequent_mask


def softplus_ext(input, beta, threshold = 20):
    '''
    This softplus function allows beta being a vector.

    input:     [..., d_input]
    beta:      [d_input]
    threshold: int
    '''
    if type(beta) == int:
        return F.softplus(input = input, beta = beta, threshold = threshold)
    
    check_tensor(beta, positive = False)
    assert input.shape[-1] == beta.shape[-1]

    input_with_beta = input * beta
    threshold_mask = (input_with_beta < threshold).float()
    masked_input = input_with_beta * threshold_mask

    output_part_1 = (1 / beta) * torch.log(1 + torch.exp(masked_input))
    output_part_2 = input * (1 - threshold_mask)

    output = output_part_1 * threshold_mask + output_part_2


    return output