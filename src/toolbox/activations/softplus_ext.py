import torch
import torch.nn.functional as F

from src.toolbox.misc import check_tensor

'''
This softplus function can accept a vector as beta.
'''
# @torch.compile()
def softplus_ext(input_data, beta, threshold = 20):
    '''
    This softplus function allows beta being a vector.

    input:     [..., d_input]
    beta:      [d_input]
    threshold: int
    '''
    if isinstance(beta, int):
        return F.softplus(input = input_data, beta = beta, threshold = threshold)

    check_tensor(beta, positive = False)

    input_with_beta = input_data * beta
    threshold_mask = (input_with_beta < threshold).float()
    masked_input = input_with_beta * threshold_mask

    output_part_1 = (1 / beta) * torch.log(1 + torch.exp(masked_input))
    output_part_2 = input_data * (1 - threshold_mask)

    return output_part_1 * threshold_mask + output_part_2
