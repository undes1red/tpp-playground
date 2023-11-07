import torch
import torch.nn.functional as F


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

    assert input.shape[-1] == beta.shape[-1]
    last_dim = input.shape[-1]
    result = []

    for index in range(last_dim):
        result.append(F.softplus(input = input[..., index], beta = beta[index].item(), threshold = threshold))

    result = torch.stack(result, dim = -1)

    return result

'''
Abandonded

def softplus_ext(input, beta, threshold = 20):
    \'''
    This softplus function allows beta being a vector.

    input:     [..., d_input]
    beta:      [d_input]
    threshold: int
    \'''
    if type(beta) == int:
        return F.softplus(input = input, beta = beta, threshold = threshold)

    assert input.shape[-1] == beta.shape[-1]

    output_part_1 = (1 / beta) * torch.log(1 + torch.exp(input * beta))
    output_part_2 = input

    threshold_mask = (input * beta > threshold).int()
    infinity_mask = torch.isinf(output_part_1).int()
    mask_for_masked_fill = (threshold_mask + infinity_mask).gt(0)

    output = torch.zeros_like(input)
    output[~mask_for_masked_fill] = output_part_1.masked_select(~mask_for_masked_fill)
    output[mask_for_masked_fill] = output_part_2.masked_select(mask_for_masked_fill)

    return output

    '''