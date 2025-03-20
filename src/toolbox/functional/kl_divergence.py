import torch
import numpy as np

from scipy.special import kl_div

from einops import rearrange


def kl_divergence(p, q, dim = -1, distribution_name = 'generic', loss = False):
    '''
    This function calculates KL(p || q) 
    = \\sum_{x}{p(x)\\log\\frac{p(x)}{q(x)}}
    = \\sum_{x}{ p(x)\\log\\frac{1}{q(x)} - p(x)\\log\\frac{1}{q(x)} }
    
    The input distributions must be in the logit space.
    '''
    if loss:
        return kl_divergence_torch(p, q, dim, distribution_name)
    else:
        return kl_divergence_numpy(p, q, dim, distribution_name)


# 1. Pytorch part
def kl_divergence_gaussian_torch(p, q, dim):
    pass

def kl_divergence_generic_torch(p, q, dim):
    assert len(p.shape) == len(q.shape), f"The dimension of p {len(p.shape)} mismatches q {len(q.shape)}."
    dim_num = len(p.shape)
    if dim != -1 or dim != dim_num - 1:
        if dim < 0:
            dim = dim_num + dim + 1
        else:
            dim = dim + 1
        einop = " ".join(["a{}".format(i) for i in range(dim)]) + ' ... -> ' + " ".join(["a{}".format(i) for i in range(dim - 1)]) + f' ... a{dim - 1}'
        p = rearrange(p, einop)                                                # [..., dim]
        q = rearrange(q, einop)                                                # [..., dim]
    
    return torch.nn.functional.kl_div(q, p, log_target = True, reduction='none').sum(dim = -1)

kl_divergence_torch_function_set = {
    'gaussian': kl_divergence_gaussian_torch,
    'generic': kl_divergence_generic_torch,
}

def kl_divergence_torch(p, q, dim, distribution_name):
    return kl_divergence_torch_function_set[distribution_name](p, q, dim)


# 2. numpy part
def kl_divergence_gaussian_numpy(p, q, dim):
    pass

def kl_divergence_generic_numpy(p, q, dim):
    assert len(p.shape) == len(q.shape), f"The dimension of p {len(p.shape)} mismatches q {len(q.shape)}."
    dim_num = len(p.shape)
    if dim != -1 or dim != dim_num - 1:
        if dim < 0:
            dim = dim_num + dim + 1
        else:
            dim = dim + 1
        einop = " ".join(["a{}".format(i) for i in range(dim)]) + ' ... -> ' + " ".join(["a{}".format(i) for i in range(dim - 1)]) + f' ... a{dim - 1}'
        p = rearrange(p, einop)                                                # [..., dim]
        q = rearrange(q, einop)                                                # [..., dim]
    
    return kl_div(np.exp(p), np.exp(q)).sum(dim = -1)

kl_divergence_numpy_function_set = {
    'gaussian': kl_divergence_gaussian_numpy,
    'generic': kl_divergence_generic_numpy,
}

def kl_divergence_numpy(p, q, dim, distribution_name):
    return kl_divergence_numpy_function_set[distribution_name](p, q, dim)


if __name__ == '__main__':
    import torch
    import numpy as np
    
    a = torch.tensor([[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]]).log()
    b = torch.tensor([[0.1, 0.4, 0.2, 0.3], [0.4, 0.1, 0.3, 0.2]]).log()
    
    kl_div_pytorch = kl_divergence(a, b, loss = True)
    kl_div_numpy = kl_divergence(a, b, loss = False)
    
    print(f'pytorch generic: {kl_div_pytorch}.')
    print(f'numpy generic: {kl_div_numpy}.')

    a = torch.tensor([[[0.25], [0.25], [0.25], [0.25]], [[0.25], [0.25], [0.25], [0.25]]]).log()
    b = torch.tensor([[[0.1], [0.4], [0.2], [0.3]], [[0.4], [0.1], [0.3], [0.2]]]).log()
    
    kl_div_pytorch = kl_divergence(a, b, dim = -2, loss = True)
    kl_div_numpy = kl_divergence(a, b, dim = -2, loss = False)
    
    print(f'pytorch generic: {kl_div_pytorch}.')
    print(f'numpy generic: {kl_div_numpy}.')
    
    a = torch.randn(3, 5, 7, 6)
    b = torch.randn(3, 5, 7, 6)
    
    a_norm = torch.nn.functional.log_softmax(a, dim = -2)
    b_norm = torch.nn.functional.log_softmax(b, dim = -2)
    
    kl_div_pytorch = kl_divergence(a_norm, b_norm, dim = -2, loss = True)
    kl_div_numpy = kl_divergence(a_norm, b_norm, dim = -2, loss = False)

    print(f'pytorch generic: {kl_div_pytorch}.')
    print(f'numpy generic: {kl_div_numpy}.')

    kl_div_pytorch = kl_divergence(a_norm, b_norm, dim = 2, loss = True)
    kl_div_numpy = kl_divergence(a_norm, b_norm, dim = 2, loss = False)

    print(f'pytorch generic: {kl_div_pytorch}.')
    print(f'numpy generic: {kl_div_numpy}.')