import torch
import torch.nn as nn


class sym_Log(nn.Module):
    '''
    The symmetry-about-a-point log activation function.
    '''
    def __init__(self, multiplier=1):
        super(sym_Log, self).__init__()
        self.multiplier = multiplier

    def forward(self, x):
        mask = (x >= 0).int() - (x < 0).int()
        activate = self.multiplier * torch.log(1 + torch.abs(x))
        return mask * activate


class sym_softplus(nn.Module):
    '''
    The symmetry-about-a-point log activation function
    '''
    def __init__(self, multiplier=1):
        super(sym_softplus, self).__init__()
        self.multiplier = multiplier

    def forward(self, x):
        mask = (x >= 0).int() - (x < 0).int()
        activate = self.multiplier * torch.nn.functional.softplus(torch.abs(x))
        return mask * activate


class sym_Polynomial(nn.Module):
    '''
    The symmetry-about-a-point polynomial activation function
    '''
    def __init__(self, multiplier=1):
        super(sym_Polynomial, self).__init__()
        self.multiplier = multiplier
        self.power = nn.parameter.Parameter(torch.tensor(.5))

    def forward(self, x):
        mask = (x >= 0).int() - (x < 0).int()
        activate = self.multiplier * torch.pow(x, self.power)
        return mask * activate