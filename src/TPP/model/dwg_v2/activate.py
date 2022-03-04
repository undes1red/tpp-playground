import torch
import torch.nn as nn


class Log(nn.Module):
    def __init__(self, multiplier=1):
        super(Log, self).__init__()
        self.multiplier = multiplier

    def forward(self, x):
        mask = (x >= 0).int() - (x < 0).int()
        activate = self.multiplier * torch.log(1 + torch.abs(x))
        return mask * activate

'''
From https://arxiv.org/abs/2112.11687
'''
class SquarePlus(nn.Module):
    def __init__(self, b = 4):
        super(SquarePlus, self).__init__()
        self.b = b
    
    def forward(self, x):
        return 0.5 * (x + torch.sqrt(x * x + self.b))