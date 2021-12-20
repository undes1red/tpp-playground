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
