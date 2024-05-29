# Code from https://github.com/facebookresearch/neural_stpp.git.

import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class ActNorm(nn.Module):
    def __init__(self, num_features, init_scale=1.0):
        super(ActNorm, self).__init__()
        self.num_features = num_features
        self.weight = nn.Parameter(torch.Tensor(num_features))
        self.bias = nn.Parameter(torch.Tensor(num_features))
        self.init_scale = init_scale
        self.register_buffer('initialized', torch.tensor(0))

    def forward(self, x):
        if not self.initialized:
            with torch.no_grad():
                # compute batch statistics
                x_ = x.reshape(-1, x.shape[-1])
                batch_mean = torch.mean(x_, dim=0)
                batch_var = torch.var(x_, dim=0)

                # for numerical issues
                batch_var = torch.max(batch_var, torch.tensor(0.2).to(batch_var))

                self.bias.data.copy_(-batch_mean)
                self.weight.data.copy_(-0.5 * torch.log(batch_var) + math.log(self.init_scale))
                self.initialized.fill_(1)

        bias = self.bias.expand_as(x)
        weight = self.weight.expand_as(x)

        # y = (x + bias) * torch.exp(weight)
        y = (x + bias) * F.softplus(weight)

        return y

    def __repr__(self):
        return ('{name}({num_features})'.format(name=self.__class__.__name__, **self.__dict__))