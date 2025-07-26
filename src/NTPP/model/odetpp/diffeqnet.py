import torch.nn as nn
from inspect import signature

from src.TPP.model.odetpp.actnorm import ActNorm


class ConcatLinear_v2(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ConcatLinear_v2, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(1, dim_out, bias=False)
        self._hyper_bias.weight.data.fill_(0.0)


    def forward(self, t, x):
        return self._layer(x) + self._hyper_bias(t.view(-1, 1))


class DiffEqWrapper(nn.Module):
    def __init__(self, module):
        super(DiffEqWrapper, self).__init__()
        self.module = module


    def forward(self, t, y):
        if len(signature(self.module.forward).parameters) == 1:
            return self.module(y)
        elif len(signature(self.module.forward).parameters) == 2:
            return self.module(t, y)
        else:
            raise ValueError("Differential equation needs to either take (t, y) or (y,) as input.")


    def __repr__(self):
        return self.module.__repr__()
    

class SequentialDiffEq(nn.Module):
    """A container for a sequential chain of layers. Supports both regular and diffeq layers.
    """
    def __init__(self, *layers):
        super(SequentialDiffEq, self).__init__()
        self.layers = nn.ModuleList([DiffEqWrapper(layer) for layer in layers])


    def forward(self, t, x):
        for layer in self.layers:
            x = layer(t, x)
        return x


ACTFNS = {
    "softplus": DiffEqWrapper(nn.Softplus()),
    "celu": DiffEqWrapper(nn.CELU()),
    "relu": DiffEqWrapper(nn.ReLU(inplace=True))
}


def construct_diffeqnet(input_dim, hidden_dims, output_dim, actfn = "softplus", zero_init = False):
    layers = []
    if len(hidden_dims) > 0:
        dims = [input_dim] + list(hidden_dims)
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers.append(ConcatLinear_v2(d_in, d_out))
            layers.append(ActNorm(d_out))
            layers.append(ACTFNS[actfn])
        layers.append(ConcatLinear_v2(hidden_dims[-1], output_dim))
    else:
        layers.append(ConcatLinear_v2(input_dim, output_dim))

    # Initialize to zero.
    if zero_init:
        for m in layers[-1].modules():
            if isinstance(m, nn.Linear):
                m.weight.data.fill_(0)
                if m.bias is not None:
                    m.bias.data.fill_(0)
    diffeqnet = SequentialDiffEq(*layers)

    return diffeqnet


