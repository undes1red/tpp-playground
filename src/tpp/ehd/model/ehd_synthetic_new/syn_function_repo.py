import torch
import torch.nn as nn

class polynomial1(nn.Module):
    def __init__(self, parameters, device):
        super(polynomial1, self).__init__()
        self.device = device
        self.parameters = torch.tensor(parameters, device = self.device)
        self.size = self.parameters.shape[0]

    def forward(self, x):
        dim_x = x.shape[-1]
        padded_parameters = torch.nn.functional.pad(self.parameters, pad = (max(dim_x - self.size, 0), 0), mode = 'constant', value = -5)
        return torch.sum(x * padded_parameters[:dim_x], axis = -1)


func_dict = {
    'polynomial': polynomial1,
}


class syn(nn.Module):
    def __init__(self, func, device, **kwargs):
        super(syn, self).__init__()
        self.device = device
        self.func = func_dict[func](device = self.device, **kwargs)


    def forward(self, x, y):
        result = self.func(x)
        return torch.abs(y - result)