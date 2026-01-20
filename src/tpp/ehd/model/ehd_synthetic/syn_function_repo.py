import torch
import torch.nn as nn


class sin1(nn.Module):
    def __init__(self, w, shift):
        super(sin1, self).__init__()
        self.w = w
        self.shift = shift
    
    def forward(self, x):
        return torch.sin(self.w * x) + self.shift


class log1(nn.Module):
    def __init__(self, a, b, shift):
        super(log1, self).__init__()
        self.a = a
        self.b = b
        self.shift = shift
    
    def forward(self, x):
        return torch.log(self.a * torch.abs(x) + self.b) + self.shift


class linear1(nn.Module):
    def __init__(self, shift):
        super(linear1, self).__init__()
        self.shift = shift

    def forward(self, x):
        return 0.5 * x - 1 + self.shift


class tanh1(nn.Module):
    def __init__(self, shift):
        super(tanh1, self).__init__()
        self.shift = shift

    def forward(self, x):
        return torch.tanh(x) + self.shift


class circle1(nn.Module):
    def __init__(self, r):
        super(circle1, self).__init__()
        self.r = r

    def forward(self, x, y):
        return torch.abs(torch.sqrt(x**2 + y**2) - self.r)


func_dict = {
    'linear': linear1,
    'sin': sin1,
    'log': log1,
    'tanh': tanh1,
    'circle': circle1
}


class syn(nn.Module):
    def __init__(self, func, **kwargs):
        super(syn, self).__init__()
        self.func_name = func
        self.func = func_dict[func](**kwargs)

    def forward(self, x, y):
        if self.func_name == 'circle':
            return self.func(x, y)
        else:
            result = self.func(x)
            return torch.abs(y - result)

    def get_true_result(self, x):
        return self.func(x)