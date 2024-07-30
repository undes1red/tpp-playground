from torch import nn


'''
An extension to tanh.
Original tanh will limit the input into [-1, 1].
This tanh can limit the input into [-parameter, parameter].
'''
class scaled_tanh(nn.Module):
    def __init__(self, parameter = 1, device = None):
        super(scaled_tanh, self).__init__()
        self.device = device
        self.parameter = parameter
    
    def forward(self, x):
        return self.parameter * nn.functional.tanh(x)