import torch
import torch.nn as nn


class state_transform(nn.Module):
    def __init__(self, d_input, d_hidden, d_output, layer, device):
        super(state_transform, self).__init__()
        self.device = device
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.d_output = d_output

        if layer == 1:
            self.model = nn.Sequential(
                nn.Linear(self.d_input, self.d_output, bias = True, device = self.device),
                nn.Tanh()
            )
        elif layer == 2:
            self.model = nn.Sequential(
                nn.Linear(self.d_input, self.d_hidden, bias = True, device = self.device),
                nn.Tanh(),
                nn.Linear(self.d_hidden, self.d_output, bias = True, device = self.device),
                nn.Tanh(),
            )
        elif layer > 2:
            self.model = nn.Sequential(
                nn.Linear(self.d_input, self.d_hidden, bias = True, device = self.device),
                nn.Tanh(),
                nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(self.d_hidden, self.d_hidden, bias = True, device = self.device),
                            nn.Tanh())
                    ] * (layer - 2)
                ),
                nn.Linear(self.d_hidden, self.d_output, bias = True, device = self.device),
                nn.Tanh()
            )
        else:
            raise Exception('Too few fully-connected layers!')

    
    def forward(self, s, x):
        self.model(x)