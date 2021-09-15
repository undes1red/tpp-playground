import torch.nn as nn
import torch

from .nonneg import NonNegLinear
from .activate import Log

TA = {
    # Vanilla Softplus harms the algorithm by shifting the entire distribution into the non-nrgative area.
    # That is to say, each scalar in the output vector is bigger than log(1 + num_of_layer(n)) if all hidden layer weights only 
    # have positive numbers like what FullyNN does.
    'softplus': nn.Softplus,
    # Some papers have pointed out that Tanh introduces significant gradient vanishment when the input time is too big. After theoretical
    # analysis, we argue that this feature is required by approaches like FullyNN to regress long-tail functions like Hawkes intensity function.
    'tanh': nn.Tanh,
    # Yet another function that has small gradients when it has big inputs. But as the log function is not bounded above, the hard integral bound introduced
    # by tanh can be alleviated.
    'log': Log,
    # This activation can perfectly show why FullyNN needs tanh to attain a trade-off between intensity function regression ability and extrapolation 
    'identity': nn.Identity
}

class FullyNN(nn.Module):
    '''
    This is our implementation of Omi's paper: Fully Neural Network based Model for General Temporal Point Processes
    Hope it can work properly.

    Currently, normalization is disabled.

    Following Babylon's paper, we would check the performance of FullyNN with integral offsets.
    '''

    def __init__(self, d_history, d_intensity, dropout, rnn_layers, mlp_layers, nonlinear, device):
        super(FullyNN, self).__init__()
        self.device = device

        self.rnn = nn.LSTM(input_size = 1, hidden_size = d_history, num_layers = rnn_layers, batch_first = True, dropout = dropout).to(device)

        self.hidden_x = NonNegLinear(1, d_intensity, bias = False).to(device)
        self.hidden_p = nn.Linear(d_history, d_intensity, bias = True).to(device)

        # The original implement counts the hidden_x as one of mlp_layers
        self.mlp = nn.ModuleList([
            NonNegLinear(d_intensity, d_intensity, bias = True) for _ in range(mlp_layers)
        ]).to(device)

        self.agg = NonNegLinear(d_intensity, 1, bias = True).to(device)

        self.activate = TA[nonlinear]()
        self.activate_final = nn.Softplus()


    def forward(self, time_history, time_next):
        '''
        Args:
            time_history: [batch_size, seq_len, 1]
            time_next:    [batch_size, seq_len, 1]
        '''
        # Reshape hidden output for full connection layers.
        output, (_, _) = self.rnn(time_history)                                # [batch_size, seq_len, d_history]

        time = self.hidden_x(time_next)                                        # [batch_size, seq_len, d_intensity]
        hidden = self.hidden_p(output)                                         # [batch_size, seq_len, d_intensity]

        output = self.activate(time + hidden)                                  # [batch_size, seq_len, d_intensity]

        for layer in self.mlp:
            output = layer(output)                                             # [batch_size, seq_len, d_intensity]
            output = self.activate(output)                                     # [batch_size, seq_len, d_intensity]

        output = self.activate_final(self.agg(output))                         # [batch_size, seq_len, 1]
        return output

    def integral_intensity(self, time_history, time_next, resolution):
        '''
        Intensity integral & intensity function prober. Perhaps, we can support intensity integral as well.
        Args:
        time_history: [batch_size, seq_len, 1]
        time_next:    [batch_size, seq_len, 1]
        resolution:   int
        '''
        output, (_, _) = self.rnn(time_history)                                # [batch_size, seq_len, d_history]
        hidden = self.hidden_p(output)                                         # [batch_size, seq_len, d_intensity]
        batch_size, seq_len, d_intensity = hidden.shape

        hidden_expand = hidden.repeat(1, 1, resolution).reshape(batch_size, -1, d_intensity)
                                                                               # [batch_size, seq_len * resolution, d_intensity]
        time_multiplier = torch.linspace(0, 1, resolution)                     # [resolution]
        time_expand = (time_multiplier * time_next).reshape(batch_size, -1, 1) # [batch_size, seq_len * resolution, 1]
        time_expand.requires_grad = True
        emb_time_expand = self.hidden_x(time_expand)                           # [batch_size, seq_len * resolution, d_intensity]
        output = self.activate(emb_time_expand + hidden_expand)                # [batch_size, seq_len * resolution, d_intensity]

        for layer in self.mlp:
            output = layer(output)                                             # [batch_size, seq_len * resolution, d_intensity]
            output = self.activate(output)                                     # [batch_size, seq_len * resolution, d_intensity]

        expand_integral = self.activate_final(self.agg(output))                # [batch_size, seq_len * resolution, 1]

        expand_intensity = torch.autograd.grad(
            outputs=expand_integral,
            inputs=time_expand,
            grad_outputs=torch.ones_like(expand_integral),
            create_graph=True,
        )[0]                                                                   # [batch_size, seq_len * resolution, 1]
        time_expand.requires_grad = False
        timestamp = time_expand.squeeze().reshape(batch_size, seq_len, resolution)
                                                                               # [batch_size, seq_len, resolution]
        timestamp = torch.cat(
            (torch.zeros((batch_size, seq_len, 1), device = self.device), timestamp.diff(dim = -1)),
            dim = -1)                                                          # [batch_size, seq_len, resolution]
        timestamp = timestamp.reshape(batch_size, seq_len * resolution)        # [batch_size, seq_len * resolution]

        return expand_integral, expand_intensity, timestamp