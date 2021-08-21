import torch.nn as nn
import numpy as np
from .nonneg import NonNegLinear

TA = {
    'softplus': nn.Softplus,
    'tanh': nn.Tanh
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