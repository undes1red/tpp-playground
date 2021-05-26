import torch.nn as nn
import numpy as np
from .nonneg import NonNegLinear


class FullyNN(nn.Module):
    '''
    This is our implementation of Omi's paper: Fully Neural Network based Model for General Temporal Point Processes
    Hope it can work properly.

    Currently, normalization is disabled.

    Following Babylon's paper, we would check the performance of FullyNN with integral offsets.
    '''

    def __init__(self, d_history, d_intensity, dropout, num_layers, mlp_layers, device):
        super(FullyNN, self).__init__()

        self.rnn = nn.LSTM(input_size = 1, hidden_size = d_history, num_layers = num_layers, batch_first = True, dropout = dropout).to(device)

        self.hidden_x = NonNegLinear(1, d_intensity, bias = False).to(device)
        self.hidden_p = nn.Linear(d_history * num_layers, d_intensity, bias = True).to(device)

        # The original implement counts the hidden_x as one of mlp_layers
        self.mlp = nn.ModuleList([
            NonNegLinear(d_intensity, d_intensity, bias = True) for _ in range(mlp_layers)
        ]).to(device)

        self.agg = NonNegLinear(d_intensity, 1, bias = True).to(device)

        self.activate = nn.Tanh()
        self.activate_final = nn.Softplus()


    def forward(self, time_history, time_happen):
        # Reshape hidden output for full connection layers.
        # hidden: [batch_size, num_rnn_layer * d_history]
        _, (hidden, _) = self.rnn(time_history.unsqueeze(-1))
        hidden = hidden.reshape(time_history.shape[0], -1)

        time = self.hidden_x(time_happen)
        hidden = self.hidden_p(hidden)

        output = self.activate(time + hidden)

        for layer in self.mlp:
            output = layer(output)
            output = self.activate(output)
        
        output = self.activate_final(self.agg(output))

        return output


    def normalize_input(self,x):
        '''
        Appear in the original implement.
        But here it is not used.
        '''
        if self.log_mode:
            self.mu_x = np.log(x).mean() 
            self.sigma_x = np.log(x).std()   
        else:
            self.mu_x = x.mean()
            self.sigma_x = x.std()