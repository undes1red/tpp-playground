import torch.nn as nn
from ..fullynn.nonneg import NonNegLinear

TA = {
    'softplus': nn.Softplus,
    'tanh': nn.Tanh
}

class FullyNNAutoregression(nn.Module):
    '''
    This is our implementation of Omi's paper: Fully Neural Network based Model for General Temporal Point Processes
    Hope it can work properly.

    Currently, normalization is disabled.

    Following Babylon's paper, we would check the performance of FullyNN with integral offsets.
    '''

    def __init__(self, d_history, d_intensity, dropout, rnn_layers, mlp_layers, nonlinear, device):
        super(FullyNNAutoregression, self).__init__()
        self.d_history = d_history
        self.d_intensity = d_intensity

        self.rnn = nn.LSTM(input_size = 1, hidden_size = d_history, num_layers = rnn_layers, batch_first = True,
                           dropout = dropout, proj_size = d_intensity).to(device)

        self.hidden_x = NonNegLinear(1, d_intensity, bias = False).to(device)

        # The original implement counts the hidden_x as one of mlp_layers
        self.mlp = nn.ModuleList([
            NonNegLinear(d_intensity, d_intensity, bias = True) for _ in range(mlp_layers)
        ]).to(device)

        self.agg = NonNegLinear(d_intensity, 1, bias = True).to(device)

        self.activate = TA[nonlinear]()
        self.activate_final = nn.Softplus()


    def forward(self, history, target):
        # Reshape hidden output for full connection layers.
        # The input should contain [BOS] and [EOS]. Their corresponding time should be 0 and the timestamp of the last event adding 0.1. 
        # (Just like what Mei does in his NHP paper.)
        
        minibatch_size, _ = history.shape

        # original hidden: [batch_size, num_layers * num_directions, d_intensity]
        # original output: [batch_size, seq_len, num_directions * d_intensity]
        output, (_, _) = self.rnn(history.unsqueeze(-1))
        # history [batch_size, sequence_length - 1, d_intensity]
        output = output.reshape(minibatch_size, -1, self.d_intensity)

        time = self.hidden_x(target.unsqueeze(-1))

        output = self.activate(time + output)

        for layer in self.mlp:
            output = layer(output)
            output = self.activate(output)
        
        output = self.activate_final(self.agg(output))

        return output