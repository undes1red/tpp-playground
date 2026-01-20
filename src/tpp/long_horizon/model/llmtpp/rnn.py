import torch.nn as nn


class RNN_layers(nn.Module):
    """
    Optional recurrent layers. This is inspired by the fact that adding
    recurrent layers on top of the Transformer helps language modeling.
    """

    def __init__(self, d_model, d_rnn, device):
        super(RNN_layers, self).__init__()
        self.device = device

        self.rnn = nn.LSTM(d_model, d_rnn, num_layers = 1, batch_first = True, device = self.device)
        self.projection = nn.Linear(d_rnn, d_model, device = self.device)


    def forward(self, data):
        out = self.rnn(data)[1][0].squeeze(dim = 0)                            # [batch_size, d_rnn]
        out = self.projection(out)                                             # [batch_size, d_model]
        return out