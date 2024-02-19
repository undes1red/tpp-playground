import torch
import torch.nn as nn
import math
from einops import rearrange


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, device, max_len = 5000):
        super(PositionalEmbedding, self).__init__()
        self.device = device
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model, device = self.device).float()
        pe.require_grad = False

        position = torch.arange(0, max_len, device = self.device).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2, device = self.device).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model, device):
        super(TokenEmbedding, self).__init__()
        self.device = device

        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels = c_in, out_channels = d_model,
                                   kernel_size = 3, padding = padding, padding_mode = 'circular', \
                                   bias = False, device = self.device)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = rearrange(x, 'b s c -> b c s')                                     # [batch_size, in_channel, seq_len]
        x = self.tokenConv(x)                                                  # [batch_size, out_channel, seq_len]
        x = rearrange(x, 'b c s -> b s c')                                     # [batch_size, seq_len, out_channel]
        return x


class DataEmbedding(nn.Module):
    def __init__(self, num_events, d_embedding, d_model, device = None, dropout = 0.1):
        super(DataEmbedding, self).__init__()
        self.num_events = num_events
        self.device = device

        self.mark_embedding = nn.Embedding(num_events, d_embedding, device = self.device)
        self.to_token_embedding = TokenEmbedding(d_embedding, d_model, device = self.device)
        self.position_embedding = PositionalEmbedding(d_model = d_model, device = self.device)
        self.dropout = nn.Dropout(p = dropout)


    def forward(self, x):
        x = self.to_token_embedding(self.mark_embedding(x)) + self.position_embedding(x)
        return self.dropout(x)