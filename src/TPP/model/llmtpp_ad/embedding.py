import torch
import torch.nn as nn
import math
from einops import rearrange, repeat, reduce


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
    def __init__(self, c_in, d_model, num_events_in_one_patch, device):
        super(TokenEmbedding, self).__init__()
        self.device = device

        self.tokenConv = nn.Conv2d(in_channels = c_in, out_channels = d_model, \
                                   kernel_size = (1, num_events_in_one_patch), bias = False, device = self.device)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = rearrange(x, 'b pl num_in_p dim -> b dim pl num_in_p')             # [batch_size, c_in, num_of_patches, events_in_one_patch]
        x = self.tokenConv(x)                                                  # [batch_size, c_out, num_of_patches, 1]
        x = rearrange(x, 'b c_out num_in_p () -> b num_in_p c_out')            # [batch_size, num_of_patches, c_out]
        return x


class TimeEmbedding(nn.Module):
    def __init__(self, d_embedding, device):
        super(TimeEmbedding, self).__init__()
        self.device = device
        self.d_embedding = d_embedding
        self.div_term = torch.exp(torch.arange(0, d_embedding, 2, device = self.device) * -(math.log(10000.0) / d_embedding)).reshape(1, 1, -1)
    
    def forward(self, time):
        pe = torch.zeros(*(time.shape), self.d_embedding, device = self.device)
        _time = time.unsqueeze(-1)
        pe[..., 0::2] = torch.sin(_time * self.div_term)
        pe[..., 1::2] = torch.cos(_time * self.div_term)
        # pe = pe * non_pad_mask.unsqueeze(-1)
        return pe


class DataEmbedding(nn.Module):
    def __init__(self, d_embedding, d_model, num_events_in_one_patch, dropout = 0.1, device = None):
        super(DataEmbedding, self).__init__()
        self.device = device

        self.time_embedding = TimeEmbedding(d_embedding, device = self.device)
        self.to_token_embedding = TokenEmbedding(d_embedding, d_model, num_events_in_one_patch, device = self.device)
        self.position_embedding = PositionalEmbedding(d_model = d_model, device = self.device)
        self.dropout = nn.Dropout(p = dropout)


    def forward(self, time, mask):
        time_embedding = self.time_embedding(time) * mask.unsqueeze(dim = -1)    # [batch_size, num_of_patch, patch_len, d_emb]
        position_embedding = self.position_embedding(time_embedding)             # [batch_size, num_of_patch, d_model]

        x = self.to_token_embedding(time_embedding) + position_embedding
                                                                                 # [batch_size, num_of_patch, d_model]
        x = self.dropout(x)                                                      # [batch_size, num_of_patch, d_model]
        return x