import torch.nn as nn
import torch
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len = 4096):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        length = x.shape[-1]
        return self.pe[:, :length]


class BiasedPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len, device):
        super().__init__()
        self.device = device
        self.d_model = d_model

        position = torch.arange(0, max_len, device = self.device).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2, device = self.device).float() * -(math.log(10000.0) / d_model)).exp()
        self.register_buffer('position', position)
        self.register_buffer('div_term', div_term)

        self.Wt = nn.Linear(1, d_model // 2 + (1 if d_model % 2 else 0), bias = False, device = self.device)


    def forward(self, x, interval):
        phi = self.Wt(interval.unsqueeze(-1))                                  # [..., d_model // 2 + 1 if d_model % 2 else 0]
        length = x.shape[-1]

        arc = (self.position[:length] * self.div_term).unsqueeze(0)            # [1, seq_len, d_model // 2 + 1 if d_model % 2 else 0]

        pe_cos = torch.cos(arc + phi)                                          # [1, seq_len, d_model // 2 + 1 if d_model % 2 else 0]
        pe_sin = torch.sin(arc + phi)                                          # [1, seq_len, d_model // 2 + 1 if d_model % 2 else 0]
        if self.d_model % 2 == 1:
            pe_sin = pe_sin[..., :-1]                                          # [1, seq_len, d_model // 2]
        pe = torch.cat([pe_sin, pe_cos], dim=-1)                               # [1, seq_len, d_model // 2]

        return pe