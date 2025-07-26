import torch
import torch.nn as nn
import math
from einops import rearrange, repeat, reduce

from src.toolbox.position_embedding import PositionalEmbedding


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
    def __init__(self, num_events, d_embedding, d_model, dropout = 0.1, device = None):
        super(DataEmbedding, self).__init__()
        self.num_events = num_events
        self.device = device

        self.mark_embedding = nn.Embedding(num_events, d_embedding, device = self.device)
        self.time_embedding = TimeEmbedding(d_embedding, device = self.device)
        
        self.position_embedding = PositionalEmbedding(d_model = d_model, device = self.device)
        
        self.scale_up = nn.Linear(d_embedding, d_model, device = self.device)
        
        self.dropout = nn.Dropout(p = dropout)


    def forward(self, x, time, mask):
        mark_embedding = self.mark_embedding(x) * mask.unsqueeze(dim = -1)       # [batch_size, seq_len, d_emb]
        time_embedding = self.time_embedding(time) * mask.unsqueeze(dim = -1)    # [batch_size, seq_len, d_emb]
        position_embedding = self.position_embedding(x)                          # [batch_size, seq_len, d_model]

        x = self.scale_up(mark_embedding + time_embedding) + position_embedding  # [batch_size, seq_len, d_model]
        x = self.dropout(x)                                                      # [batch_size, seq_len, d_model]
        return x