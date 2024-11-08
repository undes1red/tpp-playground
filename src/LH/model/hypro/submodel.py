import torch
import torch.nn as nn

from einops import rearrange, repeat, reduce, pack, unpack
from src.LH.model.hypro.transformers import TransformerTPP


class HYPRO(nn.Module):
    def __init__(self, device, num_events, d_input, d_hidden, n_layers, n_head, d_qk, d_v, dropout):
        super(HYPRO, self).__init__()
        self.device = device
        
        self.encoder = TransformerTPP(num_events, device = self.device, d_input = d_input, \
                                      d_hidden = d_hidden, n_layers = n_layers, \
                                      n_head = n_head, d_qk = d_qk, d_v = d_v, dropout = dropout)
        
        self.energy_function = nn.Linear(d_input, 1, device = self.device)
    
    
    def forward(self, time_seq, event_seq, mask_seq):
        # shape of the event_seq: [batch_size, 1 + number_of_negative_samples, seq_len]
        # shape of the time_seq:  [batch_size, 1 + number_of_negative_samples, seq_len]
        # shape of the mask_seq:  [batch_size, 1 + number_of_negative_samples, seq_len]

        seq_embeddings = self.encoder(event_time = time_seq, event_type = event_seq, non_pad_mask = mask_seq)
                                                                               # [batch_size, 1 + number_of_negative_samples, seq_len, d_input]
        seq_energy = self.energy_function(seq_embeddings)                      # [batch_size, 1 + number_of_negative_samples, seq_len, 1]
        seq_energy = seq_energy * mask_seq.unsqueeze(dim = -1)                 # [batch_size, 1 + number_of_negative_samples, seq_len, 1]
        seq_energy = reduce(seq_energy, 'b nns sl () -> b nns', 'sum')         # [batch_size, 1 + number_of_negative_samples]
        
        return seq_energy