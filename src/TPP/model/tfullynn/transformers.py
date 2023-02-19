import math, torch
import torch.nn as nn
from einops import rearrange, reduce, repeat

from src.TPP.model.tfullynn.layers import TransformerLayer
from src.TPP.model.tfullynn.utils import *


class TransEncoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(
            self,
            num_events, d_input, d_hidden,
            n_layers, n_head, d_qk, d_v, dropout,
            event_toggle, device):
        super(TransEncoder, self).__init__()
        self.device = device
        self.d_input = d_input
        self.event_toggle = event_toggle
        self.num_events = num_events

        # position vector, used for temporal encoding
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / d_input) for i in range(d_input)],
            device=self.device)

        # event type embedding
        self.event_emb = nn.Embedding(num_events + 1, d_input, padding_idx = num_events, device = self.device)

        # history time encoder
        self.history_time_emb = nn.Linear(1, d_input, device = self.device)

        self.encoder = nn.ModuleList([
            TransformerLayer(d_input = d_input, d_hidden = d_hidden, n_head = n_head,\
                             d_qk = d_qk, d_v = d_v, dropout = dropout, device = self.device)
            for _ in range(n_layers)])


    def encode_position_idx(self, idx):
        """
        Input:  [seq_len]
        Output: [batch_size, seq_len, d_input]
        """
        
        idx = rearrange(idx, '... -> ... 1')
        result = idx / self.position_vec                                       # [seq_len, d_input]
        result[:, 0::2] = torch.sin(result[:, 0::2])
        result[:, 1::2] = torch.cos(result[:, 1::2])

        result = rearrange(result, '... -> 1 ...')
        return result


    def forward(self, events_history, time_history, non_pad_mask):
        """
        Encode event sequences via masked self-attention.
        Args:
        1. events_history: historical events.        shape: [batch_size, seq_len]
        2. time_history: historical time intervals.  shape: [batch_size, seq_len]
        3. non_pad_mask: pad mask tensor.            shape: [batch_size, seq_len]
        """

        # prepare attention masks
        # slf_attn_mask is where we cannot look, i.e., the future and the padding
        _, seq_len = time_history.shape
        seq_idx = torch.arange(seq_len, device = self.device)                  # [seq_len]
        
        self_attn_mask_subseq = get_subsequent_mask(events_history)            # [batch_size, seq_len, seq_len]
        self_attn_mask_keypad = torch.ones_like(non_pad_mask, device = self.device) - non_pad_mask
                                                                               # [batch_size, seq_len]
        self_attn_mask_keypad = repeat(self_attn_mask_keypad, 'b s -> b s s1', s1 = seq_len)
                                                                               # [batch_size, seq_len, seq_len]
        self_attn_mask = (self_attn_mask_keypad + self_attn_mask_subseq).gt(0) # [batch_size, seq_len, seq_len]

        idx_emb = self.encode_position_idx(seq_idx)                            # [1, seq_len, d_input]
        time = rearrange(time_history, '... -> ... 1')                         # [batch_size, seq_len, 1]

        time_emb = self.history_time_emb(time) + idx_emb                       # [batch_size, seq_len, d_input]

        if self.event_toggle:
            time_emb = time_emb + self.event_emb(events_history)               # [batch_size, seq_len, d_input]
            
            for enc_layer in self.encoder:
                '''
                history event sequence
                '''
                time_emb, _ = enc_layer(
                    time_emb, time_emb, time_emb,
                    non_pad_mask = non_pad_mask,
                    self_attn_mask = self_attn_mask)                           # [batch_size, seq_len, d_input]
        else:
            for enc_layer in self.time_encoder:
                '''
                history event sequence
                '''
                time_emb, _ = enc_layer(
                    time_emb, time_emb, time_emb,
                    non_pad_mask = non_pad_mask,
                    self_attn_mask = self_attn_mask)                           # [batch_size, seq_len, d_input]

        return time_emb