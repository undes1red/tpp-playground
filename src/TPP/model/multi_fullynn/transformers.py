import math, torch
import torch.nn as nn

from .layers import TransformerLayer
from .utils import *
from .nonneg import NonNegLinear


class TransEncoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(
            self,
            num_types, d_input, d_hidden,
            n_layers, n_head, d_qk, d_v, dropout,
            event_toggle, wq_nonneg, wk_nonneg, wv_nonneg,
            device):
        super(TransEncoder, self).__init__()
        self.device = device
        self.d_input = d_input
        self.event_toggle = event_toggle
        self.num_types = num_types

        # position vector, used for temporal encoding
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / d_input) for i in range(d_input)],
            device=self.device)

        # event type embedding
        self.event_emb = nn.Embedding(num_types + 1, d_input, padding_idx = num_types, device = self.device)

        # history time encoder
        self.history_time_emb = nn.Linear(1, d_input, device = self.device)

        self.event_encoder = nn.ModuleList([
            TransformerLayer(d_input = d_input, d_hidden = d_hidden, n_head = n_head,\
                             d_qk = d_qk, d_v = d_v, dropout = dropout, wq_nonneg = wq_nonneg, \
                             wk_nonneg = wk_nonneg, wv_nonneg = wv_nonneg, device = self.device)
            for _ in range(n_layers)])

        self.time_encoder = nn.ModuleList([
            TransformerLayer(d_input = d_input, d_hidden = d_hidden, n_head = n_head,\
                             d_qk = d_qk, d_v = d_v, dropout = dropout, wq_nonneg = wq_nonneg, \
                             wk_nonneg = wk_nonneg, wv_nonneg = wv_nonneg, device = self.device)
            for _ in range(n_layers)])

    def encode_position_idx(self, idx):
        """
        Input:  [seq_len]
        Output: [batch_size, seq_len, d_input]
        """

        result = idx.unsqueeze(-1) / self.position_vec
        result[:, 0::2] = torch.sin(result[:, 0::2])
        result[:, 1::2] = torch.cos(result[:, 1::2])
        return result

    def forward(self, event_type, event_time, non_pad_mask):
        """
        Encode event sequences via masked self-attention.
        Args:
        1. event_type: 
        2. event_time: input time intervals. shape: [batch_size, seq_len, 1]
        3. non_pad_mask: pad mask tensor. shape: [batch_size, seq_len, 1]
        """

        # prepare attention masks
        # slf_attn_mask is where we cannot look, i.e., the future and the padding
        seq_idx = torch.arange(non_pad_mask.shape[1], device = self.device)
        
        self_attn_mask_subseq = get_subsequent_mask(event_time)
        self_attn_mask_keypad = torch.ones_like(non_pad_mask, device = self.device) - non_pad_mask
                                                                               # [batch_size, seq_len, 1]
        self_attn_mask_keypad = self_attn_mask_keypad.repeat(1, 1, self_attn_mask_keypad.shape[1])
                                                                               # [batch_size, seq_len, seq_len]
        self_attn_mask = (self_attn_mask_keypad + self_attn_mask_subseq).gt(0) # [batch_size, seq_len, seq_len]

        idx_emb = self.encode_position_idx(seq_idx).unsqueeze(dim = 0)         # [seq_len, d_input]
        time = event_time                                                      # [batch_size, seq_len, 1]

        if self.event_toggle:
            events_emb = self.event_emb(event_type) + idx_emb                  # [batch_size, seq_len, d_input]
            for enc_layer in self.event_encoder:
                '''
                history event sequence
                '''
                events_emb, _ = enc_layer(
                    events_emb, events_emb, events_emb,
                    non_pad_mask = non_pad_mask,
                    self_attn_mask = self_attn_mask)                           # [batch_size, seq_len, d_input]
            
            time_emb = self.history_time_emb(time) + idx_emb                   # [batch_size, seq_len, d_input]
            time_emb, _ = self.time_encoder[0](
                    events_emb, time_emb, time_emb,
                    non_pad_mask = non_pad_mask,
                    self_attn_mask = self_attn_mask)                           # [batch_size, seq_len, d_input]
            for enc_layer in self.time_encoder[1:]:
                '''
                history event sequence
                '''
                time_emb, _ = enc_layer(
                    time_emb, time_emb, time_emb,
                    non_pad_mask = non_pad_mask,
                    self_attn_mask = self_attn_mask)                           # [batch_size, seq_len, d_input]
            
            time_emb += events_emb                                             # [batch_size, seq_len, d_input]
        else:
            time_emb = self.history_time_emb(time) + idx_emb                   # [batch_size, seq_len, d_input]
            for enc_layer in self.time_encoder:
                '''
                history event sequence
                '''
                time_emb, _ = enc_layer(
                    time_emb, time_emb, time_emb,
                    non_pad_mask = non_pad_mask,
                    self_attn_mask = self_attn_mask)                           # [batch_size, seq_len, d_input]

        return time_emb

class HistoryTimeMixer(nn.Module):
    """ A History and Time information mixer with self attention mechanism. """

    def __init__(
            self,
            num_types, d_input, d_hidden,
            n_layers, n_head, d_qk, d_v, dropout, 
            device):
        super(HistoryTimeMixer, self).__init__()
        self.device = device
        self.d_input = d_input
        self.num_types = num_types

        # position vector, used for temporal encoding
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / d_input) for i in range(d_input)],
            device=self.device)

        # Two time Encoder here
        # Relative time encoder and absolution time encoder
        self.relative_time_emb = NonNegLinear(1, d_input, device = self.device)
        self.absolute_time_emb = NonNegLinear(1, d_input, device = self.device)

        self.mixer = nn.ModuleList([
            TransformerLayer(d_input = d_input, d_hidden = d_hidden, n_head = n_head,\
                             d_qk = d_qk, d_v = d_v, dropout = dropout, device = self.device,\
                             wq_nonneg = True, wk_nonneg = False, wv_nonneg = True)
            ] + [
            TransformerLayer(d_input = d_input, d_hidden = d_hidden, n_head = n_head,\
                             d_qk = d_qk, d_v = d_v, dropout = dropout, device = self.device,\
                             wq_nonneg = True, wk_nonneg = True, wv_nonneg = True)
            for _ in range(n_layers - 1)])

    def encode_position_idx(self, idx):
        """
        Input:  [seq_len]
        Output: [batch_size, seq_len, d_input]
        """

        result = idx.unsqueeze(-1) / self.position_vec
        result[:, 0::2] = torch.sin(result[:, 0::2])
        result[:, 1::2] = torch.cos(result[:, 1::2])
        return result

    def forward(self, history, time, non_pad_mask):
        """
        Encode event sequences via masked self-attention.
        Args:
        1. history:         shape: [batch_size, seq_len, num_events, d_intensity]
        2. relative time:   shape: [batch_size, seq_len, num_events]
        3. non_pad_mask     shape: [batch_size, seq_len, 1]
        """

        # prepare attention masks
        # slf_attn_mask is where we cannot look, i.e., the future and the padding
        seq_idx = torch.arange(non_pad_mask.shape[1], device = self.device)
        
        self_attn_mask_subseq = get_subsequent_mask(non_pad_mask)
        self_attn_mask_keypad = torch.ones_like(non_pad_mask, device = self.device) - non_pad_mask
                                                                               # [batch_size, seq_len, 1]
        self_attn_mask_keypad = self_attn_mask_keypad.repeat(1, 1, self_attn_mask_keypad.shape[1])
                                                                               # [batch_size, seq_len, seq_len]
        self_attn_mask = (self_attn_mask_keypad + self_attn_mask_subseq).gt(0) # [batch_size, seq_len, seq_len]

        idx_emb = self.encode_position_idx(seq_idx).unsqueeze(dim = 0)         # [seq_len, d_input]
        
        # Time embedding
        absolute_time = torch.cumsum(time, dim = -1)                           # [batch_size, seq_len, num_events]
        relative_time_vec = self.relative_time_emb(time.unsqueeze(dim = -1))   # [batch_size, seq_len, num_events, d_input]
        absolute_time_vec = self.absolute_time_emb(absolute_time.unsqueeze(dim = -1))
                                                                               # [batch_size, seq_len, num_events, d_input]

        output, _ = self.mixer[0](relative_time_vec + absolute_time_vec, history, relative_time_vec + absolute_time_vec, \
                                  self_attn_mask, non_pad_mask)
                                                                               # [batch_size, seq_len, num_events, d_output]
        for layer in self.mixer[1:]:
            output = layer(output, output, output, self_attn_mask, non_pad_mask)
        
        return 0
        
