import torch.nn as nn
from einops import rearrange
from .attn import MultiheadAttention, EventAttention, NonNegFFN, FFN

class TransformerLayer(nn.Module):
    def __init__(self, n_head, d_input, d_qk, d_v, device, d_hidden, wq_nonneg, wk_nonneg, wv_nonneg, dropout = 0.1):
        super(TransformerLayer, self).__init__()
        self.device = device
        self.nonneg_ffn = False
        if wq_nonneg or wk_nonneg or wv_nonneg:
            self.nonneg_ffn = True

        self.attn = MultiheadAttention(n_head = n_head, d_input = d_input, d_qk = d_qk,
                                       d_v = d_v, device = self.device, dropout = dropout,
                                       wq_nonneg = wq_nonneg, wk_nonneg = wk_nonneg, wv_nonneg = wv_nonneg)
        
        if self.nonneg_ffn:
            self.ffn = NonNegFFN(d_input = d_input, d_hidden = d_hidden, device = self.device, dropout = dropout)
        else:
            self.ffn = FFN(d_input = d_input, d_hidden = d_hidden, device = self.device, dropout = dropout)

    def forward(self, q, k, v, self_attn_mask, non_pad_mask = None):
        '''
        Args:
        1. x: input tensor. shape: [..., seq_len, d_input]
        2. self_attn_mask: mask tensor for used by self attention. shape: [seq_len, seq_len]
        3. pad_mask: mask out pad items' output values. shape: [..., seq_len, d_attn_input]
        Outputs:
        '''
        non_pad_mask = rearrange(non_pad_mask, '... -> ... 1')                 # [..., seq_len, 1]
        output, attn = self.attn(q, k, v, mask = self_attn_mask)               # [..., seq_len, d_input] & [..., n_head, seq_len, seq_len]

        if non_pad_mask is not None:
            output = output * non_pad_mask                                     # [..., seq_len, d_input]

        output = self.ffn(output)                                              # [..., seq_len, d_input]

        if non_pad_mask is not None:
            output = output * non_pad_mask

        return output, attn


class MultiEventDecodeLayer(nn.Module):
    def __init__(self, n_head, d_input, d_qk, d_v, device, d_hidden, dropout = 0.1):
        super(MultiEventDecodeLayer, self).__init__()
        self.device = device

        self.attn = MultiheadAttention(n_head = n_head, d_input = d_input, d_qk = d_qk,
                                       d_v = d_v, dropout = dropout, wq_nonneg = True, 
                                       wk_nonneg = True, wv_nonneg = True, device = self.device)
        
        self.ffn = NonNegFFN(d_input = d_v, d_hidden = d_hidden, device = self.device)

    def forward(self, q, k, v, self_attn_mask, non_pad_mask = None):
        '''
        Args:
        1. q, v: input tensor. shape: [..., seq_len, d_input]
        2. k, input tensor. shape: [..., seq_len, n_head, d_input]
        3. self_attn_mask: mask tensor for used by self attention. shape: [..., seq_len, seq_len]
        4. non_pad_mask: mask out pad items' output values. shape: [..., seq_len]
        Outputs:
        '''
        output, attn = self.attn(q, k, v, mask = self_attn_mask)               # [..., seq_len, d_input] & [..., n_head, seq_len, seq_len]

        if non_pad_mask is not None:
            output = output * non_pad_mask                                     # [..., seq_len, d_input]

        output = self.ffn(output)                                              # [..., seq_len, d_input]

        if non_pad_mask is not None:
            output = output * non_pad_mask                                     # [..., seq_len, d_input]

        return output, attn

