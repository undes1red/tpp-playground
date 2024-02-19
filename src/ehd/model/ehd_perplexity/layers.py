import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from src.ehd.model.ehd_perplexity.selfattn import SelfAttn


class TransformerDecoder(nn.Module):
    def __init__(self, d_input, d_hidden, n_head, d_qk, d_v, n_layers_history_decoder, device, dropout):
        super(TransformerDecoder, self).__init__()
        self.device = device

        self.decoder = nn.ModuleList([
                TransformerLayer(d_input = d_input, d_hidden = d_hidden, n_head = n_head, \
                                 d_qk = d_qk, d_v = d_v, dropout = dropout, device = self.device) for _ in range(n_layers_history_decoder)
            ])


    def forward(self, x, non_pad_mask, self_attn_mask):
        '''
        Args:
        1. x: input tensor. shape: [batch_size, seq_len, d_input]
        2. self_attn_mask: mask tensor for used by self attention. shape: [seq_len, seq_len]
        3. pad_mask: mask out pad items' output values. shape: [batch_size, seq_len, d_attn_input]
        Outputs:
        '''
        for dec_layer in self.decoder:
            x, _ = dec_layer(
                x, x, x, non_pad_mask = non_pad_mask,
                self_attn_mask = self_attn_mask)                               # [batch_size, seq_len, d_input]
        
        return x


class TransformerLayer(nn.Module):
    def __init__(self, n_head, d_input, d_qk, d_v, device, d_hidden, dropout, ffn = True):
        super(TransformerLayer, self).__init__()
        self.device = device

        self.attn = MultiheadAttention(n_head = n_head, d_input = d_input, d_qk = d_qk,
                                       d_v = d_v, device = self.device, dropout = dropout)
        
        if ffn:
            self.ffn = FFN(d_input = d_input, d_hidden = d_hidden, device = self.device, dropout = dropout)
        else:
            self.ffn = None


    def forward(self, q, k, v, self_attn_mask, non_pad_mask):
        '''
        Args:
        1. x: input tensor. shape: [batch_size, seq_len, d_input]
        2. self_attn_mask: mask tensor for used by self attention. shape: [seq_len, seq_len]
        3. pad_mask: mask out pad items' output values. shape: [batch_size, seq_len, d_attn_input]
        Outputs:
        '''
        output, attn = self.attn(q, k, v, mask = self_attn_mask)               # [batch_size, seq_len, d_input] & [batch_size, n_head, seq_len, seq_len]
        if non_pad_mask is not None:
            output *= rearrange(non_pad_mask, '... -> ... 1')                  # [batch_size, seq_len, d_input]

        if self.ffn:
            output = self.ffn(output)                                          # [batch_size, seq_len, d_input]
            if non_pad_mask is not None:
                output *= rearrange(non_pad_mask, '... -> ... 1')              # [batch_size, seq_len, d_input]

        return output, attn


class MultiheadAttention(nn.Module):
    def __init__(self, n_head, d_input, d_qk, d_v, device, dropout):
        '''
        Template self-attention module with multihead-attention type 2: this module concatenates original outputs and
        compress high-dimensional vectors into d_output
        '''
        super(MultiheadAttention, self).__init__()
        self.device = device

        self.d_input = d_input
        self.d_output = d_input
        self.n_head = n_head
        self.d_q = d_qk
        self.d_k = d_qk
        self.d_v = d_v
        self.dropout = dropout

        # Linear: d_input -> d_q, d_k, or d_v
        self.w_q = nn.Linear(d_input, self.d_q * self.n_head, bias = False, device = self.device)
        self.w_k = nn.Linear(d_input, self.d_k * self.n_head, bias = False, device = self.device)
        self.w_v = nn.Linear(d_input, self.d_v * self.n_head, bias = False, device = self.device)

        # Self-attention module
        self.self_attn = SelfAttn(temperature = d_qk ** 0.5, attn_dropout = self.dropout, device = self.device)

        # Linear: n_head * d_q, d_k, or d_v -> d_output
        self.fc_attn_output = nn.Linear(self.n_head * d_v, self.d_output, bias = True, device = self.device)

        # Dropout
        self.dropout = nn.Dropout(self.dropout)

        # layer normalization
        self.layer_norm = nn.LayerNorm(self.d_input, eps = 1e-6, device = self.device)


    def forward(self, q, k, v, mask = None):
        '''
        Args:
        1. q: input tensor. shape: [batch_size, seq_len, d_input]
        2. k: input tensor. shape: [batch_size, seq_len, d_input]
        3. v: input tensor. shape: [batch_size, seq_len, d_input]
        4. mask: the mask tensor used by self attention. shape: [seq_len, seq_len]
        Output:
        1. output: results of transformer layer. shape: [batch_size, seq_len, d_output]
        2. attn: self attention value. shape: [batch_size, n_head, seq_len, seq_len]
        '''

        residual = q
        q = self.layer_norm(q)                                                 # [batch_size, seq_len, n_head, d_input]
        
        # preparing for q, k, and v.
        q = rearrange(self.w_q(q), 'b s (nh dq) -> b s nh dq', nh = self.n_head)
                                                                               # [batch_size, seq_len, n_head, d_qk]
        k = rearrange(self.w_k(k), 'b s (nh dk) -> b s nh dk', nh = self.n_head)
                                                                               # [batch_size, seq_len, n_head, d_qk]
        v = rearrange(self.w_v(v), 'b s (nh dv) -> b s nh dv', nh = self.n_head)
                                                                               # [batch_size, seq_len, n_head, d_v]

        output, attn = self.self_attn(q, k, v, mask = mask)                    # [batch_size, seq_len, n_head, d_v] & [batch_size, n_head, seq_len, seq_len]
        output = rearrange(output, 'b s nh dv -> b s (nh dv)', nh = self.n_head)
                                                                               # [batch_size, seq_len, n_head * d_v]
        output = self.dropout(self.fc_attn_output(output))                     # [batch_size, seq_len, d_input]
        output += residual

        output = self.layer_norm(output)                                       # [batch_size, seq_len, d_input]

        return output, attn


class FFN(nn.Module):
    '''
    Feedforward module next to the Transformers layer.
    '''
    def __init__(self, d_input, d_hidden, device, dropout):
        super(FFN, self).__init__()
        self.device = device
        
        self.w_1 = nn.Linear(d_input, d_hidden, device = self.device)
        self.w_2 = nn.Linear(d_hidden, d_input, device = self.device)
        self.dropout = nn.Dropout(dropout)

        self.norm = nn.LayerNorm(d_input, eps = 1e-6, device = self.device)


    def forward(self, x):
        '''
        Args:
        1. x: input tensor. shape: [..., d_input]
        Outputs:
        1. output: result tensor. shape: [..., d_input]
        '''
        residual = x

        x = self.norm(x)                                                       # [..., d_input]
        x = self.dropout(F.gelu(self.w_1(x)))                                  # [..., d_hidden]
        x = self.dropout(self.w_2(x))                                          # [..., d_input]
        x += residual
        x = self.norm(x)                                                       # [..., d_input]

        return x