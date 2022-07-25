import torch.nn as nn
import torch.nn.functional as F
from .selfattn import SelfAttn
from .nonneg import NonNegLinear

class TransformerLayer(nn.Module):
    def __init__(self, n_head, d_input, d_qk, d_v, device, d_hidden, dropout = 0.1, wq_nonneg = False, wk_nonneg = False, wv_nonneg = False):
        super(TransformerLayer, self).__init__()
        self.device = device

        self.attn = MultiheadAttention(n_head = n_head, d_input = d_input, d_qk = d_qk,
                                       d_v = d_v, device = self.device, dropout = dropout,
                                       wq_nonneg = wq_nonneg, wk_nonneg = wk_nonneg, wv_nonneg = wv_nonneg)
        self.ffn = FFN(d_input = d_input, d_hidden = d_hidden, device = self.device, dropout = dropout)

    def forward(self, q, k, v, self_attn_mask, non_pad_mask):
        '''
        Args:
        1. x: input tensor. shape: [batch_size, seq_len, d_input]
        2. self_attn_mask: mask tensor for used by self attention. shape: [seq_len, seq_len]
        3. pad_mask: mask out pad items' output values. shape: [batch_size, seq_len, d_attn_input]
        Outputs:
        '''
        output, attn = self.attn(q, k, v, mask = self_attn_mask)               # [batch_size, seq_len, d_input] & [batch_size, n_head, seq_len, seq_len]
        output *= non_pad_mask                                                 # [batch_size, seq_len, d_input]

        output = self.ffn(output)                                              # [batch_size, seq_len, d_input]
        output *= non_pad_mask

        return output, attn


class MultiheadAttention(nn.Module):
    def __init__(self, n_head, d_input, d_qk, d_v, device, wq_nonneg, wk_nonneg, wv_nonneg, dropout = 0.1, ):
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

        assert self.n_head > 0

        # Linear: d_input -> d_q, d_k, or d_v
        if wq_nonneg:
            self.w_q = NonNegLinear(d_input, self.d_q * self.n_head, bias = False, device = self.device)
        else:
            self.w_q = nn.Linear(d_input, self.d_q * self.n_head, bias = False, device = self.device)
        
        if wk_nonneg:
            self.w_k = NonNegLinear(d_input, self.d_k * self.n_head, bias = False, device = self.device)
        else:
            self.w_k = nn.Linear(d_input, self.d_k * self.n_head, bias = False, device = self.device)
        
        if wv_nonneg:
            self.w_v = NonNegLinear(d_input, self.d_v * self.n_head, bias = False, device = self.device)
        else:
            self.w_v = nn.Linear(d_input, self.d_v * self.n_head, bias = False, device = self.device)

        # Self-attention module
        self.self_attn = SelfAttn(temperature = d_qk ** 0.5, attn_dropout = self.dropout, device = self.device, \
                                  wq_nonneg = wq_nonneg, wk_nonneg = wk_nonneg, wv_nonneg = wv_nonneg)

        # Linear: n_head * d_q, d_k, or d_v -> d_output
        if wv_nonneg:
            self.fc_attn_output = NonNegLinear(self.n_head * d_v, self.d_output, bias = True, device = self.device)
        else:
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

        batch_size = q.shape[0]
        residual = q
        q = self.layer_norm(q)                                                 # [batch_size, seq_len, n_head, d_qk]
        
        # preparing for q, k, and v.
        q = self.w_q(q).view(batch_size, -1, self.n_head, self.d_q)            # [batch_size, seq_len, n_head, d_qk]
        k = self.w_k(k).view(batch_size, -1, self.n_head, self.d_k)            # [batch_size, seq_len, n_head, d_qk]
        v = self.w_q(v).view(batch_size, -1, self.n_head, self.d_v)            # [batch_size, seq_len, n_head, d_qk]

        output, attn = self.self_attn(q, k, v, mask = mask)                    # [batch_size, seq_len, n_head, d_output] & [batch_size, n_head, seq_len, seq_len]
        output = output.reshape(batch_size, -1, self.n_head * self.d_v)        # [batch_size, seq_len, n_head * d_v]
        output = self.dropout(self.fc_attn_output(output))                     # [batch_size, seq_len, d_output]
        output += residual

        output = self.layer_norm(output)                                       # [batch_size, seq_len, d_output]

        return output, attn


class FFN(nn.Module):
    '''
    Feedforward module next to the Transformers layer.
    '''
    def __init__(self, d_input, d_hidden, device, dropout = 0.1):
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