import torch.nn as nn
import torch.nn.functional as F
from .selfattn import SelfAttn
from .nonneg import NonNegLinear

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
        output, attn = self.attn(q, k, v, mask = self_attn_mask)               # [..., seq_len, d_input] & [..., n_head, seq_len, seq_len]

        if non_pad_mask is not None:
            output *= non_pad_mask                                             # [..., seq_len, d_input]

        output = self.ffn(output)                                              # [..., seq_len, d_input]

        if non_pad_mask is not None:
            output *= non_pad_mask

        return output, attn


class MultiheadAttention(nn.Module):
    def __init__(self, n_head, d_input, d_qk, d_v, device, wq_nonneg, wk_nonneg, wv_nonneg, dropout = 0.1, ):
        '''
        Template self-attention module with multihead-attention type 2: this modu`le concatenates original outputs and
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
        self.nonneg_norm = False
        if wq_nonneg or wk_nonneg or wv_nonneg:
            self.nonneg_norm = True

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
        if self.nonneg_norm:
            self.layer_norm = NonNegNorm(device = self.device)
        else:
            self.layer_norm = nn.LayerNorm(self.d_input, eps = 1e-6, device = self.device)


    def forward(self, q, k, v, mask = None):
        '''
        Args:
        1. q: input tensor. shape: [..., seq_len, d_input]
        2. k: input tensor. shape: [..., seq_len, d_input]
        3. v: input tensor. shape: [..., seq_len, d_input]
        4. mask: the mask tensor used by self attention. shape: [..., seq_len, seq_len]
        Output:
        1. output: results of transformer layer. shape: [..., seq_len, d_output]
        2. attn: self attention value. shape: [..., n_head, seq_len, seq_len]
        '''

        q_shape, k_shape, v_shape = q.shape, k.shape, v.shape
        residual = q
        q = self.layer_norm(q)                                                 # [..., seq_len, d_qk]
        
        # preparing for q, k, and v.
        q = self.w_q(q).view(*q_shape[:-1], self.n_head, self.d_q)             # [..., seq_len, n_head, d_qk]
        k = self.w_k(k).view(*k_shape[:-1], self.n_head, self.d_k)             # [..., seq_len, n_head, d_qk]
        v = self.w_q(v).view(*v_shape[:-1], self.n_head, self.d_v)             # [..., seq_len, n_head, d_qk]

        output, attn = self.self_attn(q, k, v, mask = mask)                    # [..., seq_len, n_head, d_output] & [..., n_head, seq_len, seq_len]
        output = output.reshape(*q_shape[:-1], self.n_head * self.d_v).contiguous()
                                                                               # [..., seq_len, n_head * d_v]
        output = self.dropout(self.fc_attn_output(output))                     # [..., seq_len, d_output]
        output += residual

        output = self.layer_norm(output)                                       # [..., seq_len, d_output]

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

class NonNegFFN(nn.Module):
    '''
    Feedforward module next to the Transformers layer.
    '''
    def __init__(self, d_input, d_hidden, device, dropout = 0.1):
        super(NonNegFFN, self).__init__()
        self.device = device
        
        self.w_1 = NonNegLinear(d_input, d_hidden, device = self.device)
        self.w_2 = NonNegLinear(d_hidden, d_input, device = self.device)

        self.norm = NonNegNorm(device = device)

    def forward(self, x):
        '''
        Args:
        1. x: input tensor. shape: [..., d_input]
        Outputs:
        1. output: result tensor. shape: [..., d_input]
        '''
        residual = x

        x = self.norm(x)                                                       # [..., d_input]
        x = F.softplus(self.w_1(x))                                            # [..., d_hidden]
        x = self.w_2(x)                                                        # [..., d_input]
        x += residual

        x = self.norm(x)                                                       # [..., d_input]

        return x


class NonNegNorm(nn.Module):
    '''
    First, you need to answer a question: Why would we need normalisation?
    What do we suppose these normalisation do? 
    '''
    def __init__(self, device):
        super(NonNegNorm, self).__init__()
        self.device = device

    def forward(self, x):
        dim = x.shape[-1]                                                      # [..., d_input]
        return x/dim                                                           # [..., d_input]
