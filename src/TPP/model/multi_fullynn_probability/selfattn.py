import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttn(nn.Module):
    '''
    SelfAttn module, the heart of transformers' layer
    '''
    def __init__(self, temperature, attn_dropout, device, wq_nonneg, wk_nonneg, wv_nonneg):
        super(SelfAttn, self).__init__()
        self.device = device
        self.temperature = temperature
        self.wq_nonneg = wq_nonneg
        self.wk_nonneg = wk_nonneg
        self.wv_nonneg = wv_nonneg

        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask = None):
        '''
        Args:

        1. q: input tensor. shape: [batch_size, seq_len, n_head, d_qk]
        2. k: input tensor. shape: [batch_size, seq_len, n_head, d_qk]
        3. v: input tensor. shape: [batch_size, seq_len, n_head, d_v]
        4. mask: mask_out several values in the attention matrices. shape: [seq_len, seq_len]

        Output:
        1. output: the result of self attention. shape: [batch_size, seq_len, n_head, d_v]
        '''

        q /= self.temperature                                                  # [batch_size, seq_len, n_head, d_qk]

        attn = torch.einsum('...jkl, ...mkl -> ...kjm', q, k)                  # [batch_size, n_head, seq_len, seq_len]

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1), -1e9)                   # [batch_size, n_head, seq_len, seq_len]
        
        if self.wq_nonneg or self.wk_nonneg:
            attn = self.dropout(torch.sigmoid(attn))                           # [batch_size, n_head, seq_len, seq_len]
        else:
            attn = self.dropout(F.softmax(attn, dim = -1))                     # [batch_size, n_head, seq_len, seq_len]
        
        if self.wv_nonneg and not self.wq_nonneg and not self.wk_nonneg:
            attn = F.softplus(attn)                                            # [batch_size, n_head, seq_len, seq_len]

        output = torch.einsum('...jkl, ...ljn -> ...kjn', attn, v)             # [batch_size, seq_len, n_head, d_v]

        return output, attn