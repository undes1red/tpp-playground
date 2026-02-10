import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange

from src.toolbox.modules.ffn import FFN

from .get_causal_mask import get_causal_mask


class TransformerLayer(nn.Module):
    def __init__(self, n_head, d_input, d_qk, d_v, device, d_hidden, dropout=0.1):
        super().__init__()
        self.device = device

        self.attn = MultiheadAttention(
            n_head=n_head, d_input=d_input, d_qk=d_qk, d_v=d_v, device=self.device, dropout=dropout
        )
        self.ffn = FFN(d_input=d_input, d_hidden=d_hidden, device=self.device, dropout=dropout)

    def forward(self, x, non_pad_mask=None):
        """
        Args:
        1. x: input tensor. shape: [batch_size, seq_len, d_input]
        2. self_attn_mask: mask tensor for used by self attention. shape: [seq_len, seq_len]
        3. pad_mask: mask out pad items' output values. shape: [batch_size, seq_len, d_attn_input]
        Outputs:
        """
        output = self.attn(
            x, mask=non_pad_mask
        )  # [batch_size, seq_len, d_input] & [batch_size, n_head, seq_len, seq_len]

        output = self.ffn(output)  # [batch_size, seq_len, d_input]

        if non_pad_mask is not None:
            output *= rearrange(non_pad_mask, "... -> ... 1")  # [batch_size, seq_len, d_input]

        return output


class MultiheadAttention(nn.Module):
    def __init__(self, n_head, d_input, d_qk, d_v, device, dropout=0.1):
        """
        Template self-attention module with multihead-attention type 2: this module concatenates original outputs and
        compress high-dimensional vectors into d_input.
        """
        super().__init__()
        self.device = device

        self.d_input = d_input
        self.n_head = n_head
        self.d_q = d_qk
        self.d_k = d_qk
        self.d_v = d_v
        self.dropout = dropout

        # Linear: d_input -> d_q, d_k, or d_v
        self.w_q = nn.Linear(d_input, self.d_q * self.n_head, bias=False, device=self.device)
        self.w_k = nn.Linear(d_input, self.d_k * self.n_head, bias=False, device=self.device)
        self.w_v = nn.Linear(d_input, self.d_v * self.n_head, bias=False, device=self.device)

        # Self-attention module
        self.self_attn = SelfAttn(temperature=d_qk**0.5, attn_dropout=self.dropout, device=self.device)

        # Linear: n_head * d_q, d_k, or d_v -> d_input
        self.fc_attn_output = nn.Linear(self.n_head * d_v, self.d_input, bias=True, device=self.device)

        # Dropout
        self.dropout = nn.Dropout(self.dropout)

        # Layer Norm
        self.layer_norm = nn.RMSNorm(d_input, eps=1e-6, device=self.device)

    def forward(self, x, mask=None):
        """
        Args:
        1. x: input tensor. shape: [batch_size, seq_len, d_input]
        2. mask: the mask tensor used by self attention. shape: [seq_len, seq_len]
        Output:
        1. output: results of transformer layer. shape: [batch_size, seq_len, d_output]
        """
        if mask.dtype != torch.bool:
            mask = mask.to(torch.bool)  # [batch_size, seq_len]

        seq_len = mask.shape[-1]
        causal_mask = get_causal_mask(seq_len, device=self.device)  # [batch_size, seq_len, seq_len]
        attn_mask = (
            rearrange(causal_mask, "b s s1 -> b () s s1")
            & rearrange(mask, "b s -> b () () s")
            & rearrange(mask, "b s -> b () s ()")
        )
        # [batch_size, nhead, seq_len, seq_len]

        residual = x
        x = self.layer_norm(x)  # [batch_size, seq_len, n_head, d_qk]

        # preparing for q, k, and v.
        q = rearrange(self.w_q(x), "... (nh dq) -> ... nh dq", nh=self.n_head)
        # [batch_size, seq_len, n_head, d_qk]
        k = rearrange(self.w_k(x), "... (nh dk) -> ... nh dk", nh=self.n_head)
        # [batch_size, seq_len, n_head, d_qk]
        v = rearrange(self.w_v(x), "... (nh dv) -> ... nh dv", nh=self.n_head)
        # [batch_size, seq_len, n_head, d_v]

        output = self.self_attn(
            q, k, v, mask=attn_mask
        )  # [batch_size, seq_len, n_head, d_v] & [batch_size, n_head, seq_len, seq_len]
        output = rearrange(output, "...  nh dv -> ... (nh dv)", nh=self.n_head)
        # [batch_size, seq_len, n_head * d_v]

        # Replace NaN with 0 (can occur when padded positions have all -inf in attention)
        output = torch.nan_to_num(output, nan=0.0)

        output = self.dropout(self.fc_attn_output(output))  # [batch_size, seq_len, d_output]

        output += residual  # [batch_size, seq_len, d_output]

        return output


class SelfAttn(nn.Module):
    """
    SelfAttn module, the heart of transformers' layer
    """

    def __init__(self, temperature, attn_dropout, device):
        super().__init__()
        self.device = device
        self.temperature = temperature

        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        """
        Args:

        1. q: input tensor. shape: [batch_size, seq_len_q, n_head, d_qk]
        2. k: input tensor. shape: [batch_size, seq_len_k, n_head, d_qk]
        3. v: input tensor. shape: [batch_size, seq_len_k, n_head, d_v]
        4. mask: mask_out several values in the attention matrices. shape: [batch_size, n_head, seq_len_q, seq_len_k]

        Output:
        1. output: the result of self attention. shape: [batch_size, seq_len, n_head, d_v]
        """

        q /= self.temperature  # [batch_size, seq_len_q, n_head, d_qk]

        attn = einsum(
            q, k, "... slq nh dqk, ... slk nh dqk -> ... nh slq slk"
        )  # [batch_size, n_head, seq_len_q, seq_len_k]

        if mask is not None:
            attn = attn.masked_fill(~mask, -torch.inf)  # [batch_size, n_head, seq_len_q, seq_len_k]

        # F.softmax() uses float32 by default.
        # This means it will upcast the input to float32 and the output is also float32.
        # We send it the dtype of attn to force it to respect the precision cast.
        # We also have reports saying low precision softmax is GPU only.
        # Please check https://github.com/huggingface/transformers/issues/27341 for further information.
        attn = self.dropout(F.softmax(attn, dim=-1, dtype=attn.dtype))  # [batch_size, n_head, seq_len_q, seq_len_k]
        return einsum(attn, v, "... nh slq slk, ... slk nh dv -> ... slq nh dv")  # [batch_size, seq_len_q, n_head, d_v]
