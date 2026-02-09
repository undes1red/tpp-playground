import math

import torch
import torch.nn as nn
from einops import rearrange, repeat
from flash_attn import flash_attn_func

from src.toolbox.modules import FFN, AttNHPTimeEmbedding, PositionalEmbedding

max_cuda_batch = 65535


class TransformerTPP(nn.Module):
    """A sequence to sequence model with attention mechanism."""

    def __init__(
        self,
        training,
        num_marks,
        device,
        d_input,
        d_hidden,
        n_layers,
        n_head,
        d_qkv,
        dropout,
    ):
        super().__init__()
        self.device = device
        self.training = training
        self.num_marks = num_marks if num_marks > 0 else 1

        self.encoder = Encoder(
            training=training,
            num_marks=self.num_marks,
            d_input=d_input,
            d_hidden=d_hidden,
            n_layers=n_layers,
            n_head=n_head,
            d_qkv=d_qkv,
            dropout=dropout,
            device=self.device,
        )

    def forward(self, time_history, time_next, marks_history, custom_marks_history=False):
        """
        Return intensity functions' values for all marks and time and marks, if possible, predictions.
        Args:
        1. mark_time: the length of all time intervals between two adjacent marks. shape: [batch_size, seq_len]
        2. mark_type: vectors containing the information about each mark. shape: [batch_size, seq_len]
        3. non_pad_mask: padding mask. 1 refers to the existence of an mark, while 0 means a dummy mark. shape: [batch_size, seq_len]
        """

        return self.encoder(
            time_history, time_next, marks_history, custom_marks_history=custom_marks_history
        )
        # [batch_size, seq_len, d_input]

    def get_mark_embedding(self, input_mark):
        return self.encoder.get_mark_embedding(input_mark)  # [batch_size, seq_len, d_input]


class Encoder(nn.Module):
    """A encoder model with self attention mechanism."""

    def __init__(
        self,
        training,
        num_marks,
        d_input,
        d_hidden,
        n_layers,
        n_head,
        d_qkv,
        dropout,
        device,
    ):
        super().__init__()
        self.device = device
        self.d_input = d_input
        self.num_marks = num_marks

        # position vector, used for temporal encoding
        # FIXME: set max_len during runtime, current max_len = 4096
        self.position_emb = PositionalEmbedding(d_input, max_len=4096, device=self.device)
        self.time_emb = AttNHPTimeEmbedding(d_input, device=self.device)
        # mark type embedding
        self.mark_emb = nn.Embedding(num_marks + 1, d_input, padding_idx=num_marks, device=self.device)

        self.layer_stack = nn.ModuleList(
            [
                FMHCA(
                    training=training,
                    n_head=n_head,
                    d_input=d_input,
                    d_qkv=d_qkv,
                    d_hidden=d_hidden,
                    dropout=dropout,
                    device=self.device,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(
        self,
        time_history,
        expanded_time_next,
        marks_history,
        custom_marks_history,
    ):
        """
        Encode mark sequences via masked self-attention.
        Args:
        1. time_history: input time intervals. shape: [batch_size, seq_len]
        2. sample_time: shape: [..., batch_size, seq_len, sample_rate]
        3. marks_history: shape: [batch_size, seq_len]
        4. non_pad_mask: pad mask tensor. shape: [batch_size, seq_len]
        """
        batch_size, seq_len = marks_history.shape
        resolution = expanded_time_next.shape[-1]

        # [..., batch_size, seq_len, resolution]
        expanded_time_emb = repeat(
            self.time_emb(expanded_time_next, resolution_dim=True), "... di -> ... ne di", ne=self.num_marks
        )
        # [..., batch_size, seq_len, resolution, num_marks, d_input]
        # Compute all_mark_emb on-the-fly to save memory
        all_mark_emb = self.mark_emb.weight[:self.num_marks]  # [num_marks, d_input]
        all_mark_emb = rearrange(
            all_mark_emb, f"nm di -> {' '.join(['()'] * (len(expanded_time_emb.shape) - 2))} nm di"
        )
        all_mark_emb = all_mark_emb.expand(*expanded_time_emb.shape)
        # [..., batch_size, seq_len, resolution, num_marks, d_input]
        one_in_input = torch.ones(*all_mark_emb.shape[:-1], 1, device=self.device)
        # [..., batch_size, seq_len, resolution, num_marks, 1]
        expanded_time_emb = torch.cat((one_in_input, expanded_time_emb, all_mark_emb), dim=-1)
        # [..., batch_size, seq_len, resolution, num_marks, 2*d_input+1]

        time_history_emb = self.time_emb(time_history)
        # [batch_size, seq_len, d_input]
        if marks_history is not None:
            marks_history_emb = marks_history if custom_marks_history else self.mark_emb(marks_history)
            # [batch_size, seq_len, d_input]
        else:
            marks_history_emb = torch.zeros_like(time_history_emb)
            # [batch_size, seq_len, d_input]

        one_in_history = torch.ones(*time_history_emb.shape[:-1], 1, device=self.device)
        # [batch_size, seq_len, 1]
        history_emb = torch.cat((one_in_history, time_history_emb, marks_history_emb), dim=-1)
        # [batch_size, seq_len, 2*d_input+1]
        history_emb = repeat(history_emb, "b s d -> b s r ne d", r=resolution, ne=self.num_marks)
        # [batch_size, seq_len, resolution, num_marks, 2*d_input+1]

        """
        Flat the history and expanded_time_emb for attention.
        """
        history_emb = rearrange(history_emb, "b s r ne d -> (b r ne) s d")
        # [batch_size * resolution * num_marks, seq_len, 2*d_input+1]
        expanded_time_emb = rearrange(expanded_time_emb, "... b s r ne d -> ... (b r ne) s d")
        # [..., batch_size * resolution * num_marks, seq_len, 2*d_input+1]

        # handling dimension before batch_size.
        additional_shape = None
        if len(expanded_time_emb.shape) > 3:
            additional_shape = expanded_time_emb.shape[:-3]
            numbers_before_batch_size = math.prod(additional_shape)
            history_emb = repeat(history_emb, "b s di -> (d b) s di", d=numbers_before_batch_size)
            # [... * batch_size * resolution * num_marks, seq_len, d_input]
            einop = f"{' '.join([f'a{idx}' for idx in range(len(additional_shape))])} b ... -> ({' '.join([f'a{idx}' for idx in range(len(additional_shape))])} b) ..."
            expanded_time_emb = rearrange(expanded_time_emb, einop)
            # [... * batch_size * resolution * num_marks, seq_len, d_input]

        attached_tensor_kv, e_kv = torch.split(history_emb, (self.d_input + 1, self.d_input), dim=-1)
        # [... * batch_size * resolution * num_marks, seq_len, d_input+1] + [... * batch_size * resolution * num_marks, seq_len, d_input]
        attached_tensor_q, e_q = torch.split(expanded_time_emb, (self.d_input + 1, self.d_input), dim=-1)
        # [... * batch_size * resolution * num_marks, seq_len, d_input+1] + [... * batch_size * resolution * num_marks, seq_len, d_input]

        for layer in self.layer_stack:
            e_q = layer(
                e_q,
                e_kv,
                e_kv,
                attached_tensor_q=attached_tensor_q,
                attached_tensor_k=attached_tensor_kv,
                attached_tensor_v=attached_tensor_kv,
            )
            # [... * batch_size * resolution * num_marks, seq_len, d_input]

        if additional_shape is not None:
            einop = f"({' '.join([f'a{idx}' for idx in range(len(additional_shape))])} b) ... -> {' '.join([f'a{idx}' for idx in range(len(additional_shape))])} b ..."
            einop_dict = {f"a{idx}": val for idx, val in enumerate(additional_shape)}

            e_q = rearrange(e_q, einop, **einop_dict)
            # [..., batch_size * resolution * num_marks, seq_len, d_input]

        return rearrange(e_q, "... (b r ne) s d -> ... b s r ne d", b=batch_size, r=resolution)
        # [..., batch_size, seq_len, resolution, num_marks, d_input]

    def get_mark_embedding(self, input_mark):
        return self.mark_emb(input_mark)  # [batch_size, seq_len, d_input]


class FMHCA(nn.Module):
    def __init__(self, training, n_head, d_input, d_qkv, device, d_hidden, dropout=0.1):
        super().__init__()
        self.training = training
        self.device = device

        self.attn = FMHCALayer(
            training=training, n_head=n_head, d_input=d_input, d_qkv=d_qkv, device=self.device, dropout=dropout
        )
        self.ffn = FFN(d_input=d_input, d_hidden=d_hidden, device=self.device, dropout=dropout)

    def forward(
        self, q, k, v, attached_tensor_q=None, attached_tensor_k=None, attached_tensor_v=None, non_pad_mask=None
    ):
        """
        Args:
        1. x: input tensor. shape: [batch_size, seq_len, d_input]
        2. non_pad_mask: mask tensor for used by self attention and to mask out pad items. shape: [batch_size, seq_len]
        Outputs:
        1. output: results of transformer layer. shape: [batch_size, seq_len, d_input]
        """
        output = self.attn(
            q,
            k,
            v,
            attached_tensor_q=attached_tensor_q,
            attached_tensor_k=attached_tensor_k,
            attached_tensor_v=attached_tensor_v,
        )
        # [..., batch_size, seq_len, d_input]

        output = self.ffn(output)  # [..., batch_size, seq_len, d_input]

        if non_pad_mask is not None:
            output *= rearrange(non_pad_mask, "... -> ... 1")  # [..., batch_size, seq_len, d_input]

        return output


class FMHCALayer(nn.Module):
    def __init__(self, training, n_head, d_input, d_qkv, device, dropout=0.1):
        super().__init__()
        self.training = training
        self.device = device
        if device == "cpu":
            raise ValueError("Flash Attention does not work on CPU.")

        self.d_input = d_input
        self.n_head = n_head
        self.d_qkv = d_qkv
        self.dropout = dropout if self.training else 0

        # Linear: d_input -> d_q, d_k, or d_v
        self.w_q = nn.Linear(2 * d_input + 1, self.d_qkv * self.n_head, bias=True, device=self.device)
        self.w_k = nn.Linear(2 * d_input + 1, self.d_qkv * self.n_head, bias=True, device=self.device)
        self.w_v = nn.Linear(2 * d_input + 1, self.d_qkv * self.n_head, bias=True, device=self.device)

        # Linear: n_head * d_q, d_k, or d_v -> d_input
        self.fc_attn_output = nn.Linear(self.n_head * d_qkv, self.d_input, bias=True, device=self.device)

        # layer normalization
        self.layer_norm = nn.RMSNorm(2 * d_input + 1, eps=1e-6, device=self.device, dtype=torch.get_default_dtype())

    def forward(
        self,
        q,
        k,
        v,
        attached_tensor_q,
        attached_tensor_k,
        attached_tensor_v,
    ):
        """
        Args:
        1. q, k, v: input tensor. shape: [batch_size, seq_len, d_input]
        2. mask: the mask tensor used by self attention. shape: [batch_size, seq_len]
        Output:
        1. output: results of transformer layer. shape: [batch_size, seq_len, d_input]
        """

        residual = q

        # In-place concatenation and normalization to reduce intermediate tensors
        q = self.layer_norm(torch.cat((attached_tensor_q, q), dim=-1))  # [batch_size, seq_len, 2*d_input+1]
        k = self.layer_norm(torch.cat((attached_tensor_k, k), dim=-1))  # [batch_size, seq_len, 2*d_input+1]
        v = self.layer_norm(torch.cat((attached_tensor_v, v), dim=-1))  # [batch_size, seq_len, 2*d_input+1]

        q_ = self.w_q(q)  # [batch_size, seq_len, d_qkv * n_head]
        k_ = self.w_k(k)  # [batch_size, seq_len, d_qkv * n_head]
        v_ = self.w_v(v)  # [batch_size, seq_len, d_qkv * n_head]

        q_ = rearrange(q_, "b s (head dqkv) -> b s head dqkv", head=self.n_head)
        k_ = rearrange(k_, "b s (head dqkv) -> b s head dqkv", head=self.n_head)
        v_ = rearrange(v_, "b s (head dqkv) -> b s head dqkv", head=self.n_head)

        if q.shape[0] > max_cuda_batch:
            chunks_q = torch.split(q_, max_cuda_batch)
            chunks_k = torch.split(k_, max_cuda_batch)
            chunks_v = torch.split(v_, max_cuda_batch)

            fa_output = []
            for cq, ck, cv in zip(chunks_q, chunks_k, chunks_v):
                out = flash_attn_func(cq, ck, cv, dropout_p=self.dropout, causal=True, deterministic=True)
                fa_output.append(out)
            fa_output = torch.cat(fa_output, dim=0)
            # [batch_size, seq_len, nheads, d_qkv]
        else:
            fa_output = flash_attn_func(
                q_, k_, v_, dropout_p=self.dropout, causal=True, deterministic=True
            )  # [batch_size, seq_len, nheads, d_qkv]

        fa_output = rearrange(fa_output, "...  nh dqkv -> ... (nh dqkv)")
        # [batch_size, seq_len, n_head * d_qkv]

        fa_output = self.fc_attn_output(fa_output)  # [batch_size, seq_len, d_input]
        fa_output += residual  # [batch_size, seq_len, d_input]

        return fa_output  # [batch_size, seq_len, d_output]
