import torch
import torch.nn as nn
from einops import rearrange, repeat

from src.toolbox.modules import FMHSA, AttNHPTimeEmbedding, PositionalEmbedding


class TransformerTPP(nn.Module):
    """A sequence to sequence model with attention mechanism."""

    def __init__(
        self,
        training,
        m,
        M,
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

    def forward(
        self, time_history, time_next, marks_history, non_pad_mask, custom_marks_history, integration_sample_rate
    ):
        """
        Return intensity functions' values for all marks and time and marks, if possible, predictions.
        Args:
        1. mark_time: the length of all time intervals between two adjacent marks. shape: [batch_size, seq_len]
        2. mark_type: vectors containing the information about each mark. shape: [batch_size, seq_len]
        3. non_pad_mask: padding mask. 1 refers to the existence of an mark, while 0 means a dummy mark. shape: [batch_size, seq_len]
        """

        return self.encoder(
            time_history, time_next, marks_history, non_pad_mask, custom_marks_history, integration_sample_rate
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
                FMHSA(
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
        self, time_history, time_next, marks_history, non_pad_mask, custom_marks_history, integration_sample_rate
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

        # calculate the integral
        time_multiplier = torch.linspace(0, 1, self.integration_sample_rate, device=self.device)
        expanded_time = (
            time_next.unsqueeze(dim=-1) * time_multiplier
        )  # [..., batch_size, seq_len, integration_sample_rate]

        sample_time = rearrange(
            time_next, "... b s sr -> ... sr () b s"
        )  # [..., sample_rate, num_mark, batch_size, seq_len]

        sample_mark = torch.arange(self.num_marks, device=self.device)  # [num_mark]
        sample_mark = repeat(sample_mark, "ne -> sr ne b s", sr=integration_sample_rate, b=batch_size, s=seq_len)
        # [sample_rate, num_mark, batch_size, seq_len]

        time_history = repeat(time_history, "b s -> sr () b s", sr=integration_sample_rate)
        # [sample_rate, num_mark, batch_size, seq_len]
        if len(sample_time.shape) > 4:
            einop = "... -> "
            parameter_dict = {}
            for idx, val in enumerate(sample_time.shape[:-4]):
                einop += f"a{idx} "
                parameter_dict[f"a{idx}"] = val

            einop += "..."
            time_history = repeat(
                time_history, einop, **parameter_dict
            )  # [..., sample_rate, num_mark, batch_size, seq_len]

        marks_history = repeat(marks_history, "b s -> sr ne b s", sr=self.integration_sample_rate, ne=self.num_marks)
        # [sample_rate, num_mark, batch_size, seq_len]

        # Connect history with samples, further we perform masked attention on this sequence.
        connected_mark_seq = torch.cat((marks_history, sample_mark), dim=-1)
        # [sample_rate, num_mark, batch_size, 2 * seq_len]
        """
        Prepare attention masks
        AttNHP's attention mask should be carefully handled. It should ensure:
        1. each sample_mark only sees itself and history marks it should see.
        2. each mark in history only sees eariler marks. It should not know the existence of sample_mark.
        3. padding marks and EOS are invisible to history_marks and sample_marks.
        """
        self_attn_mask_from_history_to_history = get_subsequent_mask(seq_len, device=self.device)
        # [batch_size, seq_len, seq_len]
        self_attn_mask_from_history_to_sample = torch.zeros_like(self_attn_mask_from_history_to_history)
        # [batch_size, seq_len, seq_len]
        self_attn_mask_from_sample_to_history = self_attn_mask_from_history_to_history
        # [batch_size, seq_len, seq_len]
        self_attn_mask_from_sample_to_sample = torch.eye(seq_len, dtype=torch.uint8, device=self.device)
        self_attn_mask_from_sample_to_sample = rearrange(self_attn_mask_from_sample_to_sample, "s s1 -> () s s1")
        # [batch_size, seq_len, seq_len]
        self_attn_mask_from_history_all = torch.cat(
            (self_attn_mask_from_history_to_history, self_attn_mask_from_history_to_sample), dim=-1
        )
        # [batch_size, seq_len, seq_len * 2]
        self_attn_mask_from_sample_all = torch.cat(
            (self_attn_mask_from_sample_to_history, self_attn_mask_from_sample_to_sample), dim=-1
        )
        # [batch_size, seq_len, seq_len * 2]
        self_attn_mask = torch.cat((self_attn_mask_from_history_all, self_attn_mask_from_sample_all), dim=-2)
        # [batch_size, seq_len * 2, seq_len * 2]

        non_pad_mask = torch.cat((non_pad_mask, torch.ones_like(non_pad_mask)), dim=-1)
        # [batch_size, seq_len * 2]
        non_pad_mask_with_sample = rearrange(non_pad_mask, "b s -> b () s")  # [batch_size, seq_len * 2, seq_len * 2]

        self_attn_mask = self_attn_mask & non_pad_mask_with_sample  # [batch_size, seq_len * 2, seq_len * 2]

        # Time Embedding
        time_history_emb = self.position_emb(
            seq_len, time_history
        )  # [..., sample_rate, num_marks, batch_size, seq_len, d_input]
        sample_time_emb = self.position_emb(seq_len, sample_time, position_start_index=1)
        # [..., sample_rate, num_marks, batch_size, seq_len, d_input]
        time_emb = torch.cat(
            (time_history_emb, sample_time_emb), dim=-2
        )  # [..., sample_rate, num_marks, batch_size, seq_len * 2, d_input]

        # mark Embedding
        if marks_history != None:
            if custom_marks_history:
                marks_emb = marks_history
            else:
                marks_emb = self.mark_emb(
                    connected_mark_seq
                )  # [sample_rate, num_mark, batch_size, seq_len * 2, d_input]
                einop = f"... -> {'() ' * (len(time_emb.shape) - 5)}..."
                marks_emb = rearrange(
                    marks_emb, einop
                )  # [..., sample_rate, num_mark, batch_size, seq_len * 2, d_input]
        else:
            marks_emb = torch.zeros_like(
                time_emb, device=self.device
            )  # [..., sample_rate, num_mark, batch_size, seq_len * 2, d_input]

        mingled_emb = time_emb + marks_emb  # [..., sample_rate, num_mark, batch_size, seq_len * 2, d_input]

        for enc_layer in self.layer_stack:
            mingled_emb, _ = enc_layer(mingled_emb, non_pad_mask=non_pad_mask, self_attn_mask=self_attn_mask)
            # [..., sample_rate, num_mark, batch_size, seq_len * 2, d_input]

        return mingled_emb

    def get_mark_embedding(self, input_mark):
        return self.mark_emb(input_mark)  # [batch_size, seq_len, d_input]
