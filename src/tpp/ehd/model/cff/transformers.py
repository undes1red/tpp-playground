import torch.nn as nn

from src.toolbox.modules import BiasedPositionalEmbedding, TransformerLayer


class Transformer(nn.Module):
    """A sequence to sequence model with attention mechanism."""

    def __init__(
        self,
        seq_len_x,
        seq_len_h,
        num_marks,
        d_input,
        d_rnn,
        d_hidden,
        n_layers_encoder,
        n_layers_decoder,
        n_head,
        d_qk,
        d_v,
        dropout,
        device,
    ):
        super().__init__()
        self.device = device
        self.num_marks = num_marks
        self.seq_len_h = seq_len_h
        self.seq_len_x = seq_len_x

        self.embedding = nn.Embedding(num_marks + 1, d_input, padding_idx=num_marks, device=self.device)
        self.position_embedding = BiasedPositionalEmbedding(d_input, max_len=4096, device=self.device)

        self.encoder = nn.ModuleList(
            [
                TransformerLayer(
                    d_input=d_input,
                    d_hidden=d_hidden,
                    n_head=n_head,
                    d_qk=d_qk,
                    d_v=d_v,
                    dropout=dropout,
                    device=self.device,
                )
                for _ in range(n_layers_encoder)
            ]
        )

        self.decoder = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    d_input=d_input,
                    d_hidden=d_hidden,
                    n_head=n_head,
                    d_qk=d_qk,
                    d_v=d_v,
                    dropout=dropout,
                    device=self.device,
                )
                for _ in range(n_layers_decoder)
            ]
        )

    def forward(self, marks_history, marks_future, time_history, time_future, mask_history, mask_future):
        """
        Return intensity functions' values for all marks and time and marks, if possible, predictions.
        Args:
        1. mark_time: the length of all time intervals between two adjacent marks. shape: [batch_size, seq_len]
        2. mark_type: vectors containing the information about each mark. shape: [batch_size, seq_len]
        3. non_pad_mask: padding mask. 1 refers to the existence of an mark, while 0 means a dummy mark. shape: [batch_size, seq_len]
        """
        marks_history_embedding = self.embedding(marks_history)  # [batch_size, seq_len_h, d_input]
        marks_future_embedding = self.embedding(marks_future)  # [batch_size, seq_len_x, d_input]

        time_history_embedding = self.position_embedding(self.seq_len_h, time_history)
        # [batch_size, seq_len_h, d_input]
        time_future_embedding = self.position_embedding(
            self.seq_len_x, time_future, position_start_index=self.seq_len_h
        )
        # [batch_size, seq_len_x, d_input]

        history_embedding = marks_history_embedding + time_history_embedding  # [batch_size, seq_len_h, d_input]
        future_embedding = marks_future_embedding + time_future_embedding  # [batch_size, seq_len_x, d_input]

        for enc_layer in self.encoder:
            future_embedding, _ = enc_layer(future_embedding)  # [batch_size, seq_len_x, d_input]

        for dec_layer in self.decoder:
            history_embedding = dec_layer(history_embedding, future_embedding)  # [batch_size, seq_len_h, d_input]

        return history_embedding  # [batch_size, seq_len_h, d_input]


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_input, d_hidden, n_head, d_qk, d_v, dropout, device):
        super().__init__()
        self.device = device

        # Do self-attention on the representation of following marks.
        self.self_attention = TransformerLayer(
            d_input=d_input, d_hidden=d_hidden, n_head=n_head, d_qk=d_qk, d_v=d_v, dropout=dropout, device=self.device
        )
        # Do cross-attention on the representation of following marks and historical marks.
        self.cross_attention = TransformerLayer(
            d_input=d_input, d_hidden=d_hidden, n_head=n_head, d_qk=d_qk, d_v=d_v, dropout=dropout, device=self.device
        )

    def forward(self, history_representation, future_representation):
        # Unlike the vanilla transformer, here we do not need any masks.
        # All attention modules can freely access all input marks.
        history, _ = self.self_attention(history_representation)  # [batch_size, seq_len_h, d_input]
        output, _ = self.cross_attention(q=history, k=future_representation, v=future_representation)
        # [batch_size, seq_len_h, d_input]

        return output
