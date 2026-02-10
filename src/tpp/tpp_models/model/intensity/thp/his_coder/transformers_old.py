import torch
import torch.nn as nn

from src.toolbox.modules import PositionalEmbedding, THPTimeEmbedding, TransformerLayer


class Encoder(nn.Module):
    def __init__(self, num_marks, d_input, d_hidden, n_layers, n_head, d_qkv, dropout, device):
        """
        This function builds a Transformer encoder.

        ### Args
          * ```int``` num_marks
            The number of all possible marks.
          * ```torch.device``` device
            The device where we place this transformer encoder.
          * ```int``` d_input
            The dimension of the Transformer input tensor.
          * ```int``` d_hidden
              The dimension of the FFN module in the Transformer.
          * ```int``` n_layers
              The number of self attention + FFN layers in the Transformer.
          * ```int``` n_head
            The number of head in self attention.
          * ```int``` d_qk
            The dimension of matrices Q and K.
          * ```int``` d_v
            The dimension of metrix V.
          * ```float``` dropout
            Dropout rate for the history encoder.
        """
        super().__init__()
        self.device = device
        self.d_input = d_input
        self.num_marks = num_marks

        # position vector, used for temporal encoding
        self.position_emb = PositionalEmbedding(d_input, max_len=4096, device=self.device)
        self.time_emb = THPTimeEmbedding(d_input, device=self.device)

        # event type embedding
        self.event_emb = nn.Embedding(num_marks + 1, d_input, padding_idx=num_marks, device=self.device)

        self.layer_stack = nn.ModuleList(
            [
                TransformerLayer(
                    d_input=d_input,
                    d_hidden=d_hidden,
                    n_head=n_head,
                    d_qk=d_qkv,
                    d_v=d_qkv,
                    dropout=dropout,
                    device=self.device,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, event_time, event_type, non_pad_mask):
        """
        Encode the input continuous-time event stream using Transformer.

        ### Args
          * ```torch.tensor``` event_time
            shape: ```[batch_size, seq_len]```
            The length of all time intervals between two adjacent mark.
          * ```torch.tensor``` event_type
            shape: ```[batch_size, seq_len]```
            Vectors containing the information about each event.
          * ```torch.tensor``` non_pad_mask
            shape: ```[batch_size, seq_len]```
            Padding mask. 1 refers to the existence of an event, while 0 means a dummy event.
        ### Outputs
            * ```torch.tensor``` output
              shape: ```[batch_size, seq_len, d_input]```
              The representation of the original input.
        """
        # Time Embedding
        pos_emb = self.position_emb(event_time)  # [batch_size, seq_len, d_input]
        time_emb = self.time_emb(event_time)  # [batch_size, seq_len, d_input]

        if event_type is not None:
            mark_emb = self.event_emb(event_type)  # [batch_size, seq_len, d_input]
        else:
            mark_emb = torch.zeros_like(time_emb, device=self.device)  # [batch_size, seq_len, d_input]

        output = pos_emb + time_emb + mark_emb  # [batch_size, seq_len, d_input]
        for enc_layer in self.layer_stack:
            output, _ = enc_layer(
                output, non_pad_mask=non_pad_mask
            )  # [batch_size, seq_len, d_input]
        return output


class TransformerTPP(nn.Module):
    def __init__(self, training, num_marks, device, d_input, d_hidden, n_layers, n_head, d_qkv, dropout):
        """
        This function builds a Transformer encoder.

        ### Args
          * ```int``` num_marks
            The number of all possible marks.
          * ```torch.device``` device
            The device where we place this transformer encoder.
          * ```int``` d_input
            The dimension of the Transformer input tensor.
          * ```int``` d_hidden
              The dimension of the FFN module in the Transformer.
          * ```int``` n_layers
              The number of self attention + FFN layers in the Transformer.
          * ```int``` n_head
            The number of head in self attention.
          * ```int``` d_qk
            The dimension of matrices Q and K.
          * ```int``` d_v
            The dimension of metrix V.
          * ```float``` dropout
            Dropout rate for the history encoder.
        """
        super().__init__()
        self.device = device
        self.num_marks = num_marks if num_marks > 0 else 1
        dropout = dropout if training else 0

        self.encoder = Encoder(
            num_marks=self.num_marks,
            d_input=d_input,
            d_hidden=d_hidden,
            n_layers=n_layers,
            n_head=n_head,
            d_qkv=d_qkv,
            dropout=dropout,
            device=self.device,
        )

    def forward(self, event_time, event_type, non_pad_mask):
        """
        Encode the input continuous-time event stream using Transformer.

        ### Args
          * ```torch.tensor``` event_time
            shape: ```[batch_size, seq_len]```
            The length of all time intervals between two adjacent mark.
          * ```torch.tensor``` event_type
            shape: ```[batch_size, seq_len]```
            Vectors containing the information about each event.
          * ```torch.tensor``` non_pad_mask
            shape: ```[batch_size, seq_len]```
            Padding mask. 1 refers to the existence of an event, while 0 means a dummy event.
        ### Outputs
            * ```torch.tensor``` enc_output
              shape: ```[batch_size, seq_len, d_input]```
              The representation of the original input.
        """
        return self.encoder(event_time, event_type, non_pad_mask)  # [batch_size, seq_len, d_input]
