import torch.nn as nn
from einops import rearrange

from src.toolbox.modules import BiasedPositionalEmbedding, TransformerLayer, get_causal_mask


class TransEncoder(nn.Module):
    def __init__(self, num_marks, d_input, d_hidden, n_layers, n_head, d_qk, d_v, dropout, device):
        """
        This function builds a Transformer encoder.

        ### Args
          * ```int``` num_types
            The number of all possible marks.
          * ```torch.device``` device
            The device where we place this transformer encoder.
          * ```int``` d_input
            The dimension of the Transformer input tensor.
          * ```int``` d_rnn
            The dimension of RNN's hidden state.
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
        # FIXME: set max_len during runtime, current max_len = 4096
        self.position_emb = BiasedPositionalEmbedding(d_input, max_len=4096, device=self.device)

        # mark type embedding
        self.mark_emb = nn.Embedding(num_marks + 1, d_input, padding_idx=num_marks, device=self.device)

        # history time encoder
        self.history_time_emb = nn.Linear(1, d_input, device=self.device)

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
                for _ in range(n_layers)
            ]
        )

    def forward(self, time_history, marks_history, non_pad_mask):
        """
        Encode the input continuous-time mark stream using Transformer.

        ### Args
          * ```torch.tensor``` time_history
            shape: ```[batch_size, seq_len]```
            The length of all time intervals between two adjacent marks.
          * ```torch.tensor``` marks_history
            shape: ```[batch_size, seq_len]```
            Vectors containing the information about each mark.
          * ```torch.tensor``` non_pad_mask
            shape: ```[batch_size, seq_len]```
            Padding mask. 1 refers to the existence of an mark, while 0 means a dummy mark.
        ### Outputs
            * ```torch.tensor``` output
              shape: ```[batch_size, seq_len, d_input]```
              The representation of the original input.
        """
        # prepare attention masks
        # self_attn_mask is where we cannot look, i.e., the future and the padding
        seq_len = marks_history.shape[-1]
        self_attn_mask_subseq = get_causal_mask(seq_len, device=self.device)
        # [batch_size, seq_len, seq_len]
        self_attn_mask_keypad = rearrange(non_pad_mask, "b s -> b () s")  # [batch_size, seq_len, seq_len]
        self_attn_mask = self_attn_mask_keypad & self_attn_mask_subseq  # [batch_size, seq_len, seq_len]

        # Mark Embeddings
        marks_emb = self.mark_emb(marks_history)  # [batch_size, seq_len, d_input]
        # Time Embeddings
        time_emb = self.position_emb(seq_len, time_history)  # [batch_size, seq_len, d_input]
        output = marks_emb + time_emb  # [batch_size, seq_len, d_input]

        for enc_layer in self.encoder:
            """
            history mark sequence
            """
            output, _ = enc_layer(
                output, output, output, non_pad_mask=non_pad_mask, self_attn_mask=self_attn_mask
            )  # [batch_size, seq_len, d_input]

        return output
