import torch.nn as nn

from src.ehd.model.cff.transformers import Transformer


class EHDBackend(nn.Module):
    """
    This is our implementation of Omi's paper: Fully Neural Network based Model for General Temporal Point Processes
    Hope it can work properly.

    Currently, normalization is disabled.
    Update: 2022-01-19: Now you can use data normalization via synthetic dataloader.

    Following Babylon's paper, we would check the performance of FullyNN with integral offsets.
    """

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
        self.seq_len_x = seq_len_x
        self.seq_len_h = seq_len_h

        self.seq_encoder = Transformer(
            seq_len_x=seq_len_x + 1,
            seq_len_h=seq_len_h + 1,
            num_marks=num_marks,
            d_input=d_input,
            d_rnn=d_rnn,
            d_hidden=d_hidden,
            n_layers_encoder=n_layers_encoder,
            n_head=n_head,
            d_qk=d_qk,
            d_v=d_v,
            n_layers_decoder=n_layers_decoder,
            dropout=dropout,
            device=self.device,
        )

        # We get two marks: Should we remove it or not.
        self.remove_mark = nn.Linear(d_input, 2, device=self.device)
        self.normalize = nn.Softmax(dim=-1)

    def forward(self, marks_history, marks_future, time_history, time_future, mask_history, mask_future):
        """
        Args:
            marks_history:  [batch_size, seq_len]
            time_history:   [batch_size, seq_len]
            time_next:      [batch_size, seq_len, num_marks] if we need marks else [batch_size, seq_len]
            mask:           [batch_size, seq_len]
        """

        """
        Prepare the input.
        """
        seq_embedding = self.seq_encoder(
            marks_history, marks_future, time_history, time_future, mask_history, mask_future
        )
        # [batch_size, seq_len_h, d_input]
        generated_un_probability_masked = self.remove_mark(seq_embedding)  # [batch_size, seq_len_h, 2]

        # Here we get the probability p(y = 1) and p(y = 0).
        return self.normalize(generated_un_probability_masked)
        # [batch_size, seq_len_h, 2]
