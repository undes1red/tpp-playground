import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self, d_input, device, max_len=16384):
        super().__init__()
        self.device = device
        self.max_len = max_len

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_input, device=self.device, requires_grad=False)
        position = torch.arange(0, max_len, device=self.device).unsqueeze(1)
        div_term = (
            torch.arange(0, d_input, 2, device=self.device)
            * -(torch.log(torch.tensor(10000.0, device=self.device)) / d_input)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x, seq_len_dim=-1):
        # x shape: [batch_size, seq_len]
        seq_len = x.shape[seq_len_dim]
        if seq_len <= self.max_len:
            return self.pe[:seq_len, :]
        raise ValueError(f"Input sequence (length: {seq_len}) is longer than the max length, which is {self.max_len}.")
