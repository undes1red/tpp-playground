import torch
import torch.nn as nn


class THPTimeEmbedding(nn.Module):
    """Time embedding method posted in the THP paper.

    z[t_j]_i = cos(t_j/10000^{\\frac{i-1}{M}}) if i is odd else sin(t_j/10000^{\\frac{i}{M}})
    """

    def __init__(self, d_input, device):
        super().__init__()
        self.device = device
        self.d_input = d_input

        """
        0 -> 0 sin
        1 -> 0 cos
        2 -> 2 sin
        3 -> 2 cos
        4 -> 4 sin
        5 -> 4 cos
        """
        div_term = (
            torch.tensor([item - 1 if item % 2 else item for item in range(0, d_input)], device=self.device)
            * -(torch.log(torch.tensor(10000.0, device=self.device)) / d_input)
        ).exp()

        self.register_buffer("div_term", div_term)

    def forward(self, interval):
        # interval shape: [batch_size, seq_len]
        scaled_interval = interval.unsqueeze(dim=-1) * self.div_term
        scaled_interval[..., 0::2] = torch.sin(scaled_interval[..., 0::2])
        scaled_interval[..., 1::2] = torch.cos(scaled_interval[..., 1::2])

        return scaled_interval
