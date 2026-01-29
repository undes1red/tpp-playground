import torch
import torch.nn as nn


class AttNHPTimeEmbedding(nn.Module):
    """Time embedding method posted in the AttNHP paper.

    [t]_d = cos(t/(m*(\\frac{5M}{m})^(\\frac{d}{D})) if i is odd else sin(t/(m*(\\frac{5M}{m})^(\\frac{d}{D}))
    """

    def __init__(self, d_input, device):
        super().__init__()
        self.device = device
        self.d_input = d_input

    def get_div_term(self, m, M):
        # input shape: m: [batch_size], M: [batch_size]
        div_term = -(
            torch.log(m).unsqueeze(dim=-1)
            + (torch.log(5 * M) - torch.log(m)).unsqueeze(dim=-1) * \
            (torch.tensor([item - 1 if item % 2 else item for item in range(0, self.d_input)], device=self.device) / self.d_input)
        )

        return div_term.exp()

    def forward(self, interval):
        # interval shape: [batch_size, seq_len]
        m = interval.min(dim=-1).values
        M = interval.cumsum(dim=-1)[:, -1]
        scaled_interval = interval.unsqueeze(dim=-1) * self.get_div_term(m, M)

        scaled_interval[..., 0::2] = torch.sin(scaled_interval[..., 0::2])
        scaled_interval[..., 1::2] = torch.cos(scaled_interval[..., 1::2])

        return scaled_interval
