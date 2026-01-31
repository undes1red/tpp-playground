import torch
import torch.nn as nn
from einops import rearrange


class AttNHPTimeEmbedding(nn.Module):
    """Time embedding method posted in the AttNHP paper.

    [t]_d = cos(t/(m*(\\frac{5M}{m})^(\\frac{d}{D})) if i is odd else sin(t/(m*(\\frac{5M}{m})^(\\frac{d}{D}))
    """

    def __init__(self, d_input, device):
        super().__init__()
        self.device = device
        self.d_input = d_input
        self.offset = torch.tensor(
            [torch.pi / 2 if idx % 2 == 1 else 0 for idx in range(d_input)], device=self.device, requires_grad=False
        )

    def get_div_term(self, m, M):
        # input shape: m: [batch_size], M: [batch_size]
        div_term = -(
            torch.log(m).unsqueeze(dim=-1)
            + (torch.log(5 * M) - torch.log(m)).unsqueeze(dim=-1)
            * (
                torch.tensor([item - 1 if item % 2 else item for item in range(0, self.d_input)], device=self.device)
                / self.d_input
            )
        )

        return div_term.exp()

    def forward(self, interval, dim_before_batch_size=0):
        if len(interval) - 2 - dim_before_batch_size > 1:
            raise ValueError('Unexpected dim found in interval.')

        # interval shape: [..., batch_size, seq_len, (integation_sample_rate)]
        m = interval.min(dim=-1).values  # [..., batch_size]
        M = interval.cumsum(dim=-1)[:, -1]  # [..., batch_size]
        scaled_interval = interval.unsqueeze(dim=-1) * rearrange(
            self.get_div_term(m, M), f"... b d -> ... b () {'()' if (len(interval) - 2 - dim_before_batch_size) == 1 else ''} d"
        )  # [..., batch_size, seq_len, (integation_sample_rate), d_input]

        return torch.sin(scaled_interval + self.offset)


if __name__ == "__main__":
    time_encoder = AttNHPTimeEmbedding(16, "cpu")
    time = torch.ones(16, 64) * 1
    time_emb = time_encoder(time)
