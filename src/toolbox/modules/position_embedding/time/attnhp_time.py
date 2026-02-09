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

    def forward(self, interval, resolution_dim=False):
        event_time = interval[..., -1] if resolution_dim else interval  # [..., batch_size, seq_len]

        # interval shape: [..., batch_size, seq_len]
        m = torch.where(event_time == 0, torch.inf, event_time).min(dim=-1).values  # [..., batch_size]
        M = event_time.cumsum(dim=-1)[..., -1]  # [..., batch_size]
        scaled_interval = interval.unsqueeze(dim=-1) * \
            rearrange(self.get_div_term(m, M), f'... b d -> ... b () {"()" if resolution_dim else ""} d')
        # [..., d_input]

        return torch.sin(scaled_interval + self.offset)


if __name__ == "__main__":
    time_encoder = AttNHPTimeEmbedding(16, "cpu")
    time = torch.ones(32, 64) * 1
    time_emb = time_encoder(time)
    print(time_emb.shape)

    time_encoder = AttNHPTimeEmbedding(16, "cpu")
    time = torch.ones(32, 64, 100) * 1
    time_emb_1 = time_encoder(time, resolution_dim=True)
    print(time_emb.shape)

    time_encoder = AttNHPTimeEmbedding(16, "cpu")
    time = torch.ones(2, 32, 64, 100) * 1
    time_emb_2 = time_encoder(time, resolution_dim=True)
    print(time_emb.shape)
