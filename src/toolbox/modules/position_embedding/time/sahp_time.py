import torch
import torch.nn as nn


class SAHPTimeEmbedding(nn.Module):
    """Time embedding method posted in the SAHP paper.

    pe^k_{(v_i, t_i)} = cos(\\omega_k * i + w_k * t_i) if i is odd else sin(\\omega_k * i + w_k * t_i)
    """

    def __init__(self, d_input, device):
        super().__init__()
        self.device = device
        self.d_input = d_input

        """
        0 sin
        1 cos
        2 sin
        3 cos
        4 sin
        5 cos
        """
        omega = torch.pi * torch.linspace(0, 1, d_input, device=self.device)
        self.register_buffer("omega", omega)
        self.Wt = nn.Linear(1, d_input, bias=False, device=self.device)
        self.offset = torch.tensor([torch.pi/2 if idx % 2 == 1 else 0 for idx in range(d_input)], device=self.device, requires_grad=False)

    def forward(self, interval, seq_len_dim=-1):
        # interval shape: [batch_size, seq_len]
        seq_len = interval.shape[seq_len_dim]
        part1 = self.omega * torch.arange(seq_len, device=self.device).unsqueeze(dim=-1)
        # [seq_len, d_input]
        part2 = self.Wt(interval.unsqueeze(-1))
        # [batch_size, seq_len, d_input]
        time_pos_emb = part1 + part2

        return torch.sin(time_pos_emb + self.offset)

if __name__ == "__main__":
    time_encoder = SAHPTimeEmbedding(16, 'cpu')
    time = torch.ones(16, 64) * 16
    time_emb = time_encoder(time)
