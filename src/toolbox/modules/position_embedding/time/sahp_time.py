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
        1 -> 0 cos
        2 -> 2 sin
        3 -> 2 cos
        4 -> 4 sin
        5 -> 4 cos
        """
        omega = torch.pi * torch.linspace(0, 1, d_input, device=self.device)
        self.register_buffer("omega", omega)
        self.Wt = nn.Linear(1, d_input, bias=False, device=self.device)

    def forward(self, interval, seq_len_dim=-1):
        # interval shape: [batch_size, seq_len]
        seq_len = interval.shape[seq_len_dim]
        part1 = self.omega * torch.arange(seq_len, device=self.device).unsqueeze(dim=-1)
        # [seq_len, d_input]
        part2 = self.Wt(interval.unsqueeze(-1))
        # [batch_size, seq_len, d_input]
        time_pos_emb = part1 + part2

        time_pos_emb[..., 0::2] = torch.sin(time_pos_emb[..., 0::2])
        time_pos_emb[..., 1::2] = torch.cos(time_pos_emb[..., 1::2])

        return time_pos_emb
