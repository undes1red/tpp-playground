from mamba_ssm import selective_scan_fn as ssf
import torch
import torch.nn as nn
from einops import rearrange, repeat, reduce, pack, unpack

from src.toolbox.position_embedding import BiasedPositionalEmbedding

# The original mamba does not support custom timestamps which is required by mamba hawkes process.
# So we need a new mamba implementation for this usecase.

class mamba(nn.Module):
    def __init__(self, num_events, min_time_interval, max_time_interval, d_input, d_mamba, kernel_size, n_layers, dropout, expand, device):
        super(mamba, self).__init__()
        self.device = device
        self.d_input = d_input

        # Mark related
        self.event_embedding = nn.Embedding(num_embeddings = num_events + 1, embedding_dim = d_input,\
                                            padding_idx = num_events, device = self.device)
        
        self.mamba_encoder = nn.ModuleList(
            [
                mamba_layer(min_time_interval, max_time_interval, d_input, d_mamba, kernel_size, expand, device = self.device) 
                    for _ in range(n_layers)
            ])
        self.dropout = nn.Dropout(dropout)


    def forward(self, events_history, time_history, mean, std):
        output_state = self.event_embedding(events_history)                    # [batch_size, seq_len, d_input]

        for mamba_layer in self.mamba_encoder:
            output_state = mamba_layer(output_state, time_history)             # [batch_size, seq_len, d_input]
            output_state = self.dropout(output_state)                          # [batch_size, seq_len, d_input]
        
        return output_state


class mamba_layer(nn.Module):
    def __init__(self, min_time_interval, max_time_interval, d_input, d_mamba, kernel_size, expand, device):
        super(mamba_layer, self).__init__()
        self.device = device
        self.d_input = d_input
        self.d_expanded_input = expand * d_input
        # Follow the official Mamba implementation.
        # We will scale the original time interval into [0.001, 0.1]
        self.min_time_interval = min_time_interval
        self.max_time_interval = max_time_interval
        self.scaled_min_time_interval = 0.001
        self.scaled_max_time_interval = 0.1

        # Inner projection, where the dimension expansion happens.
        # Also, extract x and z for SSM processing and residual connection, respectively.
        self.linear_extract_xz = nn.Linear(self.d_input, 2 * self.d_expanded_input, bias = False, device = self.device)
        self.linear_extract_time = nn.Linear(self.d_input, self.d_expanded_input, bias = False, device = self.device)

        # Module on the forwardpropagation route of x.
        self.conv1d = nn.Conv1d(
            in_channels = self.d_expanded_input,
            out_channels = self.d_expanded_input,
            bias = True,
            kernel_size = kernel_size,
            groups = self.d_expanded_input,
            padding = kernel_size - 1,
            device = self.device
        )
        self.act = nn.SiLU()
        
        # matrix A a.k.a. the hidden state shift matrix
        self.A = nn.Parameter(torch.zeros(self.d_expanded_input, d_mamba, device = self.device, requires_grad = True))
        self.A._no_weight_decay = True

        # Linear B and C
        # B applies to the input matrix, and C applies to the output.
        self.project_b_and_c = nn.Linear(self.d_expanded_input, d_mamba * 2 , bias = False, device = self.device)

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_expanded_input, device = device))
        self.D._no_weight_decay = True

        # output layer
        self.out_proj = nn.Linear(self.d_expanded_input, d_input, device = self.device)


    def forward(self, input_state, time_history):
        _, seqlen, _ = input_state.shape

        scaled_time_history = (time_history - self.min_time_interval) / (self.max_time_interval - self.min_time_interval)
                                                                               # [batch_size, seq_len]
        scaled_time_history = scaled_time_history * (self.scaled_max_time_interval - self.scaled_min_time_interval) + self.scaled_min_time_interval
                                                                               # [batch_size, seq_len]
        delta = repeat(scaled_time_history, 'b s -> b s d', d = self.d_expanded_input)
                                                                               # [batch_size, seq_len, d_expanded_input]

        xz = rearrange(self.linear_extract_xz(input_state), 'b s dd -> b dd s')# [batch_size, d_expanded_input * 2, seq_len]
        x, z = xz.chunk(2, dim = -2)                                           # [batch_size, d_expanded_input, seq_len] * 2

        # forwardpropagation route of x
        # conv1d -> activation(silu) -> SSM -> output
        # x a.k.a. u in the code.
        x = self.act(self.conv1d(x)[..., :seqlen])                             # [batch_size, d_expanded_input, seq_len]

        # different from normal modules, ssf always assumes the seq_len is the last dimension.
        # u:     [batch_size, d_input, seq_len]
        # delta: [batch_size, d_input, seq_len]
        # A:     [d_input, d_mamba]
        # B:     [batch_size, d_mamba, seq_len]
        # C:     [batch_size, d_mamba, seq_len]
        # Delta
        delta = rearrange(delta, 'b s d -> b d s')                             # [batch_size, d_expanded_input, seq_len]
        # B and C
        B_a_C = self.project_b_and_c(rearrange(x, 'b dsi d -> b d dsi'))       # [batch_size, seq_len, d_mamba * 2]
        B, C = rearrange(B_a_C, 'b s d -> b d s').chunk(2, dim = -2)           # [batch_size, d_mamba, seq_len]

        output_state = ssf(u = x, delta = delta, A = self.A, B = B, C = C, D = self.D.float(), z = z)
                                                                               # [batch_size, d_expanded_input, seq_len]
        output_state = rearrange(output_state, 'b d s -> b s d')               # [batch_size, seq_len, d_expanded_input]

        output_state = self.out_proj(output_state)                             # [batch_size, seq_len, d_expanded_input]

        return output_state