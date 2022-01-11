import torch
import torch.nn as nn
import torch.nn.functional as F


class CTLSTMCell(nn.Module):

    def __init__(self, hidden_dim, device):
        super(CTLSTMCell, self).__init__()

        self.device = device
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(hidden_dim * 2, hidden_dim * 7, bias=True).to(device)


    def forward(
        self, rnn_input, hidden_t_i_minus, cell_t_i_minus, cell_bar_im1):
        # rnn_input: [batch, embedding]
        # hidden_t_i_minus: [batch, embedding]
        # cell_t_i_minus: [batch, embedding]
        # cell_bar_im1: [batch, embedding]

        dim_of_hidden = rnn_input.dim() - 1

        input_i = torch.cat((rnn_input, hidden_t_i_minus), dim=dim_of_hidden)  # [batch_size, hidden_dim * 2]
        output_i = self.linear(input_i)                                        # [batch_size, hidden_dim * 7]

        gate_input, \
        gate_forget, gate_output, gate_pre_c, \
        gate_input_bar, gate_forget_bar, gate_decay = output_i.chunk(
            7, dim_of_hidden)                                                  # 7 * [batch_size, hidden_dim]

        gate_input = torch.sigmoid(gate_input)                                 # [batch_size, hidden_dim], i
        gate_forget = torch.sigmoid(gate_forget)                               # [batch_size, hidden_dim], f
        gate_output = torch.sigmoid(gate_output)                               # [batch_size, hidden_dim], o
        gate_pre_c = torch.tanh(gate_pre_c)                                    # [batch_size, hidden_dim], z
        gate_input_bar = torch.sigmoid(gate_input_bar)                         # [batch_size, hidden_dim], \bar{i}
        gate_forget_bar = torch.sigmoid(gate_forget_bar)                       # [batch_size, hidden_dim], \bar{f}
        gate_decay = F.softplus(gate_decay, 1.0) # decay scale is always 1.0   # [batch_size, hidden_dim], \delta

        cell_i = gate_forget * cell_t_i_minus + gate_input * gate_pre_c
        cell_bar_i = gate_forget_bar * cell_bar_im1 + gate_input_bar * gate_pre_c

        return cell_i, cell_bar_i, gate_decay, gate_output

    def decay(self, cell_i, cell_bar_i, gate_decay, gate_output, dtime):
        # no need to consider extra_dim_particle here
        # cuz this function is applicable to any # of dims
        # cell_i : [batch_size, dim_size]

        if dtime.dim() < cell_i.dim():
            # e.g. 
            # cell_i : B x T x D 
            # dtime : B x T 
            dtime = dtime.unsqueeze(cell_i.dim()-1).expand_as(cell_i)

        cell_t_ip1_minus = cell_bar_i + (cell_i - cell_bar_i) * torch.exp(
            -gate_decay * dtime)                                               # [batch_size, hidden_dim], c(t) for intesity calculation
        hidden_t_ip1_minus = gate_output * torch.tanh(cell_t_ip1_minus)

        return cell_t_ip1_minus, hidden_t_ip1_minus