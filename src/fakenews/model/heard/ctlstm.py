import torch.nn as nn
import torch
import torch.nn.functional as F


class CTLSTMCell(nn.Module):
    def __init__(self, hidden_dim, batch_size,dp_rate,device):
        super(CTLSTMCell, self).__init__()

        self.device = device

        self.batch_size = batch_size

        self.hidden_dim = hidden_dim

        self.dp_rate = dp_rate

        self.linear = nn.Linear(hidden_dim * 2, hidden_dim * 7, bias=True).to(self.device)
    
    def init_states(self):

        self.h_d = torch.zeros(self.batch_size, self.hidden_dim, dtype=torch.float).to(self.device)
        self.c_d = torch.zeros(self.batch_size, self.hidden_dim, dtype=torch.float).to(self.device)
        self.c_bar = torch.zeros(self.batch_size, self.hidden_dim, dtype=torch.float).to(self.device)
        self.c = torch.zeros(self.batch_size, self.hidden_dim, dtype=torch.float).to(self.device)

    def forward(
            self, rnn_input,
            hidden_t_i_minus, cell_t_i_minus, cell_bar_im1):

        dim_of_hidden = rnn_input.dim() - 1

        input_i = torch.cat((rnn_input, hidden_t_i_minus), dim=dim_of_hidden)

        output_i = self.linear(input_i)

        gate_input, \
        gate_forget, gate_output, gate_pre_c, \
        gate_input_bar, gate_forget_bar, gate_decay = output_i.chunk(
            7, dim_of_hidden)

        gate_input = torch.sigmoid(gate_input)

        gate_forget = torch.sigmoid(gate_forget)

        gate_output = torch.sigmoid(gate_output)

        gate_pre_c = torch.tanh(gate_pre_c)

        gate_input_bar = torch.sigmoid(gate_input_bar)

        gate_forget_bar = torch.sigmoid(gate_forget_bar)

        gate_decay = F.softplus(gate_decay)
      

        cell_i = gate_forget * cell_t_i_minus + gate_input * gate_pre_c
        cell_bar_i = gate_forget_bar * cell_bar_im1 + gate_input_bar * gate_pre_c

        return cell_i, cell_bar_i, gate_decay, gate_output

    def decay(self, cell_i, cell_bar_i, gate_decay, gate_output, dtime,if_predict=False):
        
        if not if_predict:
            if dtime.dim() < cell_i.dim():
                dtime = dtime.unsqueeze(cell_i.dim()-1).expand_as(cell_i)
        else:
            dtime = dtime.unsqueeze(0).unsqueeze(0)
            cell_bar_i = cell_bar_i.unsqueeze(-1)
            cell_i = cell_i.unsqueeze(-1)
            gate_decay = gate_decay.unsqueeze(-1)
            gate_output = gate_output.unsqueeze(-1)

        #Eq(7)
        cell_t_ip1_minus = cell_bar_i + (cell_i - cell_bar_i) * torch.exp(
            -gate_decay * dtime)
        
        #Eq(4b)
        hidden_t_ip1_minus = gate_output * torch.tanh(cell_t_ip1_minus)
        
        return cell_t_ip1_minus, hidden_t_ip1_minus