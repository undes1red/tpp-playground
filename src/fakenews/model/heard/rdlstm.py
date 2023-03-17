import torch.nn as nn
import torch
import torch.nn.functional as F


class RDLSTMCell(nn.Module):
    def __init__(self, hidden_dim, batch_size,dp_rate,device):
        super(RDLSTMCell, self).__init__()

        self.device = device

        self.batch_size = batch_size

        self.hidden_dim = hidden_dim

        self.dp_rate = dp_rate

        self.linear = nn.Linear(hidden_dim * 2, hidden_dim * 4, bias=True).to(self.device)
        nn.init.orthogonal_(self.linear.weight)

        self.dropout = nn.Dropout(self.dp_rate)
    
    def init_states(self):
       
        self.h_d = torch.zeros(self.batch_size, self.hidden_dim, dtype=torch.float).to(self.device)
        self.c_d = torch.zeros(self.batch_size, self.hidden_dim, dtype=torch.float).to(self.device)
        
    def forward(
            self, rnn_input,
            hidden_i_minus, cell_i_minus,if_dp=True):

        dim_of_hidden = rnn_input.dim() - 1

        input_i = torch.cat((rnn_input, hidden_i_minus), dim=dim_of_hidden)

        output_i = self.linear(input_i)

        gate_input, \
        gate_forget, gate_output, gate_pre_c, = output_i.chunk(
            4, dim_of_hidden)

        gate_input = torch.sigmoid(gate_input)

        gate_forget = torch.sigmoid(gate_forget)

        gate_output = torch.sigmoid(gate_output)

        gate_pre_c = torch.tanh(gate_pre_c)-1

        if if_dp:
            cell_i = self.dropout(gate_forget) * cell_i_minus + self.dropout(gate_input) * gate_pre_c
            h_i = self.dropout(gate_output)*torch.tanh(cell_i)
        else:
            cell_i = gate_forget * cell_i_minus + gate_input * gate_pre_c
            h_i = gate_output*torch.tanh(cell_i)

        return h_i, (h_i,cell_i)