import torch
import torch.nn as nn
import torch.nn.functional as F
from .nonneg import NonNegLinear, ClampLinear
from .activate import Log


class DynamicMLP(nn.Module):
    '''
    This class implements a dynamic MLP which weight value would change depending on the history data. 
    The purpose is try to force the model output at point 0 is forever 0.
    '''

    def __init__(self, d_history, d_intensity, dropout, num_layers, mlp_layers):
        super(DynamicMLP, self).__init__()

        self.history = nn.LSTM(input_size=1, hidden_size=d_history,
                               num_layers=num_layers, batch_first=True, dropout=dropout)

        # self.weight_gen = nn.Linear(1, d_intensity, bias=True)
        self.weight_gen = ClampLinear(1, d_intensity, clamp_min=-0.10, bias=True)
        # self.weight_gen = NonNegLinear(1, d_intensity,bias=True)
        # Should we use the output of LSTM as the weight of the dynamic linear layer?
        # self.time_weight = nn.Linear(1, d_history, bias=True)
        self.time_weight = ClampLinear(1, d_history, clamp_min=-0.10, bias=True)
        # self.time_weight = NonNegLinear(1, d_history,bias=True)

        self.time = NonNegLinear(1, d_history * num_layers, bias=False)

        self.mlp = nn.ModuleList([
            NonNegLinear(d_intensity, d_intensity, bias=False) for _ in range(mlp_layers)
        ])
        self.accu = NonNegLinear(d_intensity, 1, bias=False)

        # Activate functions
        self.activate = nn.Softplus()
        self.activate_factor = nn.Parameter(torch.tensor([0.]))
        # Can tanh or sigmoid hold the trend of increasing intensity better?
        # Or we should let our model do this by itself.
        self.activate_time = Log()
        # self.activate_time = nn.Tanh()
        self.activate_time_factor = nn.Parameter(torch.tensor([0.]))

    def forward(self, time_history, time_happen):
        '''
        So the timeline should be divided into several parts.
        First, a fixed number of previous time points are choosed and feeded into LSTM, then the histiory embedding is used to generate
        the weight using an additional Linear layer.
        Interesting idea but I don't know if it works.

        Input properties:
        time_history: [batch_size, max_seq_size]. Padding is dataloaders' task.
        time_happen: [batch_size, 1]. Time that happens adter corresponding history data. Should be the subtraction results between 
        the latest history and the happening time.
        '''
        # generate time
        time = self.time(time_happen)

        # Let the weight change with the input time.
        # We discover that the intensity may be affected and start increasing when the relative time is too big.
        # Try to add a concave activation here, like log
        time_weight = self.time_weight(self.activate_time(
            F.softplus(self.activate_time_factor) * time_happen))

        # weight generation
        time_history = time_history.unsqueeze(-1)
        _, (hidden, _) = self.history(time_history)
        hidden = hidden + time_weight.unsqueeze(0)
        hidden = hidden.transpose(0, 1).reshape(time_history.shape[0], -1, 1)
        weight = self.weight_gen(hidden)
        weight = self.activate(weight)

        # Mingle history and relative time embedding.
        # time: [batch_size, 1, history_dim]
        # weight: [batch_size, history_dim, intensity_dim]
        # Result: [batch_size, history_dim]
        time = time.unsqueeze(1)
        output = torch.bmm(time, weight).squeeze()

        for layer in self.mlp:
            output = layer(output)
            # Imitate a weaker ReLU activation
            output = F.softplus(self.activate_factor) * output

        output = self.accu(output)
        return output
    
    def show_time_scale_factor(self):
        return F.softplus(self.activate_time_factor)

    def show_activate_factor(self):
        return F.softplus(self.activate_factor)