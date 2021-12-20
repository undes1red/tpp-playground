import torch
import torch.nn as nn
import torch.nn.functional as F
from .nonneg import NonNegLinear, Polynomial

class DynamicMLP(nn.Module):
    '''
    This class implements a dynamic MLP which weight value would change depending on the history data. 
    The purpose is try to force the model output at point 0 is forever 0.
    '''

    def __init__(self, d_history, d_intensity, dropout, num_layers, mlp_layers, device, time_activation, no_time_weight, no_scale,
                 weight_gen_min, time_weight_min):
        super(DynamicMLP, self).__init__()
        self.device = device
        self.no_time_weight = no_time_weight
        self.d_history = d_history

        self.history = nn.LSTM(input_size=1, hidden_size=d_history,
                               num_layers=num_layers, batch_first=True, dropout=dropout).to(self.device)

        self.time_weight = NonNegLinear(1, d_intensity, bias = True).to(self.device)
        self.time_encoder = Polynomial(dimension = d_history, polynomial_start = -2, polynomial_end = 3, device = device)

        self.time = NonNegLinear(1, d_history * num_layers, bias=False).to(self.device)

        self.mlp = nn.ModuleList([
            NonNegLinear(d_intensity, d_intensity, bias=False) for _ in range(mlp_layers)
        ]).to(self.device)
        self.accu = NonNegLinear(d_intensity, 1, bias=False).to(self.device)

        self.activate = nn.Softplus()

        if no_scale:
            self.activate_factor = torch.tensor(torch.zeros(mlp_layers, 1, device = self.device))
            self.activate_time_factor = torch.tensor([0.], device = self.device)
        else:
            self.activate_factor = nn.Parameter(torch.zeros(mlp_layers, 1, device = self.device))
            self.activate_time_factor = nn.Parameter(torch.tensor([0.], device = self.device))

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
        # History embedding
        _, (history, _) = self.history(time_history.unsqueeze(-1))
        time_vector = self.time_encoder(time_happen)
        history_time_vector = history.view(-1, self.d_history, 1) + time_vector
        weight = self.time_weight(history_time_vector)
        weight = self.activate(weight)

        # Relative time embedding
        time = self.time(time_happen)
        time = time.unsqueeze(1)

        # Mingle history and relative time embedding.
        # time: [batch_size, 1, history_dim]
        # weight: [batch_size, history_dim, intensity_dim]
        # Result: [batch_size, intensity_dim]
        output = torch.bmm(time, weight).squeeze()

        for idx, layer in enumerate(self.mlp):
            output = layer(output)
            # Imitate a weaker ReLU activation
            output = output * F.softplus(self.activate_factor[idx])

        output = self.accu(output)
        return output
    
    def show_time_scale_factor(self):
        return F.softplus(self.activate_time_factor)

    def show_activate_factor(self):
        return F.softplus(self.activate_factor)