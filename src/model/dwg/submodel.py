import torch
import torch.nn as nn
import torch.nn.functional as F
from .nonneg import NonNegLinear, ClampLinear
from .activate import Log

TA = {
    'log': Log(),
    'tanh': nn.Tanh()
}
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
        self.d_intensity = d_intensity

        self.history = nn.LSTM(input_size=1, hidden_size=d_history,
                               num_layers=num_layers, batch_first=True, dropout=dropout).to(self.device)

        # self.weight_gen = nn.Linear(1, d_intensity, bias=True)
        self.weight_gen = ClampLinear(1, d_intensity, clamp_min=weight_gen_min, bias=True).to(self.device)
        # self.weight_gen = NonNegLinear(1, d_intensity,bias=True)
        # Should we use the output of LSTM as the weight of the dynamic linear layer?
        # self.time_weight = nn.Linear(1, d_history, bias=True)
        self.time_weight = ClampLinear(1, d_history, clamp_min=time_weight_min, bias=True).to(self.device)
        # self.time_weight = NonNegLinear(1, d_history,bias=True)

        # Non-negative time encoder part 2.
        self.time_outside = NonNegLinear(1, d_history, bias=False).to(self.device)

        self.mlp = nn.ModuleList([
            NonNegLinear(d_intensity, d_intensity, bias=False) for _ in range(mlp_layers)
        ]).to(self.device)
        self.accu = NonNegLinear(d_intensity, 1, bias=False).to(self.device)

        # Activate functions
        self.activate = nn.Softplus()
        # Can tanh or sigmoid hold the trend of increasing intensity better?
        # Or we should let our model do this by itself.
        self.activate_time = TA[time_activation]
        # self.activate_time = nn.Tanh()
        if no_scale:
            self.activate_time_factor = torch.tensor([0.], device = self.device)
            self.activate_factor = torch.zeros(mlp_layers, device = self.device)
        else:
            self.activate_time_factor = nn.Parameter(torch.tensor([0.], device = self.device))
            self.activate_factor = nn.Parameter(torch.zeros(mlp_layers, device = self.device))

    def forward(self, time_history, time_next):
        '''
        So the timeline should be divided into several parts.
        First, a fixed number of previous time points are choosed and feeded into LSTM, then the histiory embedding is used to generate
        the weight using an additional Linear layer.
        Interesting idea but I don't know if it works.

        Args:
            time_history: [batch_size, seq_len, 1]
            time_next:    [batch_size, seq_len, 1]
        '''
        # generate time
        time_outside = self.time_outside(time_next)                            # [batch_size, seq_len, d_history]

        # Let the weight change with the input time.
        # We discover that the intensity may be affected and start increasing when the relative time is too big.
        # Try to add a concave activation here, like log
        time_weight = self.time_weight(self.activate_time(
            F.softplus(self.activate_time_factor) * time_next))                # [batch_size, seq_len, d_history]

        # weight generation
        output, (_, _) = self.history(time_history)                            # [batch_size, seq_len, d_history]
        hidden = output + time_weight \
                    if not self.no_time_weight else torch.zeros_like(output)   # [batch_size, seq_len, d_history]
        hidden = hidden.unsqueeze(-1)                                          # [batch_size, seq_len, d_history, 1]  
        time_weight = self.weight_gen(hidden).transpose(-1, -2)                # [batch_size, seq_len, d_intensity, d_history]
        time_weight = self.activate(time_weight)                               # [batch_size, seq_len, d_intensity, d_history]

        # Mingle history and relative time embedding.
        time_outside = time_outside.unsqueeze(-1)                              # [batch_size, seq_len, d_history, 1]
        output = torch.matmul(time_weight, time_outside).squeeze()             # [batch_size, seq_len, d_intensity]

        for layer_idx, layer in enumerate(self.mlp):
            output = layer(output)                                             # [batch_size, seq_len, d_intensity]
            # Imitate a weaker ReLU activation
            output = F.softplus(self.activate_factor[layer_idx]) * output

        output = self.accu(output)                                             # [batch_size, seq_len, 1]
        return output
    
    def show_time_scale_factor(self):
        return F.softplus(self.activate_time_factor)

    def show_activate_factor(self):
        return F.softplus(self.activate_factor)

    def intensity(self, time_history, time_next, resolution):
        '''
        Model intensity prober. Perhaps, we can support intensity integral as well.
        Args:
        time_history: [batch_size, seq_len, 1]
        time_next:    [batch_size, seq_len, 1]
        resolution:   int
        '''
        history_output, (_, _) = self.history(time_history)                    # [batch_size, seq_len, d_history]
        batch_size, seq_len, d_history = history_output.shape

        history_expand = history_output.repeat(1, 1, resolution).reshape(batch_size, -1, d_history)
                                                                               # [batch_size, seq_len * resolution, d_history]
        time_multiplier = torch.linspace(0, 1, resolution)                     # [resolution]
        time_expand = (time_multiplier * time_next).reshape(batch_size, -1, 1) # [batch_size, seq_len * resolution, 1]
        time_expand.requires_grad = True
        time_outside = self.time_outside(time_expand)                          # [batch_size, seq_len * resolution, d_history]
        time_weight = self.time_weight(self.activate_time(
            F.softplus(self.activate_time_factor) * time_expand))              # [batch_size, seq_len * resolution, d_history]
        hidden = history_expand + time_weight \
                    if not self.no_time_weight else torch.zeros_like(history_output)
                                                                               # [batch_size, seq_len * resolution, d_history]
        hidden = hidden.unsqueeze(-1)                                          # [batch_size, seq_len * resolution, d_history, 1]  
        time_weight = self.weight_gen(hidden).transpose(-1, -2)                # [batch_size, seq_len * resolution, d_intensity, d_history]
        time_weight = self.activate(time_weight)                               # [batch_size, seq_len * resolution, d_intensity, d_history]

        # Mingle history and relative time embedding.
        time_outside = time_outside.unsqueeze(-1)                              # [batch_size, seq_len * resolution, d_history, 1]
        integral = torch.matmul(time_weight, time_outside)                     # [batch_size, seq_len * resolution, d_intensity, 1]
        integral = integral.reshape(batch_size, seq_len * resolution, self.d_intensity)
                                                                               # [batch_size, seq_len * resolution, d_intensity]

        for layer_idx, layer in enumerate(self.mlp):
            integral = layer(integral)                                         # [batch_size, seq_len * resolution, d_intensity]
            # Imitate a weaker ReLU activation
            integral = F.softplus(self.activate_factor[layer_idx]) * integral

        integral = self.accu(integral)                                         # [batch_size, seq_len * resolution, 1]

        intensity = torch.autograd.grad(
            outputs = integral,
            inputs = time_expand,
            grad_outputs = torch.ones_like(integral),
            create_graph = True
        )[0]                                                                   # [batch_size, seq_len * resolution, 1]
        time_expand.requires_grad = False

        timestamp = time_expand.squeeze().reshape(batch_size, seq_len, resolution)
                                                                               # [batch_size, seq_len, resolution]
        timestamp = torch.cat(
            (torch.zeros((batch_size, seq_len, 1), device = self.device), timestamp.diff(dim = -1)),
            dim = -1)                                                          # [batch_size, seq_len, resolution]
        timestamp = timestamp.reshape(batch_size, seq_len * resolution)        # [batch_size, seq_len * resolution]

        return integral, intensity, timestamp

    def model_probe_function(self, time_history, time_next, resolution):
        '''
        Model intensity prober. Perhaps, we can support intensity integral as well.
        Args:
        time_history: [batch_size, seq_len, 1]
        time_next:    [batch_size, seq_len, 1]
        resolution:   int
        '''
        # Part 1: forward propagation.
        history_output, (_, _) = self.history(time_history)                    # [batch_size, seq_len, d_history]
        batch_size, seq_len, d_history = history_output.shape

        history_expand = history_output.repeat(1, 1, resolution).reshape(batch_size, -1, d_history)
                                                                               # [batch_size, seq_len * resolution, d_history]
        time_multiplier = torch.linspace(0, 1, resolution)                     # [resolution]
        time_expand = (time_multiplier * time_next).reshape(batch_size, -1, 1) # [batch_size, seq_len * resolution, 1]
        time_expand.requires_grad = True
        time_outside = self.time_outside(time_expand)                          # [batch_size, seq_len * resolution, d_history]
        time_weight = self.time_weight(self.activate_time(
            F.softplus(self.activate_time_factor) * time_expand))              # [batch_size, seq_len * resolution, d_history]
        hidden = history_expand + time_weight \
                    if not self.no_time_weight else torch.zeros_like(history_output)
                                                                               # [batch_size, seq_len * resolution, d_history]
        hidden = hidden.unsqueeze(-1)                                          # [batch_size, seq_len * resolution, d_history, 1]  
        time_weight = self.weight_gen(hidden).transpose(-1, -2)                # [batch_size, seq_len * resolution, d_intensity, d_history]
        time_weight = self.activate(time_weight)                               # [batch_size, seq_len * resolution, d_intensity, d_history]

        # Mingle history and relative time embedding.
        time_outside = time_outside.unsqueeze(-1)                              # [batch_size, seq_len * resolution, d_history, 1]
        integral = torch.matmul(time_weight, time_outside)                     # [batch_size, seq_len * resolution, d_intensity, 1]
        integral = integral.reshape(batch_size, seq_len * resolution, self.d_intensity)
                                                                               # [batch_size, seq_len * resolution, d_intensity]
        output_after_dwg_layer = integral                                      # [batch_size, seq_len * resolution, d_intensity]

        mlp_output = []
        for layer_idx, layer in enumerate(self.mlp):
            integral = layer(integral)                                         # [batch_size, seq_len * resolution, d_intensity]
            # Imitate a weaker ReLU activation
            integral = F.softplus(self.activate_factor[layer_idx]) * integral
            mlp_output.append(integral)                                        # [batch_size, seq_len * resolution, d_intensity] * layer

        integral = self.accu(integral)                                         # [batch_size, seq_len * resolution, 1]

        intensity = torch.autograd.grad(
            outputs = integral,
            inputs = time_expand,
            grad_outputs = torch.ones_like(integral),
            create_graph = True
        )[0]                                                                   # [batch_size, seq_len * resolution, 1]

        timestamp = time_expand.squeeze().reshape(batch_size, seq_len, resolution)
                                                                               # [batch_size, seq_len, resolution]
        timestamp = torch.cat(
            (torch.zeros((batch_size, seq_len, 1), device = self.device), timestamp.diff(dim = -1)),
            dim = -1)                                                          # [batch_size, seq_len, resolution]
        timestamp = timestamp.reshape(batch_size, seq_len * resolution)        # [batch_size, seq_len * resolution]

        # Part 2: Model detection part
        # MLP gradient
        mlp_gradient = {}
        for idx, item in enumerate(mlp_output):
            mlp_gradient[f'mlp_{idx}_grad'] = torch.autograd.grad(
                outputs = item,
                inputs = time_expand,
                grad_outputs = torch.ones_like(item),
                create_graph = True
            )[0]                                                               # [batch_size, seq_len * resolution, 1]
        
        dwg_gradient = torch.autograd.grad(
            outputs = output_after_dwg_layer,
            inputs = time_expand,
            grad_outputs = torch.ones_like(output_after_dwg_layer),
            create_graph = True
        )[0]                                                               # [batch_size, seq_len * resolution, 1]

        time_expand.requires_grad = False
        result = {
            'final_output': integral,
            'accumulated_gradient': intensity,
            **mlp_gradient,
            'dwg_gradient': dwg_gradient,
            **{"output_mlp_" + str(idx): torch.mean(item, dim = -1) for idx, item in enumerate(mlp_output)},\
            **{"output_mlp_max_" + str(idx): torch.max(item, dim = -1)[0] for idx, item in enumerate(mlp_output)},\
            **{"output_mlp_min_" + str(idx): torch.min(item, dim = -1)[0] for idx, item in enumerate(mlp_output)},\
        }

        return result, timestamp