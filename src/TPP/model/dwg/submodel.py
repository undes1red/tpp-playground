import torch
import torch.nn as nn
import torch.nn.functional as F
from .nonneg import NonNegLinear, ClampLinear
from .activate import Log

TA = {
    'log': Log,
    'tanh': nn.Tanh
}
class DynamicMLP(nn.Module):
    '''
    This class implements a dynamic MLP which weight value would change depending on the history data. 
    The purpose is try to force the model output at point 0 is forever 0.
    '''

    def __init__(self, d_history, d_intensity, dropout, num_layers, mlp_layers, device, time_activation, no_time_weight, no_scale,
                 weight_gen_min, time_weight_min, num_events, event_toggle):
        super(DynamicMLP, self).__init__()
        self.device = device
        # include the dummy event added as the first event of all events sequences.
        self.event_toggle = event_toggle
        self.num_events = num_events
        self.no_time_weight = no_time_weight
        self.d_intensity = d_intensity

        if self.event_toggle:
            self.events = nn.Embedding(self.num_events + 1, d_history, padding_idx = self.num_events, device = self.device)
            self.history = nn.LSTM(input_size = d_history + 1, hidden_size = d_history,
                               num_layers = num_layers, batch_first = True, dropout = dropout, device = self.device)
            # Non-negative time encoder part 2.
            self.time_outside = NonNegLinear(self.num_events, d_history, bias = False, device = self.device, embedding_like = True)
            self.time_weight = ClampLinear(self.num_events, d_history, clamp_min = time_weight_min, bias = True, device = self.device, embedding_like = True)
        else:
            self.events = None
            self.history = nn.LSTM(input_size = 1, hidden_size = d_history,
                               num_layers = num_layers, batch_first = True, dropout = dropout, device = self.device)
            # Non-negative time encoder part 2.
            self.time_outside = NonNegLinear(1, d_history, bias = False, device = self.device)
            self.time_weight = ClampLinear(1, d_history, clamp_min = time_weight_min, bias = True, device = self.device)

        # self.weight_gen = nn.Linear(1, d_intensity, bias=True)
        self.weight_gen = ClampLinear(1, d_intensity, clamp_min = weight_gen_min, bias = True, device = self.device)
        # self.weight_gen = NonNegLinear(1, d_intensity,bias=True)
        # Should we use the output of LSTM as the weight of the dynamic linear layer?
        # self.time_weight = nn.Linear(1, d_history, bias=True)
        # self.time_weight = NonNegLinear(1, d_history,bias=True)

        self.mlp = nn.ModuleList([
            NonNegLinear(d_intensity, d_intensity, bias = False, device = self.device) for _ in range(mlp_layers)
        ])
        self.accu = NonNegLinear(d_intensity, 1, bias = False, device = self.device)

        # Activate functions
        self.activate = nn.Softplus()
        self.event_decider = nn.Softmax(dim = -1)
        # Can tanh or sigmoid hold the trend of increasing intensity better?
        # Or we should let our model do this by itself.
        self.activate_time = TA[time_activation]()
        # self.activate_time = nn.Tanh()
        if no_scale:
            self.activate_time_factor = torch.tensor([0.], device = self.device)
            self.activate_factor = torch.zeros(mlp_layers, device = self.device)
        else:
            self.activate_time_factor = nn.Parameter(torch.tensor([0.], device = self.device))
            self.activate_factor = nn.Parameter(torch.zeros(mlp_layers, device = self.device))

    def forward(self, event_history, time_history, time_next, mean, var):
        '''
        So the timeline should be divided into several parts.
        First, a fixed number of previous time points are choosed and feeded into LSTM, then the histiory embedding is used to generate
        the weight using an additional Linear layer.
        Interesting idea but I don't know if it works.

        Update: 2022-07-23
        if self.event_toggle = False, then event information would be discarded. One should use this option when they wants to deal with
        pure TPP problems.

        Args:
            event_history: [batch_size, seq_len]
            time_history:  [batch_size, seq_len, 1]
            time_next:     [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len, 1]
        '''
        time_history = time_history / var
        time_next = time_next / var

        # generate time
        time_outside = self.time_outside(time_next)                            # [batch_size, seq_len, num_events, d_history] if we need events else [batch_size, seq_len, d_history]

        # Let the weight change with the input time.
        # We discover that the intensity may be affected and start increasing when the relative time is too big.
        # Try to add a concave activation here, like log
        time_weight = self.time_weight(self.activate_time(
            F.softplus(self.activate_time_factor) * time_next))                # [batch_size, seq_len, num_events, d_history] if we need events else [batch_size, seq_len, d_history]
        
        if self.event_toggle:
            event_history_embedding = self.events(event_history)               # [batch_size, seq_len, d_history]
            history = torch.cat([
                event_history_embedding, time_history
            ], dim = -1)
        else:
            history = time_history                                             # [batch_size, seq_len, d_history + 1] if we need events else [batch_size, seq_len, 1]
        
        
        # weight generation
        history_output, (_, _) = self.history(history)                         # [batch_size, seq_len, d_history]
        if self.event_toggle:
            history_output = history_output.unsqueeze(-2).repeat(1, 1, self.num_events, 1)
                                                                               # [batch_size, seq_len, num_events, d_history] if we need events else [batch_size, seq_len, d_history]

        hidden = history_output + time_weight \
                    if not self.no_time_weight else torch.zeros_like(history_output)
                                                                               # [batch_size, seq_len, num_events, d_history] if we need events else [batch_size, seq_len, d_history]
        hidden = hidden.unsqueeze(-1)                                          # [batch_size, seq_len, num_events, d_history, 1]  if we need events else [batch_size, seq_len, d_history, 1]
        time_weight = self.weight_gen(hidden)                                  # [batch_size, seq_len, num_events, d_history, d_intensity] if we need events else [batch_size, seq_len, d_history, d_intensity]
        time_weight = self.activate(time_weight)                               # [batch_size, seq_len, num_events, d_history, d_intensity] if we need events else [batch_size, seq_len, d_history, d_intensity]

        # Mingle history and relative time embedding.
        output = torch.matmul(time_outside.unsqueeze(-2), time_weight).squeeze(-2)
                                                                               # [batch_size, seq_len, num_events, d_intensity] if we need events else [batch_size, seq_len, d_intensity]

        for layer_idx, layer in enumerate(self.mlp):
            output = layer(output)                                             # [batch_size, seq_len, num_events, d_intensity] if we need events else [batch_size, seq_len, d_intensity]
            # Imitate a weaker ReLU activation
            output = F.softplus(self.activate_factor[layer_idx]) * output
        
        integral = self.accu(output).squeeze(-1)                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]

        return integral
    
    def show_time_scale_factor(self):
        return F.softplus(self.activate_time_factor)

    def show_activate_factor(self):
        return F.softplus(self.activate_factor)

    def intensity(self, event_history, time_history, time_next, resolution, mean, var):
        '''
        Model intensity prober. Perhaps, we can support intensity integral as well.
        Args:
        event_history:[batch_size, seq_len]
        time_history: [batch_size, seq_len, 1]
        time_next:    [batch_size, seq_len, 1]
        resolution:   int
        '''
        time_multiplier = torch.linspace(0, 1, resolution, device = self.device)
                                                                               # [resolution]
        original_time_expand = time_multiplier * time_next                     # [batch_size, seq_len, resolution]
        if self.event_toggle:
            time_next = time_next.repeat(1, 1, self.num_events)                # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len, 1]
        time_history = time_history / var
        time_next = time_next / var

        if self.event_toggle:
            event_history_embedding = self.events(event_history)               # [batch_size, seq_len, d_history]
            history = torch.cat([
                event_history_embedding, time_history
            ], dim = -1)
        else:
            history = time_history                                             # [batch_size, seq_len, d_history + 1] if we need events else [batch_size, seq_len, 1]

        history_output, (_, _) = self.history(history)                         # [batch_size, seq_len, d_history]
        batch_size, seq_len, _ = history_output.shape

        history_expand = history_output.unsqueeze(-2).repeat(1, 1, resolution, 1)
                                                                               # [batch_size, seq_len, resolution, d_history]
        if self.event_toggle:
            history_expand = history_expand.unsqueeze(-2).repeat(1, 1, 1, self.num_events, 1)
                                                                               # [batch_size, seq_len, resolution, num_events, d_history]
        time_expand = time_multiplier.reshape(1, 1, resolution, 1) * time_next.unsqueeze(-2)
                                                                               # [batch_size, seq_len, resolution, num_events] if we need events else [batch_size, seq_len, resolution, 1]
        time_expand.requires_grad = True
        time_outside = self.time_outside(time_expand)                          # [batch_size, seq_len, resolution, num_events, d_history] if we need events else [batch_size, seq_len, resolution, d_history]

        time_weight = self.time_weight(self.activate_time(
            F.softplus(self.activate_time_factor) * time_expand))              # [batch_size, seq_len, resolution, num_events, d_history] if we need events else [batch_size, seq_len, resolution, d_history]
        hidden = history_expand + time_weight \
                    if not self.no_time_weight else 0                          # [batch_size, seq_len, resolution, num_events, d_history] if we need events else [batch_size, seq_len, resolution, d_history]
        hidden = hidden.unsqueeze(-1)                                          # [batch_size, seq_len, resolution, num_events, d_history, 1] if we need events else [batch_size, seq_len, resolution, d_history, 1]
        time_weight = self.weight_gen(hidden)                                  # [batch_size, seq_len, resolution, num_events, d_history, d_intensity] if we need events else [batch_size, seq_len, resolution, d_history, d_intensity]
        time_weight = self.activate(time_weight)                               # [batch_size, seq_len, resolution, num_events, d_history, d_intensity] if we need events else [batch_size, seq_len, resolution, d_history, d_intensity]

        # Mingle history and relative time embedding.
        output = torch.matmul(time_outside.unsqueeze(-2), time_weight).squeeze(-2)
                                                                               # [batch_size, seq_len, resolution, num_events, d_intensity] if we need events else [batch_size, seq_len, resolution, d_intensity]

        for layer_idx, layer in enumerate(self.mlp):
            output = layer(output)                                             # [batch_size, seq_len, resolution, num_events, d_intensity] if we need events else [batch_size, seq_len, resolution, d_intensity]
            # Imitate a weaker ReLU activation
            output = F.softplus(self.activate_factor[layer_idx]) * output

        output = self.accu(output).squeeze(-1)                                 # [batch_size, seq_len, resolution, num_events] if we need events else [batch_size, seq_len, resolution]
        if self.event_toggle:
            integral = output.reshape(batch_size, -1, self.num_events)
        else:
            integral = output.reshape(batch_size, -1)                          # [batch_size, seq_len * resolution, num_events] if we need events else [batch_size, seq_len * resolution]

        intensity = torch.autograd.grad(
            outputs = integral,
            inputs = time_expand,
            grad_outputs = torch.ones_like(integral),
            create_graph = True
        )[0]                                                                   # [batch_size, seq_len, resolution, num_events] if we need events else [batch_size, seq_len, resolution, 1]
        time_expand.requires_grad = False

        if self.event_toggle:
            intensity = intensity.reshape(batch_size, -1, self.num_events)
        else:
            intensity = intensity.reshape(batch_size, -1)                      # [batch_size, seq_len * resolution, num_events] if we need events else [batch_size, seq_len * resolution]

        timestamp = torch.cat(
            (torch.zeros((batch_size, seq_len, 1), device = self.device), original_time_expand.diff(dim = -1)),
            dim = -1)                                                          # [batch_size, seq_len, resolution]
        timestamp = timestamp.reshape(batch_size, seq_len * resolution)        # [batch_size, seq_len * resolution]

        return integral, intensity, timestamp

    def model_probe_function(self, event_history, time_history, time_next, resolution, mean, var):
        '''
        Model intensity prober. Perhaps, we can support intensity integral as well.
        Args:
        event_history:[batch_size, seq_len]
        time_history: [batch_size, seq_len, 1]
        time_next:    [batch_size, seq_len, 1]
        resolution:   int
        '''
        time_multiplier = torch.linspace(0, 1, resolution, device=self.device) # [resolution]
        original_time_expand = time_multiplier * time_next                     # [batch_size, seq_len, resolution]
        if self.event_toggle:
            time_next = time_next.repeat(1, 1, self.num_events)                # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len, 1]
        time_history = time_history / var
        time_next = time_next / var

        # Part 1: forward propagation.
        if self.event_toggle:
            event_history_embedding = self.events(event_history)               # [batch_size, seq_len, d_history]
            history = torch.cat([
                event_history_embedding, time_history
            ], dim = -1)
        else:
            history = time_history                                             # [batch_size, seq_len, d_history + 1] if we need events else [batch_size, seq_len, 1]

        history_output, (_, _) = self.history(history)                         # [batch_size, seq_len, d_history]
        batch_size, seq_len, _ = history_output.shape

        history_expand = history_output.unsqueeze(-2).repeat(1, 1, resolution, 1)
                                                                               # [batch_size, seq_len, resolution, d_history]
        if self.event_toggle:
            history_expand = history_expand.unsqueeze(-2).repeat(1, 1, 1, self.num_events, 1)
                                                                               # [batch_size, seq_len, resolution, num_events, d_history]
        time_expand = time_multiplier.reshape(1, 1, resolution, 1) * time_next.unsqueeze(-2)
                                                                               # [batch_size, seq_len, resolution, num_events] if we need events else [batch_size, seq_len, resolution, 1]
        time_expand.requires_grad = True
        time_outside = self.time_outside(time_expand)                          # [batch_size, seq_len, resolution, num_events, d_history] if we need events else [batch_size, seq_len, resolution, d_history]

        time_weight = self.time_weight(self.activate_time(
            F.softplus(self.activate_time_factor) * time_expand))              # [batch_size, seq_len, resolution, num_events, d_history] if we need events else [batch_size, seq_len, resolution, d_history]
        hidden = history_expand + time_weight \
                    if not self.no_time_weight else 0                          # [batch_size, seq_len, resolution, num_events, d_history] if we need events else [batch_size, seq_len, resolution, d_history]
        hidden = hidden.unsqueeze(-1)                                          # [batch_size, seq_len, resolution, num_events, d_history, 1] if we need events else [batch_size, seq_len, resolution, d_history, 1]
        time_weight = self.weight_gen(hidden)                                  # [batch_size, seq_len, resolution, num_events, d_history, d_intensity] if we need events else [batch_size, seq_len, resolution, d_history, d_intensity]
        time_weight = self.activate(time_weight)                               # [batch_size, seq_len, resolution, num_events, d_history, d_intensity] if we need events else [batch_size, seq_len, resolution, d_history, d_intensity]

        # Mingle history and relative time embedding.
        output = torch.matmul(time_outside.unsqueeze(-2), time_weight).squeeze(-2)
                                                                               # [batch_size, seq_len, resolution, num_events, d_intensity] if we need events else [batch_size, seq_len, resolution, d_intensity]
        output_after_dwg_layer = output                                        # [batch_size, seq_len, resolution, num_events, d_intensity] if we need events else [batch_size, seq_len, resolution, d_intensity]

        mlp_output = []
        for layer_idx, layer in enumerate(self.mlp):
            output = layer(output)                                             # [batch_size, seq_len, resolution, num_events, d_intensity] if we need events else [batch_size, seq_len, resolution, d_intensity]
            # Imitate a weaker ReLU activation
            output = F.softplus(self.activate_factor[layer_idx]) * output
            mlp_output.append(output)                                          # [batch_size, seq_len, resolution, num_events, d_intensity] * layer if we need events else [batch_size, seq_len, resolution, d_intensity] * layer

        output = self.accu(output).squeeze(-1)                                 # [batch_size, seq_len, resolution, num_events] if we need events else [batch_size, seq_len, resolution]
        if self.event_toggle:
            integral = output.reshape(batch_size, -1, self.num_events)
        else:
            integral = output.reshape(batch_size, -1)                          # [batch_size, seq_len * resolution, num_events] if we need events else [batch_size, seq_len * resolution]

        intensity = torch.autograd.grad(
            outputs = integral,
            inputs = time_expand,
            grad_outputs = torch.ones_like(integral),
            create_graph = True
        )[0]                                                                   # [batch_size, seq_len, resolution, num_events] if we need events else [batch_size, seq_len, resolution, 1]

        timestamp = torch.cat(
            (torch.zeros((batch_size, seq_len, 1), device = self.device), original_time_expand.diff(dim = -1)),
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
            )[0].mean(dim = -1).reshape(batch_size, -1)                        # [batch_size, seq_len * resolution]

        dwg_gradient = torch.autograd.grad(
            outputs = output_after_dwg_layer,
            inputs = time_expand,
            grad_outputs = torch.ones_like(output_after_dwg_layer),
            create_graph = True
        )[0].mean(dim = -1).reshape(batch_size, -1)                            # [batch_size, seq_len * resolution]

        '''
        Compatibility with ploter
        '''
        intensity = intensity.sum(dim = -1).reshape(batch_size, -1)            # [batch_size, seq_len * resolution]
        if self.event_toggle:
            integral = integral.sum(dim = -1)                                  # [batch_size, seq_len * resolution]

        time_expand.requires_grad = False
        result = {
            'final_output': integral,
            'accumulated_gradient': intensity,
            **mlp_gradient,
            'dwg_gradient': dwg_gradient,
        }

        return result, timestamp