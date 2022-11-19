import torch.nn as nn
import torch
from scipy.stats import spearmanr
import numpy as np

from .nonneg import NonNegLinear
from .activate import *
from .transformers import TransEncoder


TA = {
    # Vanilla Softplus harms the algorithm by shifting the entire distribution into the non-nrgative area.
    # That is to say, each scalar in the output vector is bigger than log(2) if all hidden layer weights only
    # have positive numbers like what FullyNN does.
    # We have vanilla version and symmetrical version of softplus
    'softplus': nn.Softplus,
    'sym_softplus': sym_softplus,

    # Some papers have pointed out that Tanh introduces significant gradient vanishment when the input time is too big. After theoretical
    # analysis, we argue that this feature is required by approaches like FullyNN to fit long-tail functions like Hawkes intensity function.
    'tanh': nn.Tanh,
    # Yet another function that has small gradients when it has big inputs. But as the log function is not bounded above, the hard integral bound introduced
    # by tanh can be alleviated.
    'log': sym_Log,
    # This activation can perfectly show why FullyNN needs tanh to attain a trade-off between intensity function regression ability and extrapolation 
    'identity': nn.Identity,
    # Might be the redeemer, but I'm not sure.
    'ploy': sym_Polynomial
}

class FullyNN(nn.Module):
    '''
    This is our implementation of Omi's paper: Fully Neural Network based Model for General Temporal Point Processes
    Hope it can work properly.

    Currently, normalization is disabled.
    Update: 2022-01-19: Now you can use data normalization via synthetic dataloader.

    Following Babylon's paper, we would check the performance of FullyNN with integral offsets.
    '''

    def __init__(self, d_history, d_intensity, num_events, dropout, history_module, history_module_layers,
                 mlp_layers, nonlinear, event_toggle, n_head, wq_nonneg, wk_nonneg, wv_nonneg, split_comp_graph, 
                 zero_shift, device):
        super(FullyNN, self).__init__()
        self.device = device
        self.num_events = num_events
        self.event_toggle = event_toggle
        self.history_module = history_module.lower()
        self.split_comp_graph = split_comp_graph
        self.zero_shift = zero_shift

        #　Maybe we can decompose self.hidden_x into the multiplication of two smaller matrices.
        if self.event_toggle:
            self.events = nn.Embedding(num_events + 1, d_history, padding_idx = num_events, device = device)
            if self.history_module == 'lstm':
                self.his_encoder = nn.LSTM(input_size = d_history + 1, hidden_size = d_history, num_layers = history_module_layers,\
                            batch_first = True, dropout = dropout, device = device)
            elif self.history_module == 'transformers':
                self.his_encoder = TransEncoder(num_types = num_events + 1, d_input = d_history, d_hidden = 4 * d_history, \
                            n_layers = history_module_layers, n_head = n_head, d_qk = d_history, d_v = d_history, dropout = dropout, \
                            event_toggle = event_toggle, wq_nonneg = wq_nonneg, wk_nonneg = wk_nonneg, wv_nonneg = wv_nonneg, device = device)
            else:
                raise Exception(f'Unknown history module name {history_module}.')
            if split_comp_graph:
                self.hidden_x = nn.Parameter(torch.zeros((self.num_events, d_intensity), device = self.device, requires_grad = True))
            else:
                self.hidden_x = nn.Parameter(torch.zeros((1, d_intensity), device = self.device, requires_grad = True))
        else:
            self.events = None
            if self.history_module == 'lstm':
                self.his_encoder = nn.LSTM(input_size = 1, hidden_size = d_history, num_layers = history_module_layers,\
                            batch_first = True, dropout = dropout, device = device)
            elif self.history_module == 'transformers':
                self.his_encoder = TransEncoder(num_types = num_events + 1, d_input = d_history, d_hidden = 4 * d_history, \
                            n_layers = history_module_layers, n_head = n_head, d_qk = d_history, d_v = d_history, dropout = dropout, \
                            event_toggle = event_toggle, wq_nonneg = wq_nonneg, wk_nonneg = wk_nonneg, wv_nonneg = wv_nonneg, device = device)
            else:
                raise Exception(f'Unknown history module name {history_module}.')
            self.hidden_x = nn.Parameter(torch.zeros((1, d_intensity), device = self.device, requires_grad = True))

        # self.hidden_x = NonNegLinear(self.num_events, d_intensity, bias = False, device = device)
        self.hidden_time = NonNegLinear(d_intensity, d_intensity, device = self.device)
        nn.init.xavier_uniform_(self.hidden_x)

        self.hidden_p = nn.Linear(d_history, d_intensity, bias = True, device = device)

        # The original implement counts the hidden_x as one of mlp_layers
        self.mlp = nn.ModuleList([
            NonNegLinear(d_intensity, d_intensity, bias = True, device = device) for _ in range(mlp_layers)
        ])

        self.agg = NonNegLinear(d_intensity, 1, bias = True, device = device)

        self.activate = TA[nonlinear]()
        self.non_neg = nn.Softplus()


    def forward(self, events_history, time_history, time_next, mean, var, mask):
        '''
        Args:
            events_history: [batch_size, seq_len]
            time_history:   [batch_size, seq_len, 1]
            time_next:      [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len, 1]
            mask:           [batch_size, seq_len]
        '''
        # Input data normalization
        time_history = (time_history - mean) / var                             # [batch_size, seq_len, 1]
        time_next = (time_next - mean) / var                                   # [batch_size, seq_len, num_events]

        if self.event_toggle:
            events_embeddings = self.events(events_history)                    # [batch_size, seq_len, d_history]
            history = torch.cat(
                (events_embeddings, time_history), dim = -1
            )
        else:
            history = time_history                                             # [batch_size, seq_len, d_history + 1] if we need events else [batch_size, seq_len, 1]
        
        # Reshape hidden output for full connection layers.
        if self.history_module == 'lstm':
            history_output, (_, _) = self.his_encoder(history)                 # [batch_size, seq_len, d_history]
        elif self.history_module == 'transformers':
            history_output = self.his_encoder(events_history, time_history, mask.unsqueeze(dim = -1))
                                                                               # [batch_size, seq_len, d_history]

        if self.event_toggle:
            history_output = history_output.unsqueeze(-2).repeat(1, 1, self.num_events, 1)
                                                                               # [batch_size, seq_len, num_events, d_history] if we need events else [batch_size, seq_len, d_history]
            time = time_next.unsqueeze(-1) * self.non_neg(self.hidden_x)       # [batch_size, seq_len, num_events, d_intensity]
        else:
            time = time_next * self.non_neg(self.hidden_x)                     # [batch_size, seq_len, d_intensity]

        hidden_history = self.hidden_p(history_output)                         # [batch_size, seq_len, num_events, d_intensity] if we need events else [batch_size, seq_len, d_intensity]
        time = self.hidden_time(time)                                          # [batch_size, seq_len, num_events, d_intensity] if we need events else [batch_size, seq_len, d_intensity]
        output = self.activate(time + hidden_history)                          # [batch_size, seq_len, num_events, d_intensity] if we need events else [batch_size, seq_len, d_intensity]

        for layer in self.mlp:
            output = layer(output)                                             # [batch_size, seq_len, num_events, d_intensity] if we need events else [batch_size, seq_len, d_intensity]
            output = self.activate(output)                                     # [batch_size, seq_len, num_events, d_intensity] if we need events else [batch_size, seq_len, d_intensity]

        integral = self.non_neg(self.agg(output))                              # [batch_size, seq_len, num_events, 1] if we need events else [batch_size, seq_len, 1]

        if self.zero_shift:
            zero = torch.ones_like(time_next, device = self.device) * (-mean / var)
                                                                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len, 1]
            if self.event_toggle:
                zero_emb = zero.unsqueeze(-1) * self.non_neg(self.hidden_x)    # [batch_size, seq_len, num_events, d_intensity]
            else:
                zero_emb = zero * self.non_neg(self.hidden_x)                  # [batch_size, seq_len, d_intensity]
            zero_emb = self.hidden_time(zero_emb)                              # [batch_size, seq_len, num_events, d_intensity] if we need events else [batch_size, seq_len, d_intensity]
            zero_output = self.activate(zero_emb + hidden_history)             # [batch_size, seq_len, num_events, d_intensity] if we need events else [batch_size, seq_len, d_intensity]
            for layer in self.mlp:
                zero_output = layer(zero_output)                               # [batch_size, seq_len, num_events, d_intensity] if we need events else [batch_size, seq_len, d_intensity]
                zero_output = self.activate(zero_output)                       # [batch_size, seq_len, num_events, d_intensity] if we need events else [batch_size, seq_len, d_intensity]
            
            zero_integral = self.non_neg(self.agg(zero_output))                # [batch_size, seq_len, num_events, 1] if we need events else [batch_size, seq_len, 1]

        if self.zero_shift:
            integral = integral - zero_integral.detach()                       # [batch_size, seq_len, num_events, 1] if we need events else [batch_size, seq_len, 1]

        if self.event_toggle:
            integral = integral.squeeze(dim = -1)                              # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len, 1]

        return integral

    def integral_intensity(self, events_history, time_history, time_next, resolution, mean, var, mask, sum = True):
        '''
        Intensity integral & intensity function prober. Perhaps, we can support intensity integral as well.
        Args:
        events_history:[batch_size, seq_len]
        time_history:  [batch_size, seq_len, 1]
        time_next:     [batch_size, seq_len, 1]
        resolution:    int
        '''        
        time_multiplier = torch.linspace(0, 1, resolution, device = self.device)
                                                                               # [resolution]
        original_time_expand = time_multiplier * time_next                     # [batch_size, seq_len, resolution]
        time_history = (time_history - mean) / var                             # [batch_size, seq_len, 1]
        if self.event_toggle:
            time_expand = original_time_expand.unsqueeze(dim = -1).repeat(1, 1, 1, self.num_events)
                                                                               # [batch_size, seq_len, resolution, num_events]
        else:
            time_expand = original_time_expand.unsqueeze(dim = -1)             # [batch_size, seq_len, resolution, 1]

        if self.event_toggle:
            events_embeddings = self.events(events_history)                    # [batch_size, seq_len, d_history]
            history = torch.cat(
                (events_embeddings, time_history), dim = -1
            )
        else:
            history = time_history                                             # [batch_size, seq_len, d_history + 1] if we need events else [batch_size, seq_len, 1]
        
        if self.history_module == 'lstm':
            output, (_, _) = self.his_encoder(history)                         # [batch_size, seq_len, d_history]
        elif self.history_module == 'transformers':
            output = self.his_encoder(events_history, time_history, mask.unsqueeze(dim = -1))
                                                                               # [batch_size, seq_len, d_history]

        if self.event_toggle:
            output = output.unsqueeze(-2).repeat(1, 1, self.num_events, 1)     # [batch_size, seq_len, num_events, d_history]

        hidden = self.hidden_p(output)                                         # [batch_size, seq_len, num_events, d_intensity] if we need events else [batch_size, seq_len, d_intensity]
        if self.event_toggle:
            history_expand = hidden.unsqueeze(-3).repeat(1, 1, resolution, 1, 1)
                                                                               # [batch_size, seq_len, resolution, num_events, d_history]
        else:
            history_expand = hidden.unsqueeze(-2).repeat(1, 1, resolution, 1)  # [batch_size, seq_len, resolution, d_history]
        batch_size, seq_len = history_expand.shape[0], history_expand.shape[1]

        time_expand.requires_grad = True
        time_expand_norm = (time_expand - mean) / var                          # [batch_size, seq_len, resolution, num_events] is we need events else [batch_size, seq_len, resolution, 1]

        if self.event_toggle:
            emb_time_expand = time_expand_norm.unsqueeze(-1) * self.non_neg(self.hidden_x)
                                                                               # [batch_size, seq_len, resolution, num_events, d_intensity]
        else:
            emb_time_expand = time_expand_norm * self.non_neg(self.hidden_x)   # [batch_size, seq_len, resolution, d_intensity]

        emb_time_expand = self.hidden_time(emb_time_expand)                    # [batch_size, seq_len, resolution, num_events, d_intensity] if we need events else [batch_size, seq_len, resolution, d_intensity]
        output = self.activate(emb_time_expand + history_expand)               # [batch_size, seq_len, resolution, num_events, d_intensity] if we need events else [batch_size, seq_len, resolution, d_intensity]

        for layer in self.mlp:
            output = layer(output)                                             # [batch_size, seq_len, resolution, num_events, d_intensity] if we need events else [batch_size, seq_len, resolution, d_intensity]
            output = self.activate(output)                                     # [batch_size, seq_len, resolution, num_events, d_intensity] if we need events else [batch_size, seq_len, resolution, d_intensity]

        expand_integral = self.non_neg(self.agg(output)).squeeze(-1)           # [batch_size, seq_len, resolution, num_events] if we need events else [batch_size, seq_len, resolution]

        if self.zero_shift:
            if self.event_toggle:
                expand_integral = expand_integral - expand_integral[:, :, 0, :].detach().unsqueeze(dim = -2)
                                                                               # [batch_size, seq_len, resolution, num_events]
            else:
                expand_integral = expand_integral - expand_integral[:, :, 0].detach().unsqueeze(dim = -1)
                                                                               # [batch_size, seq_len, resolution]

        expand_intensity = torch.autograd.grad(
            outputs=expand_integral,
            inputs=time_expand,
            grad_outputs=torch.ones_like(expand_integral),
        )[0]                                                                   # [batch_size, seq_len, resolution, num_events] if we need events else [batch_size, seq_len, resolution, 1]
        time_expand.requires_grad = False

        expand_integral = expand_integral.detach()
        expand_intensity = expand_intensity.detach()

        if self.event_toggle:
            expand_integral = expand_integral.reshape(batch_size, seq_len * resolution, -1)
                                                                               # [batch_size, seq_len * resolution, num_events]
            expand_intensity = expand_intensity.reshape(batch_size, seq_len * resolution, -1)
                                                                               # [batch_size, seq_len * resolution, num_events]

            if sum:
                expand_integral = expand_integral.sum(dim = -1)                # [batch_size, seq_len * resolution]
                expand_intensity = expand_intensity.sum(dim = -1)              # [batch_size, seq_len * resolution]
        else:
            expand_intensity = expand_intensity.squeeze(-1).reshape(batch_size, seq_len * resolution)
                                                                               # [batch_size, seq_len * resolution]
            expand_integral = expand_integral.reshape(batch_size, seq_len * resolution)
                                                                               # [batch_size, seq_len * resolution]

        '''
        Restore the original timestamp
        '''
        timestamp = torch.cat(
            (torch.zeros((batch_size, seq_len, 1), device = self.device), original_time_expand.diff(dim = -1)),
            dim = -1)                                                          # [batch_size, seq_len, resolution]
        timestamp = timestamp.reshape(batch_size, seq_len * resolution)        # [batch_size, seq_len * resolution]

        return expand_integral, expand_intensity, timestamp

    def model_probe_function(self, events_history, time_history, time_next, resolution, mean, var, mask):
        '''
        We use this function to dive into the fullynn and find the reason of abrupt gradient drop around 0
        Args:
        time_history: [batch_size, seq_len, 1]
        time_next:    [batch_size, seq_len, num_events]
        resolution:   int
        '''
        time_multiplier = torch.linspace(0, 1, resolution, device = self.device)
                                                                               # [resolution]
        original_time_expand = time_multiplier * time_next                     # [batch_size, seq_len, resolution]
        time_history = (time_history - mean) / var                             # [batch_size, seq_len, 1]
        if self.event_toggle:
            time_expand = original_time_expand.unsqueeze(dim = -1).repeat(1, 1, 1, self.num_events)
                                                                               # [batch_size, seq_len, resolution, num_events]
        else:
            time_expand = original_time_expand.unsqueeze(dim = -1)             # [batch_size, seq_len, resolution, 1]

        if self.event_toggle:
            events_embeddings = self.events(events_history)                    # [batch_size, seq_len, d_history]
            history = torch.cat(
                (events_embeddings, time_history), dim = -1
            )
        else:
            history = time_history                                             # [batch_size, seq_len, d_history + 1] if we need events else [batch_size, seq_len, 1]

        if self.history_module == 'lstm':
            output, (_, _) = self.his_encoder(history)                         # [batch_size, seq_len, d_history]
        elif self.history_module == 'transformers':
            output = self.his_encoder(events_history, time_history, mask.unsqueeze(dim = -1))
                                                                               # [batch_size, seq_len, d_history]

        if self.event_toggle:
            output = output.unsqueeze(-2).repeat(1, 1, self.num_events, 1)     # [batch_size, seq_len, num_events, d_history]
        hidden = self.hidden_p(output)                                         # [batch_size, seq_len, num_events, d_intensity] if we need events else [batch_size, seq_len, d_intensity]
        if self.event_toggle:
            history_expand = hidden.unsqueeze(-3).repeat(1, 1, resolution, 1, 1)
                                                                               # [batch_size, seq_len, resolution, num_events, d_history]
        else:
            history_expand = hidden.unsqueeze(-2).repeat(1, 1, resolution, 1)  # [batch_size, seq_len, resolution, d_history]
        batch_size, seq_len = history_expand.shape[0], history_expand.shape[1]

        time_expand.requires_grad = True      
        time_expand_norm = (time_expand - mean) / var                          # [batch_size, seq_len, resolution]

        if self.event_toggle:
            emb_time_expand = time_expand_norm.unsqueeze(-1) * self.non_neg(self.hidden_x)
                                                                               # [batch_size, seq_len, resolution, num_events, d_intensity]
        else:
            emb_time_expand = time_expand_norm * self.non_neg(self.hidden_x)   # [batch_size, seq_len, resolution, d_intensity]

        emb_time_expand = self.hidden_time(emb_time_expand)                    # [batch_size, seq_len, resolution, num_events, d_intensity] if we need events else [batch_size, seq_len, resolution, d_intensity]
        output = self.activate(emb_time_expand + history_expand)               # [batch_size, seq_len, resolution, num_events, d_intensity] if we need events else [batch_size, seq_len, resolution, d_intensity]
        output_storage = [output]                                              # [batch_size, seq_len, resolution, num_events, d_intensity] if we need events else [batch_size, seq_len, resolution, d_intensity]

        for layer in self.mlp:
            output = layer(output)                                             # [batch_size, seq_len, resolution, num_events, d_intensity] if we need events else [batch_size, seq_len, resolution, d_intensity]
            output = self.activate(output)                                     # [batch_size, seq_len, resolution, num_events, d_intensity] if we need events else [batch_size, seq_len, resolution, d_intensity]
            output_storage.append(output)                                      # [batch_size, seq_len, resolution, num_events, d_intensity] * (self.mlp_size + 1) if we need events else [batch_size, seq_len, resolution, d_intensity] * (self.mlp_size + 1)
        
        accumulative_layer_output = self.agg(output).squeeze(-1)               # [batch_size, seq_len, resolution, num_events] if we need events else [batch_size, seq_len, resolution]
        expand_integral = self.non_neg(accumulative_layer_output)              # [batch_size, seq_len, resolution, num_events] if we need events else [batch_size, seq_len, resolution]

        timestamp = torch.cat(
            (torch.zeros((batch_size, seq_len, 1), device = self.device), original_time_expand.diff(dim = -1)),
            dim = -1)                                                          # [batch_size, seq_len, resolution]
        timestamp = timestamp.reshape(batch_size, seq_len * resolution)        # [batch_size, seq_len * resolution]

        # Gradient 1: Integral -> time
        event_gradient = torch.autograd.grad(
            outputs=expand_integral,
            inputs=time_expand,
            grad_outputs=torch.ones_like(expand_integral),
            retain_graph=True
            # create_graph=True,
        )[0]                                                                   # [batch_size, seq_len, resolution, num_events] if we need events else [batch_size, seq_len, resolution, 1]

        # Gradient 2: All layer output -> time
        output_storage_gradient = {}
        for idx, item in enumerate(output_storage):
            subgradient = torch.autograd.grad(
                outputs=item,
                inputs=time_expand,
                grad_outputs=torch.ones_like(item),
                retain_graph=True
                # create_graph=True,
            )[0]                                                               # [batch_size, seq_len, resolution, num_events] if we need events else [batch_size, seq_len, resolution, 1]
            subgradient = subgradient.sum(dim = -1).reshape(batch_size, -1)    # [batch_size, seq_len * resolution]
            output_storage_gradient[f'mlp_{idx}_grad'] = subgradient.detach()  # [batch_size, seq_len * resolution] * (self.mlp.size + 1)
                
        time_expand.requires_grad = False

        accumulated_gradient = event_gradient.sum(dim = -1).reshape(batch_size, -1).detach()
                                                                               # [batch_size, seq_len * resolution]
        if self.event_toggle:
            intensities = event_gradient.chunk(self.num_events, dim = -1)      # [batch_size, seq_len, resolution] * num_events
            event_intensity = {}
            for idx, item in enumerate(intensities):
                formed_intensity = item.reshape(batch_size, -1)
                event_intensity[f'event_intensity_{idx}'] = formed_intensity.detach()
                                                                               # [batch_size, seq_len * resolution]

            # additional plot, measure the spearman correlation across available events.
            additional_plot = {
                'heatmap': []
            }

            expand_intensity = event_gradient.reshape(event_gradient.shape[0], -1, event_gradient.shape[-1]).detach().cpu().numpy()
                                                                               # [batch_size, seq_len * resolution, num_event]

            for idx, item in enumerate(expand_intensity):
                heatmap_data = {}
                # rho: spearman coefficient
                heatmap_data['spearman'] = spearmanr(item)[0]
                if self.num_events == 2:
                    heatmap_data['spearman'] = np.array([[1, heatmap_data['spearman']], [heatmap_data['spearman'], 1]])

                # r: pearson coefficient
                heatmap_data['pearson'] = np.corrcoef(item, rowvar = False)
                # L^1 metric
                heatmap_data['L1'] = L1_distance(item, resolution = resolution, num_events = self.num_events, time_next = time_next[idx])

                # add plots
                for key, value in heatmap_data.items():
                    idx = 0
                    additional_plot['heatmap'].append(
                    [
                        f'{key}_{idx}',
                        {
                            'data': value,
                            'cmap': "YlGnBu",
                            'vmin': 0,
                            'vmax': max(5, np.max(value)),
                            'annot': True
                        }
                    ])
                    idx += 1
    
            result = {
                **{'accumulated_gradient': accumulated_gradient},\
                **output_storage_gradient,\
                **event_intensity,\
                **{"final_output": expand_integral.sum(dim = -1).reshape(batch_size, -1).detach()},
                "loss": -torch.log(accumulated_gradient) + expand_integral.sum(dim = -1).detach().reshape(batch_size, -1)
                }
        else:
            result = {
                **{'accumulated_gradient': accumulated_gradient},\
                **output_storage_gradient,\
                **{"final_output": expand_integral.reshape(batch_size, -1).detach()},
                "loss": -torch.log(accumulated_gradient) + expand_integral.detach().reshape(batch_size, -1)
                }

        return result, additional_plot, timestamp


class InvertedBottleneck(nn.Module):
    def __init__(self, d_input, d_hidden, device, no_bottleneck, no_norm, no_activate):
        super(InvertedBottleneck, self).__init__()

        self.no_bottleneck = no_bottleneck
        self.no_norm = no_norm
        self.no_activate = no_activate

        self.expand = nn.Linear(d_input, d_hidden, device = device)
        self.bottleneck = nn.Linear(d_hidden, d_input, device = device)
        self.norm = nn.LayerNorm(d_input, device = device)

        self.activate = nn.GELU()

    def forward(self, x):
        residual = x

        if not self.no_norm:
            x = self.norm(x)                                                   # [..., d_input]
        if not self.no_bottleneck:
            x = self.expand(x)                                                 # [..., d_hidden]
            x = self.bottleneck(x)                                             # [..., d_input]
        if not self.no_activate:
            x = self.activate(x)                                               # [..., d_input]

        return residual + x

def L1_distance(input, resolution, num_events, time_next):
    '''
    This function calculates the L^1 distance between two functions in scattered form.
    Input:
    1. input:      function values
                   [seq_len * resolution, num_events]
    2. resolution: int
                   the number of points from [t_{i - 1}, t_i]
    3. num_event:  int
                   the number of event types
    4. time_next:  [seq_len, num_events]
                   the length of all intervals with interpolations.
    '''

    input = input.reshape(-1, resolution, num_events)                          # [seq_len, resolution, num_events]
    input = np.transpose(input, (2, 0, 1))                                     # [num_events, seq_len, resolution]
    intensity_1 = np.expand_dims(input, axis = 1).repeat(num_events, axis = 1) # [num_events, num_events, seq_len, resolution]
    intensity_2 = np.expand_dims(input, axis = 0).repeat(num_events, axis = 0) # [num_events, num_events, seq_len, resolution]
    delta_intensity = np.abs(intensity_1 - intensity_2)                        # [num_events, num_events, seq_len, resolution]

    gap = np.expand_dims(time_next.cpu(), axis = 1)                            # [num_events, 1, seq_len]
    gap = gap / (resolution - 1)
    gap = np.transpose(gap, (2, 0, 1))                                         # [num_events, seq_len, 1]
    gap = np.expand_dims(gap, axis = 1)                                        # [num_events, 1, seq_len, 1]

    L1 = (delta_intensity * gap)[:, :, :, :-1].sum(axis = -1).sum(axis = -1)   # [num_events, num_events]

    # round up the value smaller than 1e-6
    L1[L1 < 1e-6] = 0

    return L1