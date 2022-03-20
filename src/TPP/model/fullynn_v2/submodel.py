import torch.nn as nn
import torch

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
    # analysis, we argue that this feature is required by approaches like FullyNN to regress long-tail functions like Hawkes intensity function.
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

    def __init__(self, d_history, d_intensity, num_events, dropout, self_attn_layer, mlp_layers, nonlinear, n_head, device):
        super(FullyNN, self).__init__()
        self.device = device
        self.num_events = num_events

        self.history_encoder = TransEncoder(num_types = num_events + 1, d_input = d_history, \
                                d_hidden = 4 * d_history, n_layers = self_attn_layer,\
                                n_head = n_head, d_qk = d_history, d_v = d_history, \
                                dropout = dropout, device = device)
        
        #　Maybe we can decompose self.hidden_x into the multiplication of two smaller matrices.
        self.relative_time_affine = nn.Parameter(torch.zeros(num_events,  d_intensity, device = device, requires_grad = True))
        self.absolute_time_affine = nn.Parameter(torch.zeros(num_events,  d_intensity, device = device, requires_grad = True))
        self.hidden_time = NonNegLinear(d_intensity, d_intensity, device = self.device)
        nn.init.xavier_uniform_(self.relative_time_affine)
        nn.init.xavier_uniform_(self.absolute_time_affine)

        self.hidden_p = nn.Linear(d_history, d_intensity, bias = True, device = device)

        # The original implement counts the hidden_x as one of mlp_layers
        self.mlp = nn.ModuleList([
            NonNegLinear(d_intensity, d_intensity, bias = True, device = device) for _ in range(mlp_layers)
        ])

        self.agg = NonNegLinear(d_intensity, 1, bias = True, device = device)

        self.activate = TA[nonlinear]()
        self.non_neg = nn.Softplus()
        self.non_neg_weight = nn.ReLU()


    def forward(self, events_history, time_history, time_next, mask, mean, var):
        '''
        Args:
            events_history: [batch_size, seq_len]
            time_history:   [batch_size, seq_len, 1]
            time_next:      [batch_size, seq_len, num_events]
        '''
        # Input data normalization
        time_history = (time_history - mean) / var                             # [batch_size, seq_len, 1]
        time_next = (time_next - mean) / var                                   # [batch_size, seq_len, num_events]
        mask = mask.unsqueeze(dim = -1)                                        # [batch_size, seq_len, 1]

        # Reshape hidden output for full connection layers.
        output = self.history_encoder(events_history, time_history, non_pad_mask = mask)
                                                                               # [batch_size, seq_len, d_history]
        output = output.unsqueeze(-2).repeat(1, 1, self.num_events, 1)         # [batch_size, seq_len, num_events, d_history]
        hidden = self.hidden_p(output)                                         # [batch_size, seq_len, num_events, d_intensity]

        relative_time_emb = time_next.unsqueeze(-1) * self.non_neg_weight(self.relative_time_affine)
                                                                               # [batch_size, seq_len, num_events, d_intensity]
        absolute_time_emb = torch.cumsum(time_next, dim = -1).unsqueeze(-1) * self.non_neg_weight(self.absolute_time_affine)
                                                                               # [batch_size, seq_len, num_events, d_intensity]

        relative_time_emb = self.hidden_time(relative_time_emb)                # [batch_size, seq_len, num_events, d_intensity]
        output = self.activate(relative_time_emb + hidden)                     # [batch_size, seq_len, num_events, d_intensity]

        for layer in self.mlp:
            output = layer(output)                                             # [batch_size, seq_len, num_events, d_intensity]
            output = self.activate(output)                                     # [batch_size, seq_len, num_events, d_intensity]
        
        integral_for_each_event = self.non_neg(self.agg(output)).squeeze(-1)   # [batch_size, seq_len, num_events]
        # output = self.non_neg_norm(output)                                   # [batch_size, seq_len, num_events]
        # integral_for_each_event = -torch.log(1 - integral_for_each_event + 1e-9)
        #                                                                      # [batch_size, seq_len, num_events]
        # integral = integral_for_each_event.sum(dim = -1)                       # [batch_size, seq_len]

        return integral_for_each_event

    def integral_intensity(self, events_history, time_history, time_next, resolution, mask, mean, var):
        '''
        Intensity integral & intensity function prober. Perhaps, we can support intensity integral as well.
        Args:
        time_history: [batch_size, seq_len, 1]
        time_next:    [batch_size, seq_len, 1]
        resolution:   int
        '''
        time_multiplier = torch.linspace(0, 1, resolution, device = self.device)
                                                                               # [resolution]
        original_time_expand = time_multiplier * time_next                     # [batch_size, seq_len, resolution]
        time_next = time_next.repeat(1, 1, self.num_events)                    # [batch_size, seq_len, num_events]
        time_history = (time_history - mean) / var                             # [batch_size, seq_len, 1]
        time_next = (time_next - mean) / var                                   # [batch_size, seq_len, num_events]
        mask = mask.unsqueeze(dim = -1)                                        # [batch_size, seq_len, 1]

        output = self.history_encoder(events_history, time_history, non_pad_mask = mask)
                                                                               # [batch_size, seq_len, d_history]
        output = output.unsqueeze(-2).repeat(1, 1, self.num_events, 1)         # [batch_size, seq_len, num_events, d_history]
        hidden = self.hidden_p(output)                                         # [batch_size, seq_len, num_events, d_intensity]
        batch_size, seq_len, _, _ = hidden.shape
        hidden_expand = hidden.unsqueeze(2).repeat(1, 1, resolution, 1, 1)     # [batch_size, seq_len, resolution, num_events, d_intensity]

        time_expand = time_multiplier.reshape(1, 1, resolution, 1) * time_next.unsqueeze(-2)
                                                                               # [batch_size, seq_len, resolution, num_events]
        time_expand.requires_grad = True
        emb_relative_time_expand = time_expand.unsqueeze(-1) * self.non_neg_weight(self.relative_time_affine)
                                                                               # [batch_size, seq_len, resolution, num_events, d_intensity]
        emb_absolute_time_expand = time_expand.unsqueeze(-1) * self.non_neg_weight(self.absolute_time_affine)
                                                                               # [batch_size, seq_len, resolution, num_events, d_intensity]
        emb_relative_time_expand = self.hidden_time(emb_relative_time_expand)  # [batch_size, seq_len, resolution, num_events, d_intensity]
        output = self.activate(emb_relative_time_expand + hidden_expand)
                                                                               # [batch_size, seq_len, resolution, num_events, d_intensity]

        for layer in self.mlp:
            output = layer(output)                                             # [batch_size, seq_len, resolution, num_events, d_intensity]
            output = self.activate(output)                                     # [batch_size, seq_len, resolution, num_events, d_intensity]

        expand_integral_for_each_event = self.non_neg(self.agg(output)).squeeze(-1)
                                                                               # [batch_size, seq_len, resolution, num_events]
        # output = self.non_neg_norm(output)                                   # [batch_size, seq_len, resolution, num_events]
        # expand_integral_for_each_event = -torch.log(1 - output + 1e-9)       # [batch_size, seq_len, resolution, num_events]
        expand_integral = expand_integral_for_each_event.sum(dim = -1)         # [batch_size, seq_len, resolution]
        
        expand_intensity = torch.autograd.grad(
            outputs=expand_integral,
            inputs=time_expand,
            grad_outputs=torch.ones_like(expand_integral),
            create_graph=True,
        )[0]                                                                   # [batch_size, seq_len, resolution, num_events]
        time_expand.requires_grad = False

        expand_integral = expand_integral.reshape(batch_size, -1)              # [batch_size, seq_len * resolution]
        expand_intensity = expand_intensity.sum(dim = -1).reshape(batch_size, -1)
                                                                               # [batch_size, seq_len * resolution]

        '''
        Restore the original timestamp
        '''
        timestamp = torch.cat(
            (torch.zeros((batch_size, seq_len, 1), device = self.device), original_time_expand.diff(dim = -1)),
            dim = -1)                                                          # [batch_size, seq_len, resolution]
        timestamp = timestamp.reshape(batch_size, seq_len * resolution)        # [batch_size, seq_len * resolution]

        return expand_integral, expand_intensity, timestamp

    def model_probe_function(self, events_history, time_history, time_next, resolution, mask, mean, var):
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
        time_next = time_next.repeat(1, 1, self.num_events)                    # [batch_size, seq_len, num_events]
        time_history = (time_history - mean) / var                             # [batch_size, seq_len, 1]
        time_next = (time_next - mean) / var                                   # [batch_size, seq_len, num_events]
        mask = mask.unsqueeze(dim = -1)                                        # [batch_size, seq_len, 1]

        output = self.history_encoder(events_history, time_history, non_pad_mask = mask)
                                                                               # [batch_size, seq_len, d_history]
        output = output.unsqueeze(-2).repeat(1, 1, self.num_events, 1)         # [batch_size, seq_len, num_events, d_history]
        hidden = self.hidden_p(output)                                         # [batch_size, seq_len, num_events, d_intensity]
        batch_size, seq_len, num_events, d_intensity = hidden.shape
        hidden_expand = hidden.unsqueeze(2).repeat(1, 1, resolution, 1, 1)     # [batch_size, seq_len, resolution, num_events, d_intensity]

        time_expand = time_multiplier.reshape(1, 1, resolution, 1) * time_next.unsqueeze(-2)
                                                                               # [batch_size, seq_len, resolution, num_events]
        time_expand.requires_grad = True
        emb_time_expand = time_expand.unsqueeze(-1) * self.non_neg_weight(self.relative_time_affine)
                                                                               # [batch_size, seq_len, resolution, num_events, d_intensity]
        emb_time_expand = self.hidden_time(emb_time_expand)                    # [batch_size, seq_len, resolution, num_events, d_intensity]
        output = self.activate(emb_time_expand + hidden_expand)                # [batch_size, seq_len, resolution, num_events, d_intensity]
        output_storage = [output]                                              # [batch_size, seq_len, resolution, num_events, d_intensity]

        for layer in self.mlp:
            output = layer(output)                                             # [batch_size, seq_len, resolution, num_events, d_intensity]
            output = self.activate(output)                                     # [batch_size, seq_len, resolution, num_events, d_intensity]
            output_storage.append(output)                                      # [batch_size, seq_len, resolution, num_events, d_intensity] * (self.mlp.size + 1)

        accumulative_layer_output = self.non_neg(self.agg(output)).squeeze(-1) # [batch_size, seq_len, resolution, num_events]
        # expand_integral_for_each_event = self.non_neg_norm(accumulative_layer_output)
        #                                                                        # [batch_size, seq_len, resolution, num_events]
        # expand_integral_for_each_event = -torch.log(1 - expand_integral_for_each_event + 1e-9)
        #                                                                        # [batch_size, seq_len, resolution, num_events]
        expand_integral = accumulative_layer_output.sum(dim = -1)              # [batch_size, seq_len, resolution]

        timestamp = torch.cat(
            (torch.zeros((batch_size, seq_len, 1), device = self.device), original_time_expand.diff(dim = -1)),
            dim = -1)                                                          # [batch_size, seq_len, resolution]
        timestamp = timestamp.reshape(batch_size, seq_len * resolution)        # [batch_size, seq_len * resolution]

        # Gradient 1: Integral -> time
        event_gradient = torch.autograd.grad(
            outputs=expand_integral,
            inputs=time_expand,
            grad_outputs=torch.ones_like(expand_integral),
            create_graph=True,
        )[0].reshape(batch_size, -1, self.num_events)                          # [batch_size, seq_len * resolution, num_events]
        accumulated_gradient = event_gradient.sum(dim = -1)                    # [batch_size, seq_len * resolution]
        event_gradient = event_gradient.chunk(self.num_events, dim = -1)       # [batch_size, seq_len * resolution] * num_events
        event_intensity = {}
        for idx, item in enumerate(event_gradient):
            event_intensity[f'event_{idx}'] = item                             # [batch_size, seq_len * resolution]

        # Gradient 2: All layer output -> time
        output_storage_gradient = {}
        for idx, item in enumerate(output_storage):
            subgradient = torch.autograd.grad(
            outputs=item,
            inputs=time_expand,
            grad_outputs=torch.ones_like(item),
            create_graph=True,
            )[0].sum(dim = -1).reshape(batch_size, -1)                         # [batch_size, seq_len * resolution]
            output_storage_gradient[f'mlp_{idx}_grad'] = subgradient           # [batch_size, seq_len, resolution, num_events] * (self.mlp.size + 1)
                
        time_expand.requires_grad = False

        result = {
            **{'accumulated_gradient': accumulated_gradient},\
            **output_storage_gradient,\
            **event_intensity,\
            **{"output_mlp_norm_" + str(idx): torch.norm(item, dim = -1).mean(dim = -1).reshape(batch_size, -1) for idx, item in enumerate(output_storage)},\
            **{"output_mlp_mean_" + str(idx): torch.mean(item, dim = -1).mean(dim = -1).reshape(batch_size, -1) for idx, item in enumerate(output_storage)},\
            **{"output_mlp_max_" + str(idx): torch.max(item, dim = -1)[0].mean(dim = -1).reshape(batch_size, -1) for idx, item in enumerate(output_storage)},\
            **{"output_mlp_min_" + str(idx): torch.min(item, dim = -1)[0].mean(dim = -1).reshape(batch_size, -1) for idx, item in enumerate(output_storage)},\
            **{"output_rnn_norm": torch.norm(hidden_expand, dim = -1).mean(dim = -1).reshape(batch_size, -1)},\
            **{"output_rnn_mean": torch.mean(hidden_expand, dim = -1).mean(dim = -1).reshape(batch_size, -1)},\
            **{"output_rnn_max": torch.max(hidden_expand, dim = -1)[0].mean(dim = -1).reshape(batch_size, -1)},\
            **{"output_rnn_min": torch.min(hidden_expand, dim = -1)[0].mean(dim = -1).reshape(batch_size, -1)},\
            **{"accumulate_layer_output": accumulative_layer_output.mean(dim = -1).reshape(batch_size, -1)},\
            **{"final_output": expand_integral.squeeze(-1).reshape(batch_size, -1)}
            }

        return result, timestamp


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