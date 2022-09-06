import torch.nn as nn
import torch

from .nonneg import NonNegLinear
from .activate import *
from .transformers import TransEncoder, TransDecoder


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

class AttnCM(nn.Module):
    '''
    This is our implementation of Omi's paper: Fully Neural Network based Model for General Temporal Point Processes
    Hope it can work properly.

    Currently, normalization is disabled.
    Update: 2022-01-19: Now you can use data normalization via synthetic dataloader.

    Following Babylon's paper, we would check the performance of FullyNN with integral offsets.
    '''

    def __init__(self, d_history, d_intensity, num_events, dropout, history_module, history_module_layers,
                 integral_module_layers, mlp_layers, nonlinear, event_toggle, n_head, wq_nonneg, wk_nonneg, wv_nonneg, device):
        super(AttnCM, self).__init__()
        self.device = device
        self.num_events = num_events
        self.event_toggle = event_toggle
        self.history_module = history_module.lower()

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

        self.intensity_integral_solver = TransDecoder(num_types = num_events, d_input = d_history, n_head = n_head, d_hidden = 4 * d_history,  
                                                      d_qk = d_history, d_v = d_history, integral_module_layers = integral_module_layers,
                                                      dropout = dropout, event_toggle = event_toggle, device = device)


    def forward(self, events_history, time_history, time_next, mean, var, mask_history, mask_next):
        '''
        Args:
            events_history: [batch_size, seq_len]
            time_history:   [batch_size, seq_len, 1]
            time_next:      [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len, 1]
            mask_history:   [batch_size, seq_len]
            mask_next:      [batch_size, seq_len]
        '''
        # Input data normalization
        time_history = (time_history - mean) / var                             # [batch_size, seq_len, 1]
        time_next = (time_next - mean) / var                                   # [batch_size, seq_len, num_events]
        
        if self.event_toggle:
            emb_event_history = self.events(events_history)                    # [batch_size, seq_len, d_history]

        if self.event_toggle:
            history = torch.cat(
                (emb_event_history, time_history), dim = -1
            )
        else:
            history = time_history                                             # [batch_size, seq_len, d_history + 1] if we need events else [batch_size, seq_len, 1]
        
        # Reshape hidden output for full connection layers.
        if self.history_module == 'lstm':
            output, (_, _) = self.his_encoder(history)                         # [batch_size, seq_len, d_history]
        elif self.history_module == 'transformers':
            output = self.his_encoder(emb_event_history if self.event_toggle else None, time_history, mask_history.unsqueeze(dim = -1))
                                                                               # [batch_size, seq_len, d_history]

        if self.event_toggle:
            output = output.unsqueeze(-2).repeat(1, 1, self.num_events, 1)     # [batch_size, seq_len, num_events, d_history] if we need events else [batch_size, seq_len, d_history]


        integral_original = self.intensity_integral_solver(history = output, time_next = time_next, mask_next = mask_next)
                                                                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
        integral_zero = self.intensity_integral_solver(history = output, time_next = torch.zeros_like(time_next, device = self.device), mask_next = mask_next)
                                                                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
        
        return integral_original - integral_zero

    def integral_intensity(self, events_history, time_history, time_next, resolution, mean, var, mask):
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
        if self.event_toggle:
            time_next = time_next.repeat(1, 1, self.num_events)                # [batch_size, seq_len, num_events]
        time_history = (time_history - mean) / var                             # [batch_size, seq_len, 1]
        time_next = (time_next - mean) / var                                   # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len, 1]

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
        history_expand = hidden.unsqueeze(-2).repeat(1, 1, resolution, 1)      # [batch_size, seq_len, resolution, d_history]
        batch_size, seq_len = history_expand.shape[0], history_expand.shape[1]
        if self.event_toggle:
            history_expand = history_expand.unsqueeze(-2).repeat(1, 1, 1, self.num_events, 1)
                                                                               # [batch_size, seq_len, resolution, num_events, d_history]

        time_expand = time_multiplier.reshape(1, 1, resolution, 1) * time_next.unsqueeze(-2)
                                                                               # [batch_size, seq_len, resolution, num_events] if we need events else [batch_size, seq_len, resolution, 1]
        time_expand.requires_grad = True
        if self.event_toggle:
            emb_time_expand = time_expand.unsqueeze(-1) * self.non_neg(self.hidden_x)
                                                                               # [batch_size, seq_len, resolution, num_events, d_intensity]
        else:
            emb_time_expand = time_expand * self.non_neg(self.hidden_x)        # [batch_size, seq_len, resolution, d_intensity]

        emb_time_expand = self.hidden_time(emb_time_expand)                    # [batch_size, seq_len, resolution, num_events, d_intensity] if we need events else [batch_size, seq_len, resolution, d_intensity]
        output = self.activate(emb_time_expand + history_expand)               # [batch_size, seq_len, resolution, num_events, d_intensity] if we need events else [batch_size, seq_len, resolution, d_intensity]

        for layer in self.mlp:
            output = layer(output)                                             # [batch_size, seq_len, resolution, num_events, d_intensity] if we need events else [batch_size, seq_len, resolution, d_intensity]
            output = self.activate(output)                                     # [batch_size, seq_len, resolution, num_events, d_intensity] if we need events else [batch_size, seq_len, resolution, d_intensity]

        expand_integral = self.non_neg(self.agg(output)).squeeze(-1)           # [batch_size, seq_len, resolution, num_events] if we need events else [batch_size, seq_len, resolution]
        
        expand_intensity = torch.autograd.grad(
            outputs=expand_integral,
            inputs=time_expand,
            grad_outputs=torch.ones_like(expand_integral),
            create_graph=True,
        )[0]                                                                   # [batch_size, seq_len, resolution, num_events] if we need events else [batch_size, seq_len, resolution, 1]
        if self.event_toggle:
            expand_integral = expand_integral.reshape(batch_size, seq_len * resolution, -1)
                                                                               # [batch_size, seq_len * resolution, num_events]
            expand_intensity = expand_intensity.reshape(batch_size, seq_len * resolution, -1)
                                                                               # [batch_size, seq_len * resolution, num_events]
        else:
            expand_intensity = expand_intensity.squeeze(-1).reshape(batch_size, seq_len * resolution)
                                                                               # [batch_size, seq_len * resolution]
            expand_integral = expand_integral.reshape(batch_size, seq_len * resolution)
                                                                               # [batch_size, seq_len * resolution]
        time_expand.requires_grad = False

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
        if self.event_toggle:
            time_next = time_next.repeat(1, 1, self.num_events)                # [batch_size, seq_len, num_events]        
        time_history = (time_history - mean) / var                             # [batch_size, seq_len, 1]
        time_next = (time_next - mean) / var                                   # [batch_size, seq_len, num_events]

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
        history_expand = hidden.unsqueeze(-2).repeat(1, 1, resolution, 1)      # [batch_size, seq_len, resolution, d_history]
        batch_size, seq_len = history_expand.shape[0], history_expand.shape[1]
        if self.event_toggle:
            history_expand = history_expand.unsqueeze(-2).repeat(1, 1, 1, self.num_events, 1)
                                                                               # [batch_size, seq_len, resolution, num_events, d_history]

        time_expand = time_multiplier.reshape(1, 1, resolution, 1) * time_next.unsqueeze(-2)
                                                                               # [batch_size, seq_len, resolution, num_events] if we need events else [batch_size, seq_len, resolution, 1]
        time_expand.requires_grad = True
        if self.event_toggle:
            emb_time_expand = time_expand.unsqueeze(-1) * self.non_neg(self.hidden_x)
                                                                               # [batch_size, seq_len, resolution, num_events, d_intensity]
        else:
            emb_time_expand = time_expand * self.non_neg(self.hidden_x)        # [batch_size, seq_len, resolution, d_intensity]

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
            create_graph=True,
        )[0]                                                                   # [batch_size, seq_len, resolution, num_events] if we need events else [batch_size, seq_len, resolution, 1]

        # Gradient 2: All layer output -> time
        output_storage_gradient = {}
        for idx, item in enumerate(output_storage):
            subgradient = torch.autograd.grad(
                outputs=item,
                inputs=time_expand,
                grad_outputs=torch.ones_like(item),
                create_graph=True,
            )[0]                                                               # [batch_size, seq_len, resolution, num_events] if we need events else [batch_size, seq_len, resolution, 1]
            subgradient = subgradient.sum(dim = -1).reshape(batch_size, -1)    # [batch_size, seq_len * resolution]
            output_storage_gradient[f'mlp_{idx}_grad'] = subgradient           # [batch_size, seq_len * resolution] * (self.mlp.size + 1)
                
        time_expand.requires_grad = False

        accumulated_gradient = event_gradient.sum(dim = -1).reshape(batch_size, -1)
                                                                               # [batch_size, seq_len * resolution]
        if self.event_toggle:
            event_gradient = event_gradient.chunk(self.num_events, dim = -1)   # [batch_size, seq_len, resolution] * num_events
            event_intensity = {}
            for idx, item in enumerate(event_gradient):
                event_intensity[f'event_{idx}'] = item.reshape(batch_size, -1) # [batch_size, seq_len * resolution]
    
            result = {
                **{'accumulated_gradient': accumulated_gradient},\
                **output_storage_gradient,\
                **event_intensity,\
                **{"final_output": expand_integral.sum(dim = -1).reshape(batch_size, -1)}
                }
        else:
            result = {
                **{'accumulated_gradient': accumulated_gradient},\
                **output_storage_gradient,\
                **{"final_output": expand_integral.reshape(batch_size, -1)}
                }

        return result, timestamp