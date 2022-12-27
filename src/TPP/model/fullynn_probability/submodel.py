import torch.nn as nn
import torch
from scipy.stats import spearmanr
import numpy as np
from einops import rearrange, repeat, reduce, pack, unpack
import pandas as pd

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
                 device):
        super(FullyNN, self).__init__()
        self.device = device
        self.num_events = num_events
        self.event_toggle = event_toggle
        self.history_module = history_module.lower()
        self.split_comp_graph = split_comp_graph

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
        # hidden_x initialisation
        # The original implement counts the hidden_x as one of mlp_layers
        nn.init.xavier_uniform_(self.hidden_x)

        self.hidden_time = NonNegLinear(d_intensity, d_intensity, device = self.device)
        self.hidden_p = nn.Linear(d_history, d_intensity, bias = True, device = device)

        self.mlp = nn.ModuleList([
            NonNegLinear(d_intensity, d_intensity, bias = True, device = device) for _ in range(mlp_layers)
        ])

        self.agg = NonNegLinear(d_intensity, 1, bias = True, device = device)
        self.activate = TA[nonlinear]()

        self.non_neg = nn.Softplus()
        self.non_neg_integral = nn.Sigmoid()

    def forward(self, events_history, time_history, time_next, mean, var, mask):
        '''
        Args:
            events_history: [batch_size, seq_len]
            time_history:   [batch_size, seq_len]
            time_next:      [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
            mask:           [batch_size, seq_len]
        '''
        # Input data normalization
        # Don't shift time with mean
        # mean = 0

        time_history = (time_history - mean) / var                             # [batch_size, seq_len]
        time_next = (time_next - mean) / var                                   # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
        time_next_zero = torch.ones_like(time_next) * (-mean / var)            # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]

        if self.event_toggle:
            events_embeddings = self.events(events_history)                    # [batch_size, seq_len, d_history]
            history, history_ps = pack([events_embeddings, time_history], 'b s *')
                                                                               # [batch_size, seq_len, d_history + 1]
        else:
            history = rearrange(time_history, '... -> ... 1')                  # [batch_size, seq_len, 1]
        
        # Reshape hidden output for full connection layers.
        if self.history_module == 'lstm':
            history_output, (_, _) = self.his_encoder(history)                 # [batch_size, seq_len, d_history]
        elif self.history_module == 'transformers':
            history_output = self.his_encoder(events_history, time_history, mask)
                                                                               # [batch_size, seq_len, d_history]

        if self.event_toggle:
            history_output = repeat(history_output, 'b s dh -> b s ne dh', ne = self.num_events)
                                                                               # [batch_size, seq_len, num_events, d_history]
        
        time = rearrange(time_next, '... -> ... 1') * self.non_neg(self.hidden_x)
                                                                               # [batch_size, seq_len, num_events, d_intensity] if we need events else [batch_size, seq_len, d_intensity]
        time_zero = rearrange(time_next_zero, '... -> ... 1') * self.non_neg(self.hidden_x)
                                                                               # [batch_size, seq_len, num_events, d_intensity] if we need events else [batch_size, seq_len, d_intensity]

        hidden_history = self.hidden_p(history_output)                         # [batch_size, seq_len, num_events, d_intensity] if we need events else [batch_size, seq_len, d_intensity]
        
        time = self.hidden_time(time)                                          # [batch_size, seq_len, num_events, d_intensity] if we need events else [batch_size, seq_len, d_intensity]
        time_zero = self.hidden_time(time_zero)                                # [batch_size, seq_len, num_events, d_intensity] if we need events else [batch_size, seq_len, d_intensity]
        
        output = self.activate(time + hidden_history)                          # [batch_size, seq_len, num_events, d_intensity] if we need events else [batch_size, seq_len, d_intensity]
        output_zero = self.activate(time_zero + hidden_history)                # [batch_size, seq_len, num_events, d_intensity] if we need events else [batch_size, seq_len, d_intensity]


        for layer in self.mlp:
            output = layer(output)                                             # [batch_size, seq_len, num_events, d_intensity] if we need events else [batch_size, seq_len, d_intensity]
            output = self.activate(output)                                     # [batch_size, seq_len, num_events, d_intensity] if we need events else [batch_size, seq_len, d_intensity]
            
            output_zero = layer(output_zero)                                   # [batch_size, seq_len, num_events, d_intensity] if we need events else [batch_size, seq_len, d_intensity]
            output_zero = self.activate(output_zero)                           # [batch_size, seq_len, num_events, d_intensity] if we need events else [batch_size, seq_len, d_intensity]


        integral = self.non_neg_integral(-self.agg(output))                    # [batch_size, seq_len, num_events, 1] if we need events else [batch_size, seq_len, 1]
        integral_zero = self.non_neg_integral(-self.agg(output_zero))          # [batch_size, seq_len, num_events, 1] if we need events else [batch_size, seq_len, 1]

        integral = rearrange(integral, '... 1 -> ...')                         # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len, 1]
        integral_zero = reduce(integral_zero, '... ne 1 -> ... ()', 'sum')     # [batch_size, seq_len, 1]

        return integral / integral_zero

    def probability(self, events_history, time_history, time_next, resolution, mean, var, mask, sum = True):
        '''
        Intensity integral & intensity function prober. Perhaps, we can support intensity integral as well.
        Args:
        events_history:[batch_size, seq_len]
        time_history:  [batch_size, seq_len]
        time_next:     [batch_size, seq_len]
        resolution:    int
        '''        
        time_multiplier = torch.linspace(0, 1, resolution, device = self.device)
                                                                               # [resolution]
        original_time_expand = time_multiplier * rearrange(time_next, '... -> ... 1')
                                                                               # [batch_size, seq_len, resolution]
        # Don't shift time with mean
        # mean = 0

        time_history = (time_history - mean) / var                             # [batch_size, seq_len]

        time_expand = original_time_expand.clone()                             # [batch_size, seq_len, resolution]
        if self.event_toggle:
            time_expand = repeat(original_time_expand, 'b s r -> b s r ne', ne = self.num_events)
                                                                               # [batch_size, seq_len, resolution, num_events]

        if self.event_toggle:
            events_embeddings = self.events(events_history)                    # [batch_size, seq_len, d_history]
            history, history_ps = pack([events_embeddings, time_history], 'b s *')
                                                                               # [batch_size, seq_len, d_history + 1]
        else:
            history = rearrange(time_history, '... -> ... 1')                  # [batch_size, seq_len, 1]
        
        if self.history_module == 'lstm':
            output, (_, _) = self.his_encoder(history)                         # [batch_size, seq_len, d_history]
        elif self.history_module == 'transformers':
            output = self.his_encoder(events_history, time_history, mask.unsqueeze(dim = -1))
                                                                               # [batch_size, seq_len, d_history]
        output = self.hidden_p(output)                                         # [batch_size, seq_len, d_intensity]

        if self.event_toggle:
            history_expand = repeat(output, 'b s di -> b s r ne di', r = resolution, ne = self.num_events)
                                                                               # [batch_size, seq_len, resolution, num_events, d_intensity]
        else:
            history_expand = repeat(output, 'b s di -> b s r di', r = resolution)
                                                                               # [batch_size, seq_len, resolution, d_intensity]

        time_expand.requires_grad = True
        time_expand_norm = (time_expand - mean) / var                          # [batch_size, seq_len, resolution, num_events] is we need events else [batch_size, seq_len, resolution]

        emb_time_expand = rearrange(time_expand_norm, '... -> ... 1') * self.non_neg(self.hidden_x)
                                                                               # [batch_size, seq_len, resolution, num_events, d_intensity] is we need events else [batch_size, seq_len, resolution, d_intensity]

        emb_time_expand = self.hidden_time(emb_time_expand)                    # [batch_size, seq_len, resolution, num_events, d_intensity] if we need events else [batch_size, seq_len, resolution, d_intensity]
        output = self.activate(emb_time_expand + history_expand)               # [batch_size, seq_len, resolution, num_events, d_intensity] if we need events else [batch_size, seq_len, resolution, d_intensity]

        for layer in self.mlp:
            output = layer(output)                                             # [batch_size, seq_len, resolution, num_events, d_intensity] if we need events else [batch_size, seq_len, resolution, d_intensity]
            output = self.activate(output)                                     # [batch_size, seq_len, resolution, num_events, d_intensity] if we need events else [batch_size, seq_len, resolution, d_intensity]

        expand_integral = self.non_neg_integral(-self.agg(output))             # [batch_size, seq_len, resolution, num_events, 1] if we need events else [batch_size, seq_len, resolution, 1]
        
        if self.event_toggle:
            integral_from_zero_to_inf = expand_integral[:, :, 0, :, :].detach()# [batch_size, seq_len, num_events, 1]
            integral_sum = reduce(integral_from_zero_to_inf, 'b s ne 1 -> b s ()', 'sum')
                                                                               # [batch_size, seq_len, 1]
            integral_sum = rearrange(integral_sum, 'b s 1 -> b s 1 1 1')       # [batch_size, seq_len, 1, 1, 1]
            expand_integral = expand_integral / integral_sum                   # [batch_size, seq_len, resolution, num_events, 1]
        else:
            integral_from_zero_to_inf = expand_integral[:, :, 0, :].detach()   # [batch_size, seq_len, 1]
            integral_sum = rearrange(integral_from_zero_to_inf, 'b s 1 -> b s 1 1')
                                                                               # [batch_size, seq_len, 1, 1]
            expand_integral = expand_integral / integral_sum                   # [batch_size, seq_len, resolution, 1]

        expand_probability = - torch.autograd.grad(
            outputs=expand_integral,
            inputs=time_expand,
            grad_outputs=torch.ones_like(expand_integral),
        )[0]                                                                   # [batch_size, seq_len, resolution, num_events] if we need events else [batch_size, seq_len, resolution]
        time_expand.requires_grad = False

        expand_probability = expand_probability.detach()                       # [batch_size, seq_len, resolution, num_events] if we need events else [batch_size, seq_len, resolution]

        if self.event_toggle:
            expand_probability = rearrange(expand_probability, 'b s r ne -> b (s r) ne')
                                                                               # [batch_size, seq_len * resolution, num_events]

            if sum:
                expand_probability = expand_probability.sum(dim = -1)          # [batch_size, seq_len * resolution]
        else:
            expand_probability = rearrange(expand_probability, 'b s r -> b (s r)')
                                                                               # [batch_size, seq_len * resolution]

        '''
        Restore the original timestamp
        '''
        batch_size, seq_len = history_expand.shape[0], history_expand.shape[1]
        dummy_inception = torch.zeros((batch_size, seq_len, 1), device = self.device)
        timestamp, timestamp_ps = pack(
            [dummy_inception, original_time_expand.diff(dim = -1)],
            'b s *')                                                           # [batch_size, seq_len, resolution]
        timestamp = rearrange(timestamp, 'b s r -> b (s r)')                   # [batch_size, seq_len * resolution]

        return expand_probability, timestamp

    def model_probe_function(self, events_history, time_history, time_next, resolution, mean, var, mask):
        '''
        We use this function to dive into the fullynn and find the reason of abrupt gradient drop around 0
        Args:
        time_history: [batch_size, seq_len]
        time_next:    [batch_size, seq_len]
        resolution:   int
        '''
        time_multiplier = torch.linspace(0, 1, resolution, device = self.device)
                                                                               # [resolution]
        original_time_expand = time_multiplier * rearrange(time_next, '... -> ... 1')
                                                                               # [batch_size, seq_len, resolution]

        # Don't shift time with mean
        # mean = 0
        time_history = (time_history - mean) / var                             # [batch_size, seq_len]

        time_expand = original_time_expand.clone()                             # [batch_size, seq_len, resolution]
        if self.event_toggle:
            time_expand = repeat(original_time_expand, 'b s r -> b s r ne', ne = self.num_events)
                                                                               # [batch_size, seq_len, resolution, num_events]

        if self.event_toggle:
            events_embeddings = self.events(events_history)                    # [batch_size, seq_len, d_history]
            history, history_ps = pack(
                [events_embeddings, time_history],
                'b s *'
            )                                                                  # [batch_size, seq_len, d_history + 1]
        else:
            history = rearrange(time_history, '... -> ... 1')                  # [batch_size, seq_len, 1]

        if self.history_module == 'lstm':
            output, (_, _) = self.his_encoder(history)                         # [batch_size, seq_len, d_history]
        elif self.history_module == 'transformers':
            output = self.his_encoder(events_history, time_history, mask.unsqueeze(dim = -1))
                                                                               # [batch_size, seq_len, d_history]
        history = self.hidden_p(output)                                        # [batch_size, seq_len, d_intensity]

        if self.event_toggle:
            history_expand = repeat(history, 'b s di -> b s r ne di', r = resolution, ne = self.num_events)
                                                                               # [batch_size, seq_len, resolution, num_events, d_intensity]
        else:
            history_expand = repeat(history, 'b s di -> b s r di', r = resolution)
                                                                               # [batch_size, seq_len, resolution, d_intensity]

        time_expand.requires_grad = True      
        time_expand_norm = (time_expand - mean) / var                          # [batch_size, seq_len, resolution, num_events] if we need events else [batch_size, seq_len, resolution]

        emb_time_expand = rearrange(time_expand_norm, '... -> ... 1') * self.non_neg(self.hidden_x)
                                                                               # [batch_size, seq_len, resolution, num_events, d_intensity] is we need events else [batch_size, seq_len, resolution, d_intensity]

        emb_time_expand = self.hidden_time(emb_time_expand)                    # [batch_size, seq_len, resolution, num_events, d_intensity] if we need events else [batch_size, seq_len, resolution, d_intensity]
        output = self.activate(emb_time_expand + history_expand)               # [batch_size, seq_len, resolution, num_events, d_intensity] if we need events else [batch_size, seq_len, resolution, d_intensity]
        output_storage = [output]                                              # [batch_size, seq_len, resolution, num_events, d_intensity] if we need events else [batch_size, seq_len, resolution, d_intensity]

        for layer in self.mlp:
            output = layer(output)                                             # [batch_size, seq_len, resolution, num_events, d_intensity] if we need events else [batch_size, seq_len, resolution, d_intensity]
            output = self.activate(output)                                     # [batch_size, seq_len, resolution, num_events, d_intensity] if we need events else [batch_size, seq_len, resolution, d_intensity]
            output_storage.append(output)                                      # [batch_size, seq_len, resolution, num_events, d_intensity] * (self.mlp_size + 1) if we need events else [batch_size, seq_len, resolution, d_intensity] * (self.mlp_size + 1)
        
        accumulative_layer_output = rearrange(self.agg(output), '... 1 -> ...')# [batch_size, seq_len, resolution, num_events] if we need events else [batch_size, seq_len, resolution]
        expand_integral = self.non_neg_integral(-accumulative_layer_output)    # [batch_size, seq_len, resolution, num_events] if we need events else [batch_size, seq_len, resolution]

        if self.event_toggle:
            integral_from_zero_to_inf = expand_integral[:, :, 0, :].detach()   # [batch_size, seq_len, num_events]
            integral_sum = reduce(integral_from_zero_to_inf, 'b s ne -> b s ()', 'sum')
                                                                               # [batch_size, seq_len, 1]
            integral_sum = rearrange(integral_sum, 'b s 1 -> b s 1 1')         # [batch_size, seq_len, 1, 1]
            expand_integral = expand_integral / integral_sum                   # [batch_size, seq_len, resolution, num_events]
        else:
            integral_from_zero_to_inf = expand_integral[:, :, 0].detach()      # [batch_size, seq_len]
            integral_sum = rearrange(integral_from_zero_to_inf, 'b s -> b s 1')# [batch_size, seq_len, 1, 1]
            expand_integral = expand_integral / integral_sum                   # [batch_size, seq_len, resolution]

        # Gradient 1: Integral -> time
        events_gradient = - torch.autograd.grad(
            outputs=expand_integral,
            inputs=time_expand,
            grad_outputs=torch.ones_like(expand_integral),
            retain_graph=True
        )[0]                                                                   # [batch_size, seq_len, resolution, num_events] if we need events else [batch_size, seq_len, resolution]

        # Gradient 2: All layer output -> time
        output_storage_gradient = {}
        for idx, item in enumerate(output_storage):
            subgradient = torch.autograd.grad(
                outputs=item,
                inputs=time_expand,
                grad_outputs=torch.ones_like(item),
                retain_graph=True
            )[0]                                                               # [batch_size, seq_len, resolution, num_events] if we need events else [batch_size, seq_len, resolution]
            subgradient = reduce(subgradient, 'b s r ... -> b (s r)', 'sum')
                                                                               # [batch_size, seq_len * resolution]
            output_storage_gradient[f'mlp_{idx}_grad'] = subgradient.detach()  # [batch_size, seq_len * resolution] * (self.mlp.size + 1)
                
        time_expand.requires_grad = False

        accumulated_integral = reduce(expand_integral, 'b s r ... -> b (s r)', 'sum')
                                                                               # [batch_size, seq_len * resolution]
        accumulated_gradient = reduce(events_gradient, 'b s r ... -> b (s r)', 'sum')
                                                                               # [batch_size, seq_len * resolution]

        # Timestamp part
        batch_size, seq_len = history_expand.shape[0], history_expand.shape[1]
        zero_inception = torch.zeros((batch_size, seq_len, 1), device = self.device)
        timestamp, timstamp_ps = pack(
            [zero_inception, original_time_expand.diff(dim = -1)],
            'b s *')                                                           # [batch_size, seq_len, resolution]
        timestamp = rearrange(timestamp, 'b s r -> b (s r)')                   # [batch_size, seq_len * resolution]

        if self.event_toggle:
            probability_for_each_event = events_gradient.chunk(self.num_events, dim = -1)
                                                                               # [batch_size, seq_len, resolution] * num_events
            probability_integral_for_each_event = expand_integral.chunk(self.num_events, dim = -1)
                                                                               # [batch_size, seq_len, resolution] * num_events

            events_probability = {}
            events_probability_integral = {}
            for idx, (probability, probability_integral) in enumerate(zip(probability_for_each_event, probability_integral_for_each_event)):
                events_probability[f'event_probability_{idx}'] = rearrange(probability, 'b s r 1 -> b (s r)')
                                                                               # [batch_size, seq_len * resolution]
                events_probability_integral[f'event_probability_integral_{idx}'] = rearrange(probability_integral, 'b s r 1 -> b (s r)')
                                                                               # [batch_size, seq_len * resolution]


            # additional plot, measure the spearman correlation across available events.
            expand_probability = rearrange(events_gradient.cpu(), 'b s r ne -> b (s r) ne')
                                                                               # [batch_size, seq_len * resolution, num_event]
            
            additional_plot = []
            for idx, (expand_probability_per_seq, mask_per_seq, time_next_per_seq) in enumerate(zip(expand_probability, mask, time_next)):
                additional_plot_per_seq = {
                    'heatmap': []
                }

                seq_len = mask_per_seq.sum()
                heatmap_data = {}
                # rho: spearman coefficient
                heatmap_data['spearman'] = spearmanr(expand_probability_per_seq[:seq_len * resolution])[0]
                if self.num_events == 2:
                    heatmap_data['spearman'] = np.array([[1, heatmap_data['spearman']], [heatmap_data['spearman'], 1]])

                # r: pearson coefficient
                heatmap_data['pearson'] = np.corrcoef(expand_probability_per_seq[:seq_len * resolution], rowvar = False)
                # L^1 metric
                heatmap_data['L1'] = L1_distance(expand_probability_per_seq[:seq_len * resolution], 
                                                 resolution = resolution, num_events = self.num_events,
                                                 time_next = time_next_per_seq[:seq_len])

                # Transfer the result matrices into DataFrames.
                def matrix_to_pd(matrix, index_name, column_name, value_name):
                    index, column = matrix.shape

                    # The index and column list
                    index_list = [ele for ele in range(index) for _ in range(column)]
                    column_list = list(range(column)) * index

                    df = pd.DataFrame.from_dict({
                        index_name: index_list,
                        column_name: column_list,
                        value_name: matrix.flatten()
                    })

                    df = df.pivot(index = index_name, columns = column_name, values = value_name)

                    return df
                
                heatmap_data['L1'] = \
                    matrix_to_pd(heatmap_data['L1'], index_name = 'Event type', column_name = 'Event type ', value_name = 'L1')
                heatmap_data['pearson'] = \
                    matrix_to_pd(heatmap_data['pearson'], index_name = 'Event type', column_name = 'Event type ', value_name = 'pearson')
                heatmap_data['spearman'] = \
                    matrix_to_pd(heatmap_data['spearman'], index_name = 'Event type', column_name = 'Event type ', value_name = 'spearman')

                # add plots
                for key, value in heatmap_data.items():
                    idx = 0
                    additional_plot_per_seq['heatmap'].append(
                    [
                        f'{key}_{idx}',
                        {
                            'data': value,
                            'cmap': "YlGnBu",
                            'vmin': 0,
                            'vmax': max(5, np.max(value.values)),
                            'annot': True
                        }
                    ])
                    idx += 1
                
                additional_plot.append(additional_plot_per_seq)
    
            result = {
                **{'accumulated_gradient': accumulated_gradient},\
                **output_storage_gradient,\
                **events_probability,\
                **events_probability_integral,\
                **{"final_output": accumulated_integral},
                }
        else:
            result = {
                **{'accumulated_gradient': accumulated_gradient},\
                **output_storage_gradient,\
                **{"final_output": accumulated_integral},
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
    3. num_events: int
                   the number of event types
    4. time_next:  [seq_len, num_events]
                   the length of all intervals with interpolations.
    '''

    input = rearrange(input, '(s r) ne -> ne s r', r = resolution)             # [num_events, seq_len, resolution]
    intensity_1 = repeat(input, 'ne s r -> ne new_d s r', new_d = num_events)  # [num_events, num_events, seq_len, resolution]
    intensity_2 = repeat(input, 'ne s r -> new_d ne s r', new_d = num_events)  # [num_events, num_events, seq_len, resolution]
    delta_intensity = np.abs(intensity_1 - intensity_2)                        # [num_events, num_events, seq_len, resolution]

    gap = time_next.detach().cpu().numpy() / (resolution - 1)                  # [seq_len]
    gap = rearrange(gap, 's -> 1 1 s 1')                                       # [num_events, num_events, seq_len, 1]

    L1 = reduce((delta_intensity * gap)[:, :, :, :-1], 'ne1 ne2 s r -> ne1 ne2', 'sum')
                                                                               # [num_events, num_events]
    # round off the value smaller than 1e-6
    L1[L1 < 1e-6] = 0

    return L1