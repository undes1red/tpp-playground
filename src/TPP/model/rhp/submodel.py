import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce, pack
import numpy as np
from scipy.stats import spearmanr

from src.TPP.model.utils import L1_distance_across_events


class RHPModule(nn.Module):
    def __init__(self, device, num_events, history_module_name, d_mark_embedding, d_input, d_hidden, \
                 history_encoder_layers, dropout, monte_carlo_resolution):
        '''
        A CTLSTM implementation, based on existing SAHP codes.
        '''
        super(RHPModule, self).__init__()
        self.num_events = num_events
        self.device = device
        self.resolution = monte_carlo_resolution

        self.gelu = nn.GELU()

        self.start_layer = nn.Sequential(
            nn.Linear(d_input, d_input, bias = True, device = self.device),
            self.gelu
        )

        self.converge_layer = nn.Sequential(
            nn.Linear(d_input, d_input, bias = True, device = self.device),
            self.gelu
        )

        self.decay_layer = nn.Sequential(
            nn.Linear(d_input, d_input, bias = True, device = self.device),
            nn.Softplus(beta = 10.0)
        )

        # This layer translates decayed hidden states into intensity function values.
        self.intensity_layer = nn.Sequential(
            nn.Linear(d_input, self.num_events, bias = True, device = self.device),
            nn.Softplus(beta = 1.)
        )

        # mark embedding layer.
        self.events_embedding = nn.Embedding(num_events + 1, d_mark_embedding, padding_idx = num_events, device = device)

        # History encoder.
        try:
            self.history_encoder = getattr(nn, history_module_name)(device = self.device, \
                                           input_size = d_mark_embedding + 1, hidden_size = d_hidden, \
                                           dropout = dropout, num_layers = history_encoder_layers, batch_first = True)
        except Exception as e:
            print(e)
            raise Exception("Unrecognized history module name!")
        
        self.history_mapper = nn.Linear(d_hidden, d_input, device = self.device)


    def state_decay(self, mu, eta, gamma, duration_t):
        '''
        mu, eta, gamma: shape: [batch_size, seq_len, d_hidden]
        dutation_t:     shape: [batch_size, seq_len, (resolution, num_events)]
        '''
        if len(duration_t.shape) == 3:
            # add additional dimension to mu, eta, and gamma.
            mu = rearrange(mu, 'b s d_i -> b s 1 d_i')                         # [batch_size, seq_len, 1, d_input]
            eta = rearrange(eta, 'b s d_i -> b s 1 d_i')                       # [batch_size, seq_len, 1, d_input]
            gamma = rearrange(gamma, 'b s d_i -> b s 1 d_i')                   # [batch_size, seq_len, 1, d_input]
        elif len(duration_t.shape) == 4:
            # add additional dimension to mu, eta, and gamma.
            mu = rearrange(mu, 'b s d_i -> b s 1 1 d_i')                       # [batch_size, seq_len, 1, 1, d_input]
            eta = rearrange(eta, 'b s d_i -> b s 1 1 d_i')                     # [batch_size, seq_len, 1, 1, d_input]
            gamma = rearrange(gamma, 'b s d_i -> b s 1 1 d_i')                 # [batch_size, seq_len, 1, 1, d_input]

        duration_t = duration_t.unsqueeze(dim = -1)                            # [batch_size, seq_len, (resolution, num_events), 1]
        cell_t = torch.tanh(mu + (eta - mu) * torch.exp(-gamma * duration_t))  # [batch_size, seq_len, (resolution, num_events), d_input]
        return cell_t

    
    def monte_carlo_integration_estimator(self, time_history, time_next, events_history):
        events_embeddings = self.events_embedding(events_history)              # [batch_size, seq_len, d_mark_embedding]
        history, history_ps = pack([events_embeddings, time_history], 'b s *') # [batch_size, seq_len, d_mark_embedding + 1]

        history, (_, _) = self.history_encoder(history)                        # [batch_size, seq_len, d_hidden]
        history = self.history_mapper(history)                                 # [batch_size, seq_len, d_input]

        eta = self.start_layer(history)                                        # [batch_size, seq_len, d_input]
        mu = self.converge_layer(history)                                      # [batch_size, seq_len, d_input]
        gamma = self.decay_layer(history)                                      # [batch_size, seq_len, d_input]

        time_multiplier = torch.linspace(0, 1, self.resolution, device = self.device)
        expanded_time = time_next.unsqueeze(dim = -1) * time_multiplier        # [batch_size, seq_len, resolution]
        expanded_hidden_state_at_t = self.state_decay(mu = mu, eta = eta, gamma = gamma, duration_t = expanded_time)
                                                                               # [batch_size, seq_len, resolution, d_input]
        expanded_intensity_all_events = self.intensity_layer(expanded_hidden_state_at_t)
                                                                               # [batch_size, seq_len, resolution, num_events]
        expanded_intensity_all_events_monte_carlo, \
        expanded_intensity_all_events_monte_carlo_ps = pack(
            (expanded_intensity_all_events[:, :, 0, :], 
             expanded_intensity_all_events[:, :, :-1, :]),
             'b s * ne'
        )                                                                      # [batch_size, seq_len, resolution, num_events]
        timestamp_monte_carlo, timestamp_monte_carlo_ps = pack(
            (torch.zeros_like(time_next), expanded_time.diff(dim = -1)),
            'b s *'
        )                                                                      # [batch_size, seq_len, resolution]
        integral_all_events = (expanded_intensity_all_events_monte_carlo * timestamp_monte_carlo.unsqueeze(dim = -1)).sum(dim = -2)
                                                                               # [batch_size, seq_len, num_events]
        
        return integral_all_events


    def forward(self, time_history, time_next, events_history, mask_history):
        events_embeddings = self.events_embedding(events_history)              # [batch_size, seq_len, d_mark_embedding]
        history, history_ps = pack([events_embeddings, time_history], 'b s *') # [batch_size, seq_len, d_mark_embedding + 1]

        history, (_, _) = self.history_encoder(history)                        # [batch_size, seq_len, d_hidden]
        history = self.history_mapper(history)                                 # [batch_size, seq_len, d_input]

        eta = self.start_layer(history)                                        # [batch_size, seq_len, d_input]
        mu = self.converge_layer(history)                                      # [batch_size, seq_len, d_input]
        gamma = self.decay_layer(history)                                      # [batch_size, seq_len, d_input]

        hidden_state_at_t = self.state_decay(mu = mu, eta = eta, gamma = gamma, duration_t = time_next)
                                                                               # [batch_size, seq_len, d_input]
        intensity_all_events = self.intensity_layer(hidden_state_at_t)         # [batch_size, seq_len, num_events]

        integral_all_events = self.monte_carlo_integration_estimator(time_history = time_history,\
                                                                     time_next = time_next, \
                                                                     events_history = events_history)
                                                                               # [batch_size, seq_len, num_events]

        return integral_all_events, intensity_all_events


    def integral_intensity_time_next_2d(self, events_history, time_history, time_next, mask_history, resolution, mean, var):
        events_embeddings = self.events_embedding(events_history)              # [batch_size, seq_len, d_mark_embedding]
        history, history_ps = pack([events_embeddings, time_history], 'b s *') # [batch_size, seq_len, d_mark_embedding + 1]
        
        history, (_, _) = self.history_encoder(history)                        # [batch_size, seq_len, d_hidden]
        history = self.history_mapper(history)                                 # [batch_size, seq_len, d_input]


        eta = self.start_layer(history)                                        # [batch_size, seq_len, d_input]
        mu = self.converge_layer(history)                                      # [batch_size, seq_len, d_input]
        gamma = self.decay_layer(history)                                      # [batch_size, seq_len, d_input]

        time_multiplier = torch.linspace(0, 1, resolution, device = self.device)
        expanded_time = time_next.unsqueeze(dim = -1) * time_multiplier        # [batch_size, seq_len, resolution]
        expanded_hidden_state_at_t = self.state_decay(mu = mu, eta = eta, gamma = gamma, duration_t = expanded_time)
                                                                               # [batch_size, seq_len, resolution, d_input]

        expanded_intensity_all_events = self.intensity_layer(expanded_hidden_state_at_t)
                                                                               # [batch_size, seq_len, resolution, num_events]

        expanded_intensity_all_events_monte_carlo, \
        expanded_intensity_all_events_monte_carlo_ps = pack(
            (expanded_intensity_all_events[:, :, 0, :], 
             expanded_intensity_all_events[:, :, :-1, :]),
             'b s * ne'
        )                                                                      # [batch_size, seq_len, resolution, num_events]

        # Obtain timestamp
        timestamp, timestamp_ps = pack(
            (torch.zeros_like(time_next), expanded_time.diff(dim = -1)),
            'b s *'
        )                                                                      # [batch_size, seq_len, resolution]
        expanded_integral_all_events = (expanded_intensity_all_events_monte_carlo * timestamp.unsqueeze(dim = -1)).cumsum(dim = -2)
                                                                               # [batch_size, seq_len, resolution, num_events]

        return expanded_integral_all_events, expanded_intensity_all_events, timestamp


    def integral_intensity_time_next_3d(self, events_history, time_history, time_next, mask_history, resolution, mean, var):
        events_embeddings = self.events_embedding(events_history)              # [batch_size, seq_len, d_mark_embedding]
        history, history_ps = pack([events_embeddings, time_history], 'b s *') # [batch_size, seq_len, d_mark_embedding + 1]
        
        history, (_, _) = self.history_encoder(history)                        # [batch_size, seq_len, d_hidden]
        history = self.history_mapper(history)                                 # [batch_size, seq_len, d_input]

        eta = self.start_layer(history)                                        # [batch_size, seq_len, d_input]
        mu = self.converge_layer(history)                                      # [batch_size, seq_len, d_input]
        gamma = self.decay_layer(history)                                      # [batch_size, seq_len, d_input]

        time_multiplier = torch.linspace(0, 1, resolution, device = self.device)
        expanded_time = time_next.unsqueeze(dim = -1) * time_multiplier        # [batch_size, seq_len, num_events, resolution]
        expanded_hidden_state_at_t = self.state_decay(mu = mu, eta = eta, gamma = gamma, duration_t = expanded_time)
                                                                               # [batch_size, seq_len, num_events, resolution, d_input]

        expanded_intensity_all_events = self.intensity_layer(expanded_hidden_state_at_t)
                                                                               # [batch_size, seq_len, num_events, resolution, num_events]

        expanded_intensity_all_events_monte_carlo, \
        expanded_intensity_all_events_monte_carlo_ps = pack(
            (expanded_intensity_all_events[:, :, :, 0, :], 
             expanded_intensity_all_events[:, :, :, :-1, :]),
             'b s ne * ne1'
        )                                                                      # [batch_size, seq_len, num_events, resolution, num_events]

        # Obtain timestamp
        timestamp, timestamp_ps = pack(
            (torch.zeros_like(time_next), expanded_time.diff(dim = -1)),
            'b s ne *'
        )                                                                      # [batch_size, seq_len, num_events, resolution]
        expanded_integral_all_events = (expanded_intensity_all_events_monte_carlo * timestamp.unsqueeze(dim = -1)).cumsum(dim = -2)
                                                                               # [batch_size, seq_len, num_events, resolution, num_events]

        return expanded_integral_all_events, expanded_intensity_all_events, timestamp


    def model_probe_function(self, events_history, time_history, time_next, mask_history, mask_next, resolution, mean, var):
        history, (_, _) = self.history_encoder(time_history, events_history, mask_history)
                                                                               # [batch_size, seq_len, d_input]
        eta = self.start_layer(history)                                        # [batch_size, seq_len, d_input]
        mu = self.converge_layer(history)                                      # [batch_size, seq_len, d_input]
        gamma = self.decay_layer(history)                                      # [batch_size, seq_len, d_input]

        time_multiplier = torch.linspace(0, 1, resolution, device = self.device)
        expanded_time = time_next.unsqueeze(dim = -1) * time_multiplier        # [batch_size, seq_len, resolution]
        expanded_hidden_state_at_t = self.state_decay(mu = mu, eta = eta, gamma = gamma, duration_t = expanded_time)
                                                                               # [batch_size, seq_len, resolution, d_input]

        expanded_intensity_all_events = self.intensity_layer(expanded_hidden_state_at_t)
                                                                               # [batch_size, seq_len, resolution, num_events]

        expanded_intensity_all_events_monte_carlo, \
        expanded_intensity_all_events_monte_carlo_ps = pack(
            (expanded_intensity_all_events[:, :, 0, :], 
             expanded_intensity_all_events[:, :, :-1, :]),
             'b s * ne'
        )                                                                      # [batch_size, seq_len, resolution, num_events]

        # Obtain timestamp
        timestamp, timestamp_ps = pack(
            (torch.zeros_like(time_next), expanded_time.diff(dim = -1)),
            'b s *'
        )                                                                      # [batch_size, seq_len, resolution]
        expanded_integral_all_events = (expanded_intensity_all_events_monte_carlo * timestamp.unsqueeze(dim = -1)).cumsum(dim = -2)
                                                                               # [batch_size, seq_len, resolution, num_events]                                                        # [batch_size, seq_len, resolution]
        
        # construct the plot dict
        data = {}
        data['expand_intensity_for_each_event'] = expanded_intensity_all_events# [batch_size, seq_len, resolution, num_events]
        data['expand_integral_for_each_event'] = expanded_integral_all_events  # [batch_size, seq_len, resolution, num_events]

        # THP always assumes that the event information is present.
        # So model_probe_function() always provides spearman, pearson coefficient and L1 distance.

        expand_intensity = rearrange(expanded_intensity_all_events.detach().cpu(), 'b s r ne -> b (s r) ne')
                                                                               # [batch_size, seq_len * resolution, num_event]
        expand_integral = rearrange(expanded_integral_all_events.detach().cpu(), 'b s r ne -> b (s r) ne')
                                                                               # [batch_size, seq_len * resolution, num_event]
            
        spearman_matrix = []
        pearson_matrix = []
        L1_matrix = []
        for idx, (expand_intensity_per_seq, expand_integral_per_seq, mask_per_seq, time_next_per_seq) \
            in enumerate(zip(expand_intensity, expand_integral, mask_next, time_next)):
            seq_len = mask_per_seq.sum()

            probability_distribution = expand_intensity_per_seq * torch.exp(-expand_integral_per_seq)
            # rho: spearman coefficient
            spearman_matrix_per_seq = spearmanr(probability_distribution[:seq_len * resolution])[0]
            if self.num_events == 2:
                spearman_matrix_per_seq = np.array([[1, spearman_matrix], [spearman_matrix, 1]])

            # r: pearson coefficient
            pearson_matrix_per_seq = np.corrcoef(probability_distribution[:seq_len * resolution], rowvar = False)
            # L^1 metric
            L1_matrix_per_seq = L1_distance_across_events(probability_distribution[:seq_len * resolution], 
                                            resolution = resolution, num_events = self.num_events,
                                            time_next = time_next_per_seq[:seq_len])
            spearman_matrix.append(spearman_matrix_per_seq)
            pearson_matrix.append(pearson_matrix_per_seq)
            L1_matrix.append(L1_matrix_per_seq)

        data['spearman_matrix'] = spearman_matrix
        data['pearson_matrix'] = pearson_matrix
        data['L1_matrix'] = L1_matrix
        
        return data, timestamp