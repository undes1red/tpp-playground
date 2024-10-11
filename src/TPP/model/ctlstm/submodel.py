import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce, pack
import numpy as np
from scipy.stats import spearmanr

from src.toolbox.misc import move_from_tensor_to_ndarray
from src.toolbox.metrics import L1_distance_across_events
from src.toolbox.integration import approximate_integration
from src.toolbox.patchifier import Patchifier
from src.toolbox.position_embedding import BiasedPositionalEmbedding


class CTLSTM(nn.Module):
    def __init__(self, device, num_events, history_module_name, d_mark_embedding, d_input, d_hidden, \
                 history_encoder_layers, dropout, integration_sample_rate, patchify, patch_length):
        '''
        A CTLSTM implementation, based on existing SAHP codes.
        '''
        super(CTLSTM, self).__init__()
        self.num_events = num_events
        self.device = device
        self.integration_sample_rate = integration_sample_rate
        self.patchify = patchify

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
        
        # Mark embedding layer.
        self.events_embedding = nn.Embedding(num_events + 1, d_mark_embedding, padding_idx = num_events, device = device)
        # Time embedding layer
        self.position_emb = BiasedPositionalEmbedding(d_mark_embedding, max_len = 4096, device = self.device)

        if self.patchify:
            self.patchifier = Patchifier(d_mark_embedding, patch_length = patch_length, device = self.device)

        # History encoder.
        self.history_encoder = getattr(nn, history_module_name)(device = self.device, \
                                       input_size = d_mark_embedding, hidden_size = d_hidden, \
                                       dropout = dropout, num_layers = history_encoder_layers, batch_first = True)
        self.history_mapper = nn.Linear(d_hidden, d_input, device = self.device)
    
    
    def state_decay(self, mu, eta, gamma, duration_t, num_dimension_prior_batch):
        '''
        mu, eta, gamma: shape: [batch_size, seq_len, d_hidden]
        dutation_t:     shape: [batch_size, seq_len, (integration_sample_rate, num_events)]
        '''
        assert len(duration_t.shape) - 2 - num_dimension_prior_batch >= 0, "Too few dimensions in duration_t!"

        # add additional dimension to mu, eta, and gamma.
        mu = rearrange(mu, f'... d_i -> {"() " * num_dimension_prior_batch}... {"() " * (len(duration_t.shape) - 2 - num_dimension_prior_batch)}d_i')
                                                                               # [..., batch_size, seq_len, (integration_sample_rate, num_events), d_input]
        eta = rearrange(eta, f'... d_i -> {"() " * num_dimension_prior_batch}... {"() " * (len(duration_t.shape) - 2 - num_dimension_prior_batch)}d_i')
                                                                               # [..., batch_size, seq_len, (integration_sample_rate, num_events), d_input]
        gamma = rearrange(gamma, f'... d_i -> {"() " * num_dimension_prior_batch}... {"() " * (len(duration_t.shape) - 2 - num_dimension_prior_batch)}d_i')
                                                                               # [..., batch_size, seq_len, (integration_sample_rate, num_events), d_input]

        duration_t = duration_t.unsqueeze(dim = -1)                            # [..., batch_size, seq_len, (integration_sample_rate, num_events), 1]
        cell_t = mu + (eta - mu) * torch.exp(-gamma * duration_t)              # [..., batch_size, seq_len, (integration_sample_rate, num_events), d_input]
        
        return cell_t


    def forward(self, time_history, time_next, events_history, num_dimension_prior_batch = 0):
        seq_len = events_history.shape[-1]
        events_embeddings = self.events_embedding(events_history)              # [batch_size, seq_len, d_mark_embedding]
        time_embeddings = self.position_emb(seq_len, time_history)             # [batch_size, seq_len, d_mark_embedding]
        history = events_embeddings + time_embeddings                          # [batch_size, seq_len, d_mark_embedding]

        if self.patchify:
            history = self.patchifier(history)                                 # [batch_size, seq_len, d_hidden]
            
        history, (_, _) = self.history_encoder(history)                        # [batch_size, seq_len, d_hidden]
        history = self.history_mapper(history)                                 # [batch_size, seq_len, d_input]

        eta = self.start_layer(history)                                        # [batch_size, seq_len, d_input]
        mu = self.converge_layer(history)                                      # [batch_size, seq_len, d_input]
        gamma = self.decay_layer(history)                                      # [batch_size, seq_len, d_input]

        hidden_state_at_t = self.state_decay(mu = mu, eta = eta, gamma = gamma, duration_t = time_next, num_dimension_prior_batch = num_dimension_prior_batch)
                                                                               # [..., batch_size, seq_len, d_input]
        # calculate the intensity.
        intensity_all_events = self.intensity_layer(hidden_state_at_t)         # [..., batch_size, seq_len, num_events]
        # calculate the integral
        time_multiplier = torch.linspace(0, 1, self.integration_sample_rate, device = self.device)
        expanded_time = time_next.unsqueeze(dim = -1) * time_multiplier        # [..., batch_size, seq_len, integration_sample_rate]
        expanded_hidden_state_at_t = self.state_decay(mu = mu, eta = eta, gamma = gamma, duration_t = expanded_time, num_dimension_prior_batch = num_dimension_prior_batch)
                                                                               # [..., batch_size, seq_len, integration_sample_rate, num_events]
        expanded_intensity_all_events = self.intensity_layer(expanded_hidden_state_at_t)
                                                                               # [..., batch_size, seq_len, integration_sample_rate, num_events]
        integral_all_events = approximate_integration(expanded_intensity_all_events, expanded_time, dim = -2, only_integral = True)
                                                                               # [..., batch_size, seq_len, num_events]

        return integral_all_events, intensity_all_events


    def integral_intensity_time_next_2d(self, events_history, time_history, time_next, integration_sample_rate, num_dimension_prior_batch = 0, time_next_start = None):
        if time_next_start is None:
            time_next_start = torch.zeros_like(time_next)                      # [batch_size, seq_len]

        events_embeddings = self.events_embedding(events_history)              # [batch_size, seq_len, d_mark_embedding]
        history = torch.cat((events_embeddings, time_history.unsqueeze(dim = -1)), dim = -1)
                                                                               # [batch_size, seq_len, d_mark_embedding + 1]
        
        history, (_, _) = self.history_encoder(history)                        # [batch_size, seq_len, d_hidden]
        history = self.history_mapper(history)                                 # [batch_size, seq_len, d_input]

        eta = self.start_layer(history)                                        # [batch_size, seq_len, d_input]
        mu = self.converge_layer(history)                                      # [batch_size, seq_len, d_input]
        gamma = self.decay_layer(history)                                      # [batch_size, seq_len, d_input]

        time_multiplier = torch.linspace(0, 1, integration_sample_rate, device = self.device)
        expanded_time = (time_next - time_next_start).unsqueeze(dim = -1) * time_multiplier + time_next_start.unsqueeze(dim = -1)
                                                                               # [..., batch_size, seq_len, integration_sample_rate]
        expanded_hidden_state_at_t = self.state_decay(mu = mu, eta = eta, gamma = gamma, duration_t = expanded_time, num_dimension_prior_batch = num_dimension_prior_batch)
                                                                               # [..., batch_size, seq_len, integration_sample_rate, d_input]

        expanded_intensity_all_events = self.intensity_layer(expanded_hidden_state_at_t)
                                                                               # [..., batch_size, seq_len, integration_sample_rate, num_events]
        expanded_integral_all_events = approximate_integration(expanded_intensity_all_events, expanded_time, dim = -2)
                                                                               # [..., batch_size, seq_len, integration_sample_rate, num_events]

        return expanded_integral_all_events, expanded_intensity_all_events, expanded_time


    def integral_intensity_time_next_3d(self, events_history, time_history, time_next, integration_sample_rate, num_dimension_prior_batch = 0):
        events_embeddings = self.events_embedding(events_history)              # [batch_size, seq_len, d_mark_embedding]
        history, history_ps = pack([events_embeddings, time_history], 'b s *') # [batch_size, seq_len, d_mark_embedding + 1]
        
        history, (_, _) = self.history_encoder(history)                        # [batch_size, seq_len, d_hidden]
        history = self.history_mapper(history)                                 # [batch_size, seq_len, d_input]

        eta = self.start_layer(history)                                        # [batch_size, seq_len, d_input]
        mu = self.converge_layer(history)                                      # [batch_size, seq_len, d_input]
        gamma = self.decay_layer(history)                                      # [batch_size, seq_len, d_input]

        time_multiplier = torch.linspace(0, 1, integration_sample_rate, device = self.device)
        expanded_time = time_next.unsqueeze(dim = -1) * time_multiplier        # [..., batch_size, seq_len, num_events, integration_sample_rate]
        expanded_hidden_state_at_t = self.state_decay(mu = mu, eta = eta, gamma = gamma, duration_t = expanded_time, num_dimension_prior_batch = num_dimension_prior_batch)
                                                                               # [..., batch_size, seq_len, num_events, integration_sample_rate, d_input]

        expanded_intensity_all_events = self.intensity_layer(expanded_hidden_state_at_t)
                                                                               # [..., batch_size, seq_len, num_events, integration_sample_rate, num_events]
        expanded_integral_all_events = approximate_integration(expanded_intensity_all_events, expanded_time, dim = -2)
                                                                               # [..., batch_size, seq_len, num_events, integration_sample_rate, num_events]

        return expanded_integral_all_events, expanded_intensity_all_events, expanded_time


    def model_probe_function(self, events_history, time_history, time_next, mask_next, integration_sample_rate):
        events_embeddings = self.events_embedding(events_history)              # [batch_size, seq_len, d_mark_embedding]
        history, history_ps = pack([events_embeddings, time_history], 'b s *') # [batch_size, seq_len, d_mark_embedding + 1]
        
        history, (_, _) = self.history_encoder(history)                        # [batch_size, seq_len, d_hidden]
        history = self.history_mapper(history)
                                                                               # [batch_size, seq_len, d_input]
        eta = self.start_layer(history)                                        # [batch_size, seq_len, d_input]
        mu = self.converge_layer(history)                                      # [batch_size, seq_len, d_input]
        gamma = self.decay_layer(history)                                      # [batch_size, seq_len, d_input]

        time_multiplier = torch.linspace(0, 1, integration_sample_rate, device = self.device)
        expanded_time = time_next.unsqueeze(dim = -1) * time_multiplier        # [batch_size, seq_len, integration_sample_rate]
        expanded_hidden_state_at_t = self.state_decay(mu = mu, eta = eta, gamma = gamma, duration_t = expanded_time, num_dimension_prior_batch = 0)
                                                                               # [batch_size, seq_len, integration_sample_rate, d_input]

        expanded_intensity_all_events = self.intensity_layer(expanded_hidden_state_at_t)
                                                                               # [batch_size, seq_len, integration_sample_rate, num_events]
        expanded_integral_all_events = approximate_integration(expanded_intensity_all_events, expanded_time, dim = -2)
                                                                               # [batch_size, seq_len, integration_sample_rate, num_events]
        
        # construct the plot dict
        data = {}
        data['expand_intensity_for_each_event'] = expanded_intensity_all_events# [batch_size, seq_len, integration_sample_rate, num_events]
        data['expand_integral_for_each_event'] = expanded_integral_all_events  # [batch_size, seq_len, integration_sample_rate, num_events]

        # THP always assumes that the event information is present.
        # So model_probe_function() always provides spearman, pearson coefficient and L1 distance.

        expand_intensity = rearrange(expanded_intensity_all_events, 'b s r ne -> b (s r) ne')
                                                                               # [batch_size, seq_len * integration_sample_rate, num_event]
        expand_integral = rearrange(expanded_integral_all_events, 'b s r ne -> b (s r) ne')
                                                                               # [batch_size, seq_len * integration_sample_rate, num_event]
            
        spearman_matrix = []
        pearson_matrix = []
        L1_matrix = []
        for idx, (expand_intensity_per_seq, expand_integral_per_seq, mask_per_seq, expanded_time_per_seq) \
            in enumerate(zip(expand_intensity, expand_integral, mask_next, expanded_time)):
            seq_len = mask_per_seq.sum()
            probability_distribution = expand_intensity_per_seq * torch.exp(-expand_integral_per_seq)
            probability_distribution = move_from_tensor_to_ndarray(probability_distribution)

            # rho: spearman coefficient
            if self.num_events == 1:
                spearman_matrix_per_seq = np.array([[1.,],])
            else:
                spearman_matrix_per_seq = spearmanr(probability_distribution[:seq_len * integration_sample_rate])[0]
                if self.num_events == 2:
                    spearman_matrix_per_seq = np.array([[1, spearman_matrix_per_seq], [spearman_matrix_per_seq, 1]])

            # r: pearson coefficient
            pearson_matrix_per_seq = np.corrcoef(probability_distribution[:seq_len * integration_sample_rate], rowvar = False)
            if self.num_events == 1:
                pearson_matrix_per_seq = rearrange(np.array(pearson_matrix_per_seq), ' -> () ()')
            
            # L^1 metric
            L1_matrix_per_seq = L1_distance_across_events(probability_distribution[:seq_len * integration_sample_rate], 
                                                          time_next = expanded_time_per_seq[:seq_len], has_flatten = True)
            spearman_matrix.append(spearman_matrix_per_seq)
            pearson_matrix.append(pearson_matrix_per_seq)
            L1_matrix.append(L1_matrix_per_seq)

        data['spearman_matrix'] = spearman_matrix
        data['pearson_matrix'] = pearson_matrix
        data['L1_matrix'] = L1_matrix
        
        return data, expanded_time