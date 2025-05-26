import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce, pack
import numpy as np
from scipy.stats import spearmanr

from src.toolbox.misc import move_from_tensor_to_ndarray
from src.toolbox.metrics import L1_distance_across_events
from src.toolbox.integration import approximate_integration
from src.toolbox.position_embedding import BiasedPositionalEmbedding


class CTLSTM(nn.Module):
    def __init__(self, device, num_events, history_module_name, d_mark_embedding, d_input, d_hidden, \
                 history_encoder_layers, dropout, integration_sample_rate):
        '''
        This function creates a CTLSTM model.
        
        ### Args
            * ```int``` d_mark_embedding
              The dimension of the mark embeddings.
            * ```str``` history_module_name
              Which RNN model do we use to encode the history? Default is LSTM. We don't recommend to change it to something else.
            * ```int``` d_hidden
              The dimension of the history representation.
            * ```float``` dropout
              Dropout rate for the history encoder. Only works when history_encoder_layers > 1.
            * ```int``` history_encoder_layers
              How many layer of RNN our model will have?
            * ```int``` d_input
              The dimension of the cumulative hazard function network.
            * ```namespace``` opt
              Model arguments.
            * ```torch.device``` device
              Running models on GPU or CPU?
            * ```int``` integration_sample_rate
              The number of interpolated points in a time interval between two adjoint events for integration estimation.
              The number of interpolated points counts the start and end point of the interval.
        '''
        super(CTLSTM, self).__init__()
        self.num_events = num_events
        self.device = device
        self.integration_sample_rate = integration_sample_rate

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

        # History encoder.
        self.history_encoder = getattr(nn, history_module_name)(device = self.device, \
                                       input_size = d_mark_embedding, hidden_size = d_hidden, \
                                       dropout = dropout, num_layers = history_encoder_layers, batch_first = True)
        self.history_mapper = nn.Linear(d_hidden, d_input, device = self.device)
    
    
    def state_decay(self, mu, eta, gamma, duration_t, num_dimension_prior_batch):
        '''
        This function decays the hidden state using a Hawkes-like rule by time.
        
        ### Args:
          * ```torch.tensor``` mu
            shape: ```[..., batch_size, seq_len, d_hidden]```
          * ```torch.tensor``` eta
            shape: ```[..., batch_size, seq_len, d_hidden]```
          * ```torch.tensor``` gamma
            shape: ```[..., batch_size, seq_len, d_hidden]```
            mu, eta, and gamma for state decay.
          * ```torch.tensor``` duration_t
            shape: ```[batch_size, seq_len, (integration_sample_rate, num_events)]```
            Decay by how much time?
          * ```int``` num_dimension_prior_batch
            How many dimensions does the input mu, eta, and gamma have before the batch_size dim?
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
        cell_t = F.tanh(mu + (eta - mu) * torch.exp(-gamma * duration_t))      # [..., batch_size, seq_len, (integration_sample_rate, num_events), d_input]
        
        return cell_t


    def forward(self, time_history, time_next, events_history, num_dimension_prior_batch = 0):
        '''
        CTLSTM's forwardpropagation function for training.
        
        ### Args
            * ```torch.tensor``` events_history
              shape: ```[batch_size, seq_len]```
              Historical event sequence.
            * ```torch.tensor``` time_history
              shape: ```[batch_size, seq_len]```
              Historical time sequence.
            * ```torch.tensor``` time_next
              shape: ```[..., batch_size, seq_len]```
              Guessed or real time when the next event will happen.
            * ```int``` num_dimension_prior_batch
              How many dimensions does the input mu, eta, and gamma have before the batch_size dim?
        ### Outputs
            * ```torch.tensor``` integral_all_events
              shape: ```[..., batch_size, seq_len, num_events]```
              The value of \\Lambda^*(m, t) on [t_{i-1}, t_i).
            * ```torch.tensor``` intensity_all_events
              shape: ```[..., batch_size, seq_len, num_events]```
              The value of \\lambda^*(m, t) on at t_i.
        '''
        seq_len = events_history.shape[-1]
        events_embeddings = self.events_embedding(events_history)              # [batch_size, seq_len, d_mark_embedding]
        time_embeddings = self.position_emb(seq_len, time_history)             # [batch_size, seq_len, d_mark_embedding]
        history = events_embeddings + time_embeddings                          # [batch_size, seq_len, d_mark_embedding]
            
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
                                                                               # [..., batch_size, seq_len, integration_sample_rate, d_input]
        expanded_intensity_all_events = self.intensity_layer(expanded_hidden_state_at_t)
                                                                               # [..., batch_size, seq_len, integration_sample_rate, num_events]
        integral_all_events = approximate_integration(expanded_intensity_all_events, expanded_time, dim = -2, only_integral = True)
                                                                               # [..., batch_size, seq_len, num_events]

        return integral_all_events, intensity_all_events


    def nhps_get_history_state(self, time_history, events_history):
        seq_len = events_history.shape[-1]
        events_embeddings = self.events_embedding(events_history)              # [batch_size, seq_len, d_mark_embedding]
        time_embeddings = self.position_emb(seq_len, time_history)             # [batch_size, seq_len, d_mark_embedding]
        history = events_embeddings + time_embeddings                          # [batch_size, seq_len, d_mark_embedding]

        history, (_, _) = self.history_encoder(history)                        # [batch_size, seq_len, d_hidden]
        history = self.history_mapper(history)                                 # [batch_size, seq_len, d_input]
        
        return history


    def nhps_get_decayed_state(self, history, time_next, num_dimension_prior_batch = 0):
        eta = self.start_layer(history)                                        # [batch_size, seq_len, d_input]
        mu = self.converge_layer(history)                                      # [batch_size, seq_len, d_input]
        gamma = self.decay_layer(history)                                      # [batch_size, seq_len, d_input]

        hidden_state_at_t = self.state_decay(mu = mu, eta = eta, gamma = gamma, duration_t = time_next, num_dimension_prior_batch = num_dimension_prior_batch)
                                                                               # [..., batch_size, seq_len, d_input]
        '''
        time_multiplier = torch.linspace(0, 1, self.integration_sample_rate, device = self.device)
        expanded_time = time_next.unsqueeze(dim = -1) * time_multiplier        # [..., batch_size, seq_len, integration_sample_rate]
        expanded_hidden_state_at_t = self.state_decay(mu = mu, eta = eta, gamma = gamma, duration_t = expanded_time, num_dimension_prior_batch = num_dimension_prior_batch)
                                                                               # [..., batch_size, seq_len, integration_sample_rate, d_input]
        '''
        return hidden_state_at_t


    def nhps_get_decayed_state_of_a_interval(self, history, time_interval_start, time_interval_length, num_dimension_prior_batch = 0):
        eta = self.start_layer(history)                                        # [batch_size, seq_len, d_input]
        mu = self.converge_layer(history)                                      # [batch_size, seq_len, d_input]
        gamma = self.decay_layer(history)                                      # [batch_size, seq_len, d_input]

        time_multiplier = torch.linspace(0, 1, self.integration_sample_rate, device = self.device)
        expanded_time = time_interval_length.unsqueeze(dim = -1) * time_multiplier + time_interval_start.unsqueeze(dim = -1)
                                                                               # [..., batch_size, seq_len, integration_sample_rate]
        expanded_hidden_states = self.state_decay(mu = mu, eta = eta, gamma = gamma, duration_t = expanded_time.float(), num_dimension_prior_batch = num_dimension_prior_batch)
                                                                               # [..., batch_size, seq_len, integration_sample_rate, d_input]
        
        return expanded_hidden_states, expanded_time
    
    
    def nhps_get_intensity(self, input_state):
        return self.intensity_layer(input_state)                               # [..., num_events]
    
    
    def sample_for_tm(self, time_history, time_next, events_history):
        '''
        CTLSTM's forwardpropagation function specific for sampling time first then mark.
        
        ### Args
            * ```torch.tensor``` events_history
              shape: ```[number_of_sampled_sequences, sampled_seq_len]```
              Historical event sequence.
            * ```torch.tensor``` time_history
              shape: ```[number_of_sampled_sequences, sampled_seq_len]```
              Historical time sequence.
            * ```torch.tensor``` time_next
              shape: ```[number_of_sampled_sequences, num_events]```
              Guessed time when the next event will happen.
        ### Outputs
            * ```torch.tensor``` integral_all_events
              shape: ```[number_of_sampled_sequences, num_events]```
              The value of \\Lambda^*(m, t) on [t_{i-1}, t_i).
            * ```torch.tensor``` intensity_all_events
              shape: ```[number_of_sampled_sequences, num_events]```
              The value of \\lambda^*(m, t) on at t_i.
        '''
        seq_len = events_history.shape[-1]
        events_embeddings = self.events_embedding(events_history)              # [number_of_sampled_sequences, seq_len, d_mark_embedding]
        time_embeddings = self.position_emb(seq_len, time_history)             # [number_of_sampled_sequences, seq_len, d_mark_embedding]
        history = events_embeddings + time_embeddings                          # [number_of_sampled_sequences, seq_len, d_mark_embedding]
            
        _, (sampled_history_embedding, _) = self.history_encoder(history)      # [1, number_of_sampled_sequences, d_hidden]
        if len(sampled_history_embedding.shape) == 3:
            sampled_history_embedding = rearrange(sampled_history_embedding, '() bs dh -> bs () dh')
                                                                               # [number_of_sampled_sequences, 1, d_history]
        history = self.history_mapper(sampled_history_embedding)               # [number_of_sampled_sequences, 1, d_input]

        eta = self.start_layer(history)                                        # [number_of_sampled_sequences, 1, d_input]
        mu = self.converge_layer(history)                                      # [number_of_sampled_sequences, 1, d_input]
        gamma = self.decay_layer(history)                                      # [number_of_sampled_sequences, 1, d_input]
        
        time_next = time_next.unsqueeze(dim = -1)                              # [number_of_sampled_sequences, 1]
        hidden_state_at_t = self.state_decay(mu = mu, eta = eta, gamma = gamma, duration_t = time_next, num_dimension_prior_batch = 0)
                                                                               # [number_of_sampled_sequences, 1, d_input]
        # calculate the intensity.
        intensity_all_events = self.intensity_layer(hidden_state_at_t)         # [number_of_sampled_sequences, 1, num_events]
        intensity_all_events = intensity_all_events.squeeze(dim = -2)          # [number_of_sampled_sequences, num_events]
        # calculate the integral
        time_multiplier = torch.linspace(0, 1, self.integration_sample_rate, device = self.device)
        expanded_time = time_next.unsqueeze(dim = -1) * time_multiplier        # [number_of_sampled_sequences, integration_sample_rate]
        expanded_hidden_state_at_t = self.state_decay(mu = mu, eta = eta, gamma = gamma, duration_t = expanded_time, num_dimension_prior_batch = 0)
                                                                               # [number_of_sampled_sequences, 1, integration_sample_rate, d_input]
        expanded_intensity_all_events = self.intensity_layer(expanded_hidden_state_at_t)
                                                                               # [number_of_sampled_sequences, 1, integration_sample_rate, num_events]
        integral_all_events = approximate_integration(expanded_intensity_all_events, expanded_time, dim = -2, only_integral = True)
                                                                               # [number_of_sampled_sequences, 1, num_events]
        integral_all_events = integral_all_events.squeeze(dim = -2)            # [number_of_sampled_sequences, num_events]

        return integral_all_events, intensity_all_events


    def sample_for_mt(self, time_history, time_next, events_history):
        '''
        CTLSTM's forwardpropagation function specific for sampling mark first then time.
        
        ### Args
            * ```torch.tensor``` events_history
              shape: ```[number_of_sampled_sequences, sampled_seq_len]```
              Historical event sequence.
            * ```torch.tensor``` time_history
              shape: ```[number_of_sampled_sequences, sampled_seq_len]```
              Historical time sequence.
            * ```torch.tensor``` time_next
              shape: ```[number_of_sampled_sequences, num_events]```
              Guessed time when the next event will happen.
        ### Outputs
            * ```torch.tensor``` integral_all_events
              shape: ```[number_of_sampled_sequences, num_events]```
              The value of \\Lambda^*(m, t) on [t_{i-1}, t_i).
            * ```torch.tensor``` intensity_all_events
              shape: ```[number_of_sampled_sequences, num_events]```
              The value of \\lambda^*(m, t) on at t_i.
        '''
        seq_len = events_history.shape[-1]
        events_embeddings = self.events_embedding(events_history)              # [number_of_sampled_sequences, seq_len, d_mark_embedding]
        time_embeddings = self.position_emb(seq_len, time_history)             # [number_of_sampled_sequences, seq_len, d_mark_embedding]
        history = events_embeddings + time_embeddings                          # [number_of_sampled_sequences, seq_len, d_mark_embedding]
            
        _, (sampled_history_embedding, _) = self.history_encoder(history)      # [1, number_of_sampled_sequences, d_hidden]
        sampled_history_embedding = rearrange(sampled_history_embedding, '() bs dh -> bs () dh')
                                                                               # [number_of_sampled_sequences, 1, d_history]
        history = self.history_mapper(sampled_history_embedding)               # [number_of_sampled_sequences, 1, d_input]

        eta = self.start_layer(history)                                        # [number_of_sampled_sequences, 1, d_input]
        mu = self.converge_layer(history)                                      # [number_of_sampled_sequences, 1, d_input]
        gamma = self.decay_layer(history)                                      # [number_of_sampled_sequences, 1, d_input]
        
        time_next = time_next.unsqueeze(dim = -1)                              # [number_of_sampled_sequences, 1]
        hidden_state_at_t = self.state_decay(mu = mu, eta = eta, gamma = gamma, duration_t = time_next, num_dimension_prior_batch = 0)
                                                                               # [number_of_sampled_sequences, 1, d_input]
        # calculate the intensity.
        intensity_all_events = self.intensity_layer(hidden_state_at_t)         # [number_of_sampled_sequences, 1, num_events]
        intensity_all_events = intensity_all_events.squeeze(dim = -2)          # [number_of_sampled_sequences, num_events]
        # calculate the integral
        time_multiplier = torch.linspace(0, 1, self.integration_sample_rate, device = self.device)
        expanded_time = time_next.unsqueeze(dim = -1) * time_multiplier        # [number_of_sampled_sequences, integration_sample_rate]
        expanded_hidden_state_at_t = self.state_decay(mu = mu, eta = eta, gamma = gamma, duration_t = expanded_time, num_dimension_prior_batch = 0)
                                                                               # [number_of_sampled_sequences, 1, integration_sample_rate, d_input]
        expanded_intensity_all_events = self.intensity_layer(expanded_hidden_state_at_t)
                                                                               # [number_of_sampled_sequences, 1, integration_sample_rate, num_events]
        integral_all_events = approximate_integration(expanded_intensity_all_events, expanded_time, dim = -2, only_integral = True)
                                                                               # [number_of_sampled_sequences, 1, num_events]
        integral_all_events = integral_all_events.squeeze(dim = -2)            # [number_of_sampled_sequences, num_events]

        return integral_all_events, intensity_all_events


    def integral_intensity_time_next_2d(self, events_history, time_history, time_next, integration_sample_rate, num_dimension_prior_batch = 0, time_next_start = None):
        '''
        Probe the value of the intensity function and its integral at sampled timestamps.
        In this function, all marks share the sampled timestmaps, so the dimension of time_next does not include num_event.
        
        ### Args
            * ```torch.tensor``` events_history
              shape: ```[batch_size, seq_len]```
              Historical event sequence.
            * ```torch.tensor``` time_history
              shape: ```[batch_size, seq_len]```
              Historical time sequence.
            * ```torch.tensor``` time_next
              shape: ```[..., batch_size, seq_len]```
              Guessed or real time when the next event will happen.
            * ```int``` integration_sample_rate
              The number of interpolated points in a time interval between two adjoint events for integration estimation.
              The number of interpolated points counts the start and end point of the interval.
            * ```int``` num_dimension_prior_batch
              How many dimensions does the input mu, eta, and gamma have before the batch_size dim?
            * ```torch,tensor``` time_next_start
              shape: ```[..., batch_size, seq_len]``` if not None
              When given, this function computes the integral between [time_next_start, t_i]. time_next_start are expected to be non-negative.
              This affects the integral, intensity, and timestamp.
        ### Outputs
            * ```torch.tensor``` expanded_integral_all_events
              shape: ```[..., batch_size, seq_len, resolution, num_events]```
              The value of \\Lambda^*(m, t) at sampled times.
            * ```torch.tensor``` expanded_intensity_all_events
              shape: ```[..., batch_size, seq_len, resolution, num_events]```
              The value of \\lambda^*(m, t) at sampled times.
            * ```torch.tensor``` expanded_time
              shape: ```[..., batch_size, seq_len, resolution]```
              The value of sampled times.
        '''
        if time_next_start is None:
            time_next_start = torch.zeros_like(time_next)                      # [batch_size, seq_len]

        seq_len = events_history.shape[-1]
        events_embeddings = self.events_embedding(events_history)              # [batch_size, seq_len, d_mark_embedding]
        time_embeddings = self.position_emb(seq_len, time_history)             # [batch_size, seq_len, d_mark_embedding]
        history = events_embeddings + time_embeddings                          # [batch_size, seq_len, d_mark_embedding]
        
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
        '''
        Probe the value of the intensity function and its integral at sampled timestamps.
        In this function, all marks can have their sampled timestmaps, so the dimension of time_next is ```[..., batch_size, seq_len, num_events]```.
        This function is supposed to be much slower than integral_intensity_time_next_2d().
        
        ### Args
            * ```torch.tensor``` events_history
              shape: ```[batch_size, seq_len]```
              Historical event sequence.
            * ```torch.tensor``` time_history
              shape: ```[batch_size, seq_len]```
              Historical time sequence.
            * ```torch.tensor``` time_next
              shape: ```[..., batch_size, seq_len, num_events]```
              Guessed or real time when the next event will happen.
            * ```int``` integration_sample_rate
              The number of interpolated points in a time interval between two adjoint events for integration estimation.
              The number of interpolated points counts the start and end point of the interval.
            * ```int``` num_dimension_prior_batch
              How many dimensions does the input mu, eta, and gamma have before the batch_size dim?
        ### Outputs
            * ```torch.tensor``` expanded_integral_all_events
              shape: ```[..., batch_size, seq_len, resolution, num_events]```
              The value of \\Lambda^*(m, t) at sampled times.
            * ```torch.tensor``` expanded_intensity_all_events
              shape: ```[..., batch_size, seq_len, resolution, num_events]```
              The value of \\lambda^*(m, t) at sampled times.
            * ```torch.tensor``` expanded_time
              shape: ```[..., batch_size, seq_len, resolution]```
              The value of sampled times.
        '''
        seq_len = events_history.shape[-1]
        events_embeddings = self.events_embedding(events_history)              # [batch_size, seq_len, d_mark_embedding]
        time_embeddings = self.position_emb(seq_len, time_history)             # [batch_size, seq_len, d_mark_embedding]
        history = events_embeddings + time_embeddings                          # [batch_size, seq_len, d_mark_embedding]
        
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
        '''
        Probe the value of the intensity function and its integral at sampled timestamps.
        In this function, all marks can have their sampled timestmaps, so the dimension of time_next is ```[..., batch_size, seq_len, num_events]```.
        This function is supposed to be much slower than integral_intensity_time_next_2d().
        
        ### Args
            * ```torch.tensor``` events_history
              shape: ```[batch_size, seq_len]```
              Historical event sequence.
            * ```torch.tensor``` time_history
              shape: ```[batch_size, seq_len]```
              Historical time sequence.
            * ```torch.tensor``` time_next
              shape: ```[..., batch_size, seq_len]```
              Guessed or real time when the next event will happen.
            * ```torch.tensor``` mask_next
              shape: ```[..., batch_size, seq_len]```
              Tell which event in *_next is the real event so should be considered in metric calculation.
            * ```int``` integration_sample_rate
              The number of interpolated points in a time interval between two adjoint events for integration estimation.
              The number of interpolated points counts the start and end point of the interval.
        ### Outputs
            * ```dict``` data
              Probed data used for plot drawing.
            * ```torch.tensor``` expanded_time
              shape: ```[batch_size, seq_len, resolution]```
              The value of sampled times.
        '''
        seq_len = events_history.shape[-1]
        events_embeddings = self.events_embedding(events_history)              # [batch_size, seq_len, d_mark_embedding]
        time_embeddings = self.position_emb(seq_len, time_history)             # [batch_size, seq_len, d_mark_embedding]
        history = events_embeddings + time_embeddings                          # [batch_size, seq_len, d_mark_embedding]        

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