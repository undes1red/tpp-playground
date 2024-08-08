import torch
import torch.nn as nn
import numpy as np
from scipy.stats import spearmanr
from einops import rearrange, reduce, repeat

from src.toolbox.misc import move_from_tensor_to_ndarray
from src.toolbox.metrics import L1_distance_across_events


class MRMTPPModule(nn.Module):
    def __init__(self, input_size, hidden_size, history_encoder_layers, dropout, 
                 num_events, output_size, limited_history_norm, time_scalar_min, device):
        super(MRMTPPModule, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_events = num_events
        self.limited_history_norm = limited_history_norm
        self.time_scalar_min = time_scalar_min

        self.time_embedding = nn.Linear(1, input_size, device = self.device)
        self.rnn = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = history_encoder_layers, batch_first = True, \
                            dropout = dropout, device = self.device)
        self.project = nn.Linear(hidden_size, output_size, device = self.device)
        
        # Mark related
        self.event_embedding = nn.Embedding(num_embeddings = num_events + 1, embedding_dim = input_size,\
                                            padding_idx = num_events, device = self.device)

        self.non_neg_activation = nn.Softplus()

        self.intensity = nn.Linear(output_size, self.num_events, device = self.device)
        self.time_scalar = nn.Linear(output_size, self.num_events, device = self.device)
        self.base_intensity = nn.Linear(output_size, self.num_events, device = self.device)


    def clamp_time_scalar(self, time_scalar):
        time_scalar_sign = (time_scalar >= 0).int() - (time_scalar < 0).int()  # [batch_size, seq_len, num_events] if self.event_toggle else [batch_size, seq_len, 1]
        shifted_time_scalar_abs_value = torch.abs(time_scalar).clamp(min = self.time_scalar_min)
                                                                               # [batch_size, seq_len, num_events] if self.event_toggle else [batch_size, seq_len, 1]
        time_scalar = shifted_time_scalar_abs_value * time_scalar_sign         # [batch_size, seq_len, num_events] if self.event_toggle else [batch_size, seq_len, 1]
        return time_scalar


    def forward(self, events_history, time_history, time_next, mean, std, custom_events_history = False):
        '''
        This implementation is in fact an advanced RMTPP with history-event-related time scaler and base intensity.
        '''
        time_history = (time_history - mean) / std
        time_history = time_history.unsqueeze(dim = -1)                        # [batch_size, seq_len, 1]
        time_vec = self.time_embedding(time_history)                           # [batch_size, seq_len, input_size]

        if custom_events_history:
            events_vec = events_history                                        # [batch_size, seq_len, input_size]
        else:
            events_vec = self.event_embedding(events_history)                  # [batch_size, seq_len, input_size]
        input_vec = time_vec + events_vec

        hidden_history, (_, _) = self.rnn(input_vec)                           # [batch_size, seq_len, hidden_size]
        hidden_history = self.project(hidden_history)                          # [batch_size, seq_len, output_size]
        hidden_history = torch.relu(hidden_history)                            # [batch_size, seq_len, output_size]

        history_part = self.intensity(hidden_history)                          # [batch_size, seq_len, num_events]

        if self.limited_history_norm:
            history_part = torch.tanh(history_part)                            # [batch_size, seq_len, num_events]

        history_part = torch.exp(history_part)                                 # [batch_size, seq_len, num_events]

        constant = history_part * torch.exp(self.base_intensity(hidden_history))
                                                                               # [batch_size, seq_len, num_events]
        time_scalar = self.time_scalar(hidden_history)                         # [batch_size, seq_len, num_events]

        # time_scalar can not be zero.
        time_scalar = self.clamp_time_scalar(time_scalar)                      # [batch_size, seq_len, num_events]

        time_next = (time_next) / std
        time_next = time_next.unsqueeze(dim = -1)                              # [..., batch_size, seq_len, 1]

        # reshape the parameters.
        ein_ops = f'... -> {"() " * (len(time_next.shape) - len(time_scalar.shape))}...'
        time_scalar = rearrange(time_scalar, ein_ops)                          # [..., batch_size, seq_len, num_events]
        constant = rearrange(constant, ein_ops)                                # [..., batch_size, seq_len, num_events]

        # Get the intensity function and corresponding integral.
        intensity = torch.exp(time_scalar * time_next) * constant              # [..., batch_size, seq_len, num_events]
        integral = (intensity - constant) / time_scalar * std                  # [..., batch_size, seq_len, num_events]

        return integral, intensity, history_part
    

    def get_event_embedding(self, input_event):
        return self.event_embedding(input_event)                               # [batch_size, seq_len, input_size]


    def integral_intensity_time_next_2d(self, events_history, time_history, time_next, resolution, mean, std):
        time_history = ((time_history - mean) / std).unsqueeze(dim = -1)

        time_vec = self.time_embedding(time_history)                           # [batch_size, seq_len, input_size]
        events_vec = self.event_embedding(events_history)                      # [batch_size, seq_len, input_size]
        input_vec = time_vec + events_vec                                      # [batch_size, seq_len, input_size]

        output, (_, _) = self.rnn(input_vec)                                   # [batch_size, seq_len, hidden_size]
        history_output = self.project(output)                                  # [batch_size, seq_len, output_size]
        history_output = torch.relu(history_output)                            # [batch_size, seq_len, output_size]

        history_part = self.intensity(history_output)                          # [batch_size, seq_len, num_events]

        if self.limited_history_norm:
            history_part = torch.tanh(history_part)                            # [batch_size, seq_len, num_events]

        history_part = torch.exp(history_part)                                 # [batch_size, seq_len, num_events]

        constant = (history_part * torch.exp(self.base_intensity(history_output))).unsqueeze(-1)
                                                                               # [batch_size, seq_len, num_events, 1]

        time_scalar = self.time_scalar(history_output).unsqueeze(dim = -1)     # [batch_size, seq_len, num_events, 1]

        # time_scalar can not be zero.
        time_scalar = self.clamp_time_scalar(time_scalar)                      # [batch_size, seq_len, num_events, 1]

        time_multiplier = torch.linspace(0, 1, resolution, device = self.device)
        original_time_expand = time_next.unsqueeze(dim = -1) * time_multiplier # [batch_size, seq_len, resolution]
        original_time_expand_normed = original_time_expand / std               # [batch_size, seq_len, resolution]
        expanded_time = original_time_expand_normed.unsqueeze(dim = -2)        # [batch_size, seq_len, 1, resolution]
        
        intensity_events = torch.exp(time_scalar * expanded_time) * constant   # [batch_size, seq_len, num_events, resolution]
        integral_events = (intensity_events - constant) / time_scalar * std    # [batch_size, seq_len, num_events, resolution]

        # aggregated timestamp
        batch_size, seq_len, _ = original_time_expand.shape
        timestamp = torch.cat(
            (torch.zeros((batch_size, seq_len, 1), device = self.device), original_time_expand.diff(dim = -1)),
            dim = -1)                                                          # [batch_size, seq_len, resolution]
        
        intensity = rearrange(intensity_events, 'b s ne r -> b s r ne')        # [batch_size, seq_len, resolution, num_events]
        integral = rearrange(integral_events, 'b s ne r -> b s r ne')          # [batch_size, seq_len, resolution, num_events]

        return integral, intensity, timestamp
    

    def integral_intensity_time_next_3d(self, events_history, time_history, time_next, resolution, mean, std):
        '''
        Shape of time_next: [..., batch_size, seq_len, num_events]
        '''
        time_history = ((time_history - mean) / std).unsqueeze(dim = -1)

        time_vec = self.time_embedding(time_history)                           # [batch_size, seq_len, input_size]
        events_vec = self.event_embedding(events_history)                      # [batch_size, seq_len, input_size]
        input_vec = time_vec + events_vec                                      # [batch_size, seq_len, input_size]

        output, (_, _) = self.rnn(input_vec)                                   # [batch_size, seq_len, hidden_size]
        history_output = self.project(output)                                  # [batch_size, seq_len, output_size]
        history_output = torch.relu(history_output)                            # [batch_size, seq_len, output_size]

        history_part = self.intensity(history_output)                          # [batch_size, seq_len, num_events]

        if self.limited_history_norm:
            history_part = torch.tanh(history_part)                            # [batch_size, seq_len, num_events]

        history_part = torch.exp(history_part)                                 # [batch_size, seq_len, num_events]

        constant = (history_part * torch.exp(self.base_intensity(history_output)))
                                                                               # [batch_size, seq_len, num_events]
        time_scalar = self.time_scalar(history_output)                         # [batch_size, seq_len, num_events]

        # time_scalar can not be zero.
        time_scalar = self.clamp_time_scalar(time_scalar)                      # [batch_size, seq_len, num_events]

        einop = f'... ne -> {"() " * (len(time_next.shape) - 3)}... () () ne'
        constant = rearrange(constant, einop)                                  # [..., batch_size, seq_len, 1, 1, num_events]
        time_scalar = rearrange(time_scalar, einop)                            # [..., batch_size, seq_len, 1, 1, num_events]

        time_multiplier = torch.linspace(0, 1, resolution, device = self.device)
        original_time_expand = time_next.unsqueeze(dim = -1) * time_multiplier # [..., batch_size, seq_len, num_events, resolution]
        original_time_expand_normed = original_time_expand / std               # [..., batch_size, seq_len, num_events, resolution]
        expanded_time = original_time_expand_normed.unsqueeze(dim = -1)        # [..., batch_size, seq_len, num_events, resolution, 1]
        
        intensity_events = torch.exp(time_scalar * expanded_time) * constant   # [..., batch_size, seq_len, num_events, resolution, num_events]
        integral_events = (intensity_events - constant) / time_scalar * std    # [..., batch_size, seq_len, num_events, resolution, num_events]

        # aggregated timestamp
        timestamp = torch.cat(
            (torch.zeros((*time_next.shape[:-1], self.num_events, 1), device = self.device), original_time_expand.diff(dim = -1)),
            dim = -1)                                                          # [..., batch_size, seq_len, resolution]
        
        return integral_events, intensity_events, timestamp


    def model_probe_function(self, events_history, time_history, time_next, mask_next, resolution, mean, std):
        time_history = ((time_history - mean) / std).unsqueeze(dim = -1)

        time_vec = self.time_embedding(time_history)                           # [batch_size, seq_len, input_size]
        events_vec = self.event_embedding(events_history)                      # [batch_size, seq_len, input_size]
        input_vec = time_vec + events_vec

        output, (_, _) = self.rnn(input_vec)                                   # [batch_size, seq_len, hidden_size]
        history_output = self.project(output)                                  # [batch_size, seq_len, output_size]
        history_output = torch.relu(history_output)                            # [batch_size, seq_len, output_size]

        history_part = self.intensity(history_output)                          # [batch_size, seq_len, num_events]
        if self.limited_history_norm:
            history_part = torch.tanh(history_part)                            # [batch_size, seq_len, num_events]

        history_part = torch.exp(history_part)                                 # [batch_size, seq_len, num_events]

        constant = (history_part * torch.exp(self.base_intensity(history_output))).unsqueeze(-1)
                                                                               # [batch_size, seq_len, num_events, 1]

        time_scalar = self.time_scalar(history_output).unsqueeze(dim = -1)     # [batch_size, seq_len, num_events, 1]

        # time_scalar can not be zero.
        time_scalar = self.clamp_time_scalar(time_scalar)                      # [batch_size, seq_len, num_events, 1]

        time_multiplier = torch.linspace(0, 1, resolution, device = self.device)
        original_time_expand = time_next.unsqueeze(dim = -1) * time_multiplier # [batch_size, seq_len, resolution]
        original_time_expand_normed = original_time_expand / std               # [batch_size, seq_len, resolution]
        expanded_time = original_time_expand_normed.unsqueeze(dim = -2)        # [batch_size, seq_len, 1, resolution]
        
        intensity_events = torch.exp(time_scalar * expanded_time) * constant   # [batch_size, seq_len, 1, resolution]
        integral_events = (intensity_events - constant) / time_scalar * std    # [batch_size, seq_len, 1, resolution]

        # aggregated timestamp
        batch_size, seq_len, _ = original_time_expand.shape
        timestamp = torch.cat(
            (torch.zeros((batch_size, seq_len, 1), device = self.device), original_time_expand.diff(dim = -1)),
            dim = -1)                                                          # [batch_size, seq_len, resolution]
        
        intensity = rearrange(intensity_events, 'b s ne r -> b s r ne')        # [batch_size, seq_len, resolution, num_events]
        integral = rearrange(integral_events, 'b s ne r -> b s r ne')          # [batch_size, seq_len, resolution, num_events]

        '''
        Here we start constructing data dict.
        '''
        data = {}
        data['expand_intensity_for_each_event'] = intensity                    # [batch_size, seq_len, resolution, num_events]
        data['expand_integral_for_each_event'] = integral                      # [batch_size, seq_len, resolution, num_events]

        expand_intensity = rearrange(intensity, 'b s r ne -> b (s r) ne')      # [batch_size, seq_len * integration_sample_rate, num_event]
        expand_integral = rearrange(integral, 'b s r ne -> b (s r) ne')        # [batch_size, seq_len * integration_sample_rate, num_event]
            
        spearman_matrix = []
        pearson_matrix = []
        L1_matrix = []
        for idx, (expand_intensity_per_seq, expand_integral_per_seq, mask_per_seq, time_next_per_seq) \
            in enumerate(zip(expand_intensity, expand_integral, mask_next, time_next)):
            seq_len = mask_per_seq.sum()
            probability_distribution = expand_intensity_per_seq * torch.exp(-expand_integral_per_seq)
            probability_distribution = move_from_tensor_to_ndarray(probability_distribution)

            # rho: spearman coefficient
            if self.num_events == 1:
                spearman_matrix_per_seq = np.array([[1.,],])
            else:
                spearman_matrix_per_seq = spearmanr(probability_distribution[:seq_len * resolution])[0]
                if self.num_events == 2:
                    spearman_matrix_per_seq = np.array([[1, spearman_matrix_per_seq], [spearman_matrix_per_seq, 1]])

            # r: pearson coefficient
            pearson_matrix_per_seq = np.corrcoef(probability_distribution[:seq_len * resolution], rowstd = False)
            if self.num_events == 1:
                pearson_matrix_per_seq = rearrange(np.array(pearson_matrix_per_seq), ' -> () ()')
            
            # L^1 metric
            L1_matrix_per_seq = L1_distance_across_events(probability_distribution[:seq_len * resolution], 
                                                          resolution = resolution,
                                                          time_next = time_next_per_seq[:seq_len])
            spearman_matrix.append(spearman_matrix_per_seq)
            pearson_matrix.append(pearson_matrix_per_seq)
            L1_matrix.append(L1_matrix_per_seq)

        data['spearman_matrix'] = spearman_matrix
        data['pearson_matrix'] = pearson_matrix
        data['L1_matrix'] = L1_matrix

        return data, timestamp