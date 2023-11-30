import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce, pack
import numpy as np
from scipy.stats import spearmanr

from src.TPP.model.utils import L1_distance_across_events, move_from_tensor_to_ndarray
from src.TPP.model.thp.utils import softplus_ext
from src.TPP.model.thp.transformers import TransformerTPP


class THP(nn.Module):
    def __init__(self, device, num_events, d_input, d_rnn, d_hidden, n_layers, n_head, d_qk,\
                 d_v, dropout, beta, integration_sample_rate, history_time_offset):
        super(THP, self).__init__()
        self.device = device
        self.num_events = num_events
        self.integration_sample_rate = integration_sample_rate
        self.history_time_offset = history_time_offset

        # parameter for the weight of time difference
        self.alpha = nn.Parameter(torch.ones((self.num_events), dtype = torch.float32, \
                                  device = self.device, requires_grad = True))

        # parameter for the softplus function
        self.beta = nn.Parameter(torch.ones((self.num_events), dtype = torch.float32, \
                                  device = self.device, requires_grad = True) * beta)
        
        # convert hidden vectors into valid intensity function values.
        self.linear = nn.Linear(d_input, num_events, device = self.device)

        # the history encoder
        self.history_encoder = TransformerTPP(num_events, device = self.device, d_input = d_input, \
                                              d_rnn = d_rnn, d_hidden = d_hidden, n_layers = n_layers, \
                                              n_head = n_head, d_qk = d_qk, d_v = d_v, dropout = dropout)


    def extract_history_embeddings(self, time, events, mask):
        '''
        Args:
        1. time: the sequence containing events' timestamps. shape: [batch_size, seq_len + 1]
        2. events: the sequence containing information about events. shape: [batch_size, seq_len + 1]
        3. mask: the padding mask introduced by the dataloader. shape: [batch_size, seq_len + 1]
        '''

        time_history, _ = self.divide_history_and_next(time)                   # [batch_size, seq_len]
        events_history, _ = self.divide_history_and_next(events)               # [batch_size, seq_len]
        mask_history, _ = self.divide_history_and_next(mask)                   # [batch_size, seq_len]

        history = self.history_encoder(time_history, events_history, mask_history)
                                                                               # [batch_size, seq_len, num_events]
        return history


    def integration_estimator(self, expanded_intensity_value, expanded_time, integration_sample_rate):
        # tensor check
        assert expanded_intensity_value.shape[-2:] == (integration_sample_rate, self.num_events)
        assert expanded_time.shape[-1] == integration_sample_rate
        
        expanded_intensity_value_1 = expanded_intensity_value[..., :-1, :]     # [..., integration_sample_rate - 1, num_events]
        expanded_intensity_value_2 = expanded_intensity_value[..., 1:, :]      # [..., integration_sample_rate - 1, num_events]
        timestamp_for_integral = expanded_time.diff(dim = -1)                  # [..., integration_sample_rate - 1]

        # \int_{a}{b}{f(x)dx} = \sum_{i = 0}^{N - 2}{f(\frac{(b - a)i}{N - 1}) * \frac{(b - a)}{N - 1}}
        integral_of_all_events_1 = (expanded_intensity_value_1 * timestamp_for_integral.unsqueeze(dim = -1)).cumsum(dim = -2)
                                                                               # [..., integration_sample_rate - 1, num_events]
        # \int_{a}{b}{f(x)dx} = \sum_{i = 0}^{N - 2}{f(\frac{(b - a)(i + 1)}{N - 1}) * \frac{(b - a)}{N - 1}}
        integral_of_all_events_2 = (expanded_intensity_value_2 * timestamp_for_integral.unsqueeze(dim = -1)).cumsum(dim = -2)
                                                                               # [..., integration_sample_rate - 1, num_events]
        # Effectively increase the precision.
        integral_of_all_events = (integral_of_all_events_1 + integral_of_all_events_2) / 2
                                                                               # [..., integration_sample_rate - 1, num_events]
        
        # Prepend 0 to integral_of_all_events because \int_{t_l}^{t_l}{\lambda^*(\tau)d\tau} = 0
        # We have to check the shape.
        integral_start_from_zero = torch.zeros(*(integral_of_all_events).shape[:-2], 1, self.num_events, device = self.device)
                                                                               # [..., 1, num_events]
        integral_of_all_events = torch.concat([integral_start_from_zero, integral_of_all_events], dim = -2)
                                                                               # [..., integration_sample_rate, num_events]

        return integral_of_all_events


    def integration_probability_estimator(self, expanded_probability_value, expanded_time, integration_sample_rate):
        # tensor check
        assert expanded_probability_value.shape[-2:] == (self.num_events, integration_sample_rate)
        assert expanded_time.shape[-1] == integration_sample_rate
        
        expanded_probability_value_1 = expanded_probability_value[..., :-1]    # [..., integration_sample_rate - 1]
        expanded_probability_value_2 = expanded_probability_value[..., 1:]     # [..., integration_sample_rate - 1]
        timestamp_for_integral = expanded_time.diff(dim = -1)                  # [..., integration_sample_rate - 1]

        # \int_{a}{b}{f(x)dx} = \sum_{i = 0}^{N - 2}{f(\frac{(b - a)i}{N - 1}) * \frac{(b - a)}{N - 1}}
        integral_of_all_events_1 = (expanded_probability_value_1 * timestamp_for_integral).cumsum(dim = -1)
                                                                               # [..., integration_sample_rate - 1]
        # \int_{a}{b}{f(x)dx} = \sum_{i = 0}^{N - 2}{f(\frac{(b - a)(i + 1)}{N - 1}) * \frac{(b - a)}{N - 1}}
        integral_of_all_events_2 = (expanded_probability_value_2 * timestamp_for_integral).cumsum(dim = -1)
                                                                               # [..., integration_sample_rate - 1]
        # Effectively increase the precision.
        integral_of_all_events = (integral_of_all_events_1 + integral_of_all_events_2) / 2
                                                                               # [..., integration_sample_rate - 1]
        
        # Prepend 0 to integral_of_all_events because \int_{t_l}^{t_l}{\lambda^*(\tau)d\tau} = 0
        # We have to check the shape.
        integral_start_from_0 = torch.zeros(*(integral_of_all_events).shape[:-1], 1, device = self.device)
                                                                               # [..., 1]
        integral_of_all_events = torch.concat((integral_start_from_0, integral_of_all_events), dim = -1)
                                                                               # [..., integration_sample_rate]

        return integral_of_all_events


    def forward(self, time_history, time_next, events_history, mask_history, mask_next = None):
        history = self.history_encoder(time_history, events_history, mask_history)
                                                                               # [batch_size, seq_len, d_input]

        aggregate_time = time_history.cumsum(dim = -1)                         # [batch_size, seq_len]
        # Avoid zero denominator
        aggregate_time = aggregate_time + self.history_time_offset             # [batch_size, seq_len]
        aggregate_time = rearrange(aggregate_time, f'... -> {"() " * (len(time_next.shape) - len(aggregate_time.shape))}...')
                                                                               # [..., batch_size, seq_len]
        history = rearrange(history, f'... -> {"() " * (len(time_next.shape) - len(aggregate_time.shape))}...')
                                                                               # [..., batch_size, seq_len, d_input]

        scaled_time = (time_next / aggregate_time).unsqueeze(dim = -1)         # [..., batch_size, seq_len, 1]
        intensity_all_events = softplus_ext(self.linear(history) + self.alpha * scaled_time, beta = F.softplus(self.beta))
                                                                               # [..., batch_size, seq_len, num_events]
        
        reshaped_aggregate_time = rearrange(time_history.cumsum(dim = -1), \
                                            f'... -> {"() " * (len(time_next.shape) - len(aggregate_time.shape))}... () ()')
                                                                               # [..., batch_size, seq_len, 1, 1]
        reshaped_aggregate_time = reshaped_aggregate_time + self.history_time_offset
                                                                               # [..., batch_size, seq_len, 1, 1]
        time_multiplier = torch.linspace(0, 1, self.integration_sample_rate, device = self.device)
        expanded_time = time_next.unsqueeze(dim = -1) * time_multiplier        # [..., batch_size, seq_len, integration_sample_rate]
        expanded_scaled_time = self.alpha * expanded_time.unsqueeze(dim = -1) / reshaped_aggregate_time
                                                                               # [..., batch_size, seq_len, integration_sample_rate, num_events]
        intensity_all_events_pre_softplus = self.linear(history)               # [..., batch_size, seq_len, num_events]
        intensity_all_events_pre_softplus = repeat(intensity_all_events_pre_softplus, '... ne -> ... r ne', r = self.integration_sample_rate)
                                                                               # [..., batch_size, seq_len, integration_sample_rate, num_events]
        all_lambda = softplus_ext(intensity_all_events_pre_softplus + expanded_scaled_time, F.softplus(self.beta))
                                                                               # [..., batch_size, seq_len, integration_sample_rate, num_events]
        integral_all_events = self.integration_estimator(all_lambda, expanded_time, self.integration_sample_rate)[..., -1, :]
                                                                               # [..., batch_size, seq_len, num_events]
        
        return integral_all_events, intensity_all_events


    def integral_intensity_time_next_2d(self, events_history, time_history, time_next, mask_history, integration_sample_rate, mean, var):
        assert len(time_next.shape) == 2, "Wrong input time tensor shape."

        history = self.history_encoder(time_history, events_history, mask_history)
                                                                               # [batch_size, seq_len, d_input]
        history = repeat(history, 'b s di -> b s 1 di')                        # [batch_size, seq_len, 1, d_input]

        aggregate_time = time_history.cumsum(dim = -1).unsqueeze(dim = -1)     # [batch_size, seq_len]
        # Avoid zero denominator
        aggregate_time = aggregate_time + self.history_time_offset             # [batch_size, seq_len]

        time_multiplier = torch.linspace(0, 1, integration_sample_rate, device = self.device)
        expanded_time = time_next.unsqueeze(dim = -1) * time_multiplier        # [batch_size, seq_len, integration_sample_rate]

        scaled_time = (expanded_time / aggregate_time).unsqueeze(dim = -1)     # [batch_size, seq_len, integration_sample_rate, 1]
        expanded_intensity_all_events = softplus_ext(self.linear(history) + self.alpha * scaled_time, beta = F.softplus(self.beta))
                                                                               # [batch_size, seq_len, integration_sample_rate, num_events]
        
        expanded_integral_all_events \
            = self.integration_estimator(expanded_intensity_all_events, expanded_time, integration_sample_rate)
                                                                               # [batch_size, seq_len, integration_sample_rate, num_events]
        # aggregated timestamp
        batch_size, seq_len, _ = expanded_time.shape
        timestamp = torch.cat(
            (torch.zeros((batch_size, seq_len, 1), device = self.device), expanded_time.diff(dim = -1)),
            dim = -1)                                                          # [batch_size, seq_len, integration_sample_rate]
        
        return expanded_integral_all_events, expanded_intensity_all_events, timestamp
        

    def integral_intensity_time_next_3d(self, events_history, time_history, time_next, mask_history, integration_sample_rate, mean, var):
        history = self.history_encoder(time_history, events_history, mask_history)
                                                                               # [batch_size, seq_len, d_input]

        # Intensity and integral estimation
        time_multiplier = torch.linspace(0, 1, integration_sample_rate, device = self.device)
                                                                               # [integration_sample_rate]
        original_expanded_time = time_next.unsqueeze(dim = -1) * time_multiplier
                                                                               # [..., batch_size, seq_len, num_event, integration_sample_rate]
        expanded_time = original_expanded_time.unsqueeze(dim = -1)             # [..., batch_size, seq_len, num_event, integration_sample_rate, 1]
        
        history = rearrange(history, f'... -> {"() " * (len(time_next.shape) - len(time_history.shape) - 1)}...')
                                                                               # [..., batch_size, seq_len, d_input]
        aggregate_time = rearrange(torch.cumsum(time_history, dim = -1), \
                                   f'... -> {"() " * (len(time_next.shape) - len(time_history.shape) - 1)}... () () ()')
                                                                               # [..., batch_size, seq_len, 1, 1, 1]
        aggregate_time = aggregate_time + self.history_time_offset             # [..., batch_size, seq_len, 1, 1, 1]
        scaled_expanded_time = expanded_time / aggregate_time                  # [..., batch_size, seq_len, num_event, integration_sample_rate, 1]

        intensity_for_each_event = self.linear(history)                        # [..., batch_size, seq_len, num_events]
        intensity_for_each_event = rearrange(intensity_for_each_event, '... ne -> ... () () ne')
                                                                               # [..., batch_size, seq_len, 1, 1, num_events]
        expanded_intensity_across_all_events = softplus_ext(self.alpha * scaled_expanded_time + intensity_for_each_event, F.softplus(self.beta))
                                                                               # [..., batch_size, seq_len, num_events, integration_sample_rate, num_events]
        expanded_integral_across_all_events \
            = self.integration_estimator(expanded_intensity_across_all_events, original_expanded_time, integration_sample_rate)
                                                                               # [..., batch_size, seq_len, num_events, integration_sample_rate, num_events]
        
        # aggregated timestamp
        timestamp = torch.cat(
            (torch.zeros((*original_expanded_time.shape[:-1], 1), device = self.device), original_expanded_time.diff(dim = -1)),
            dim = -1)                                                          # [..., batch_size, seq_len, num_events, integration_sample_rate]
        
        return expanded_integral_across_all_events, expanded_intensity_across_all_events, timestamp
    

    def model_probe_function(self, events_history, time_history, time_next, mask_history, mask_next, integration_sample_rate, mean, var):
        history = self.history_encoder(time_history, events_history, mask_history)
                                                                               # [batch_size, seq_len, d_input]
        history = repeat(history, 'b s di -> b s 1 di')                        # [batch_size, seq_len, 1, d_input]

        aggregate_time = time_history.cumsum(dim = -1).unsqueeze(dim = -1)     # [batch_size, seq_len]
        # Avoid zero denominator
        aggregate_time = aggregate_time + self.history_time_offset             # [batch_size, seq_len]

        time_multiplier = torch.linspace(0, 1, integration_sample_rate, device = self.device)
        expanded_time = time_next.unsqueeze(dim = -1) * time_multiplier        # [batch_size, seq_len, integration_sample_rate]

        scaled_time = (expanded_time / aggregate_time).unsqueeze(dim = -1)     # [batch_size, seq_len, integration_sample_rate, 1]
        expanded_intensity_all_events = softplus_ext(self.linear(history) + self.alpha * scaled_time, beta = F.softplus(self.beta))
                                                                               # [batch_size, seq_len, integration_sample_rate, num_events]

        expanded_integral_all_events \
            = self.integration_estimator(expanded_intensity_all_events, expanded_time, integration_sample_rate)
                                                                               # [batch_size, seq_len, integration_sample_rate, num_events]

        # aggregated timestamp
        batch_size, seq_len, _ = expanded_time.shape
        timestamp = torch.cat(
            (torch.zeros((batch_size, seq_len, 1), device = self.device), expanded_time.diff(dim = -1)),
            dim = -1)                                                          # [batch_size, seq_len, integration_sample_rate]
        
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
        for idx, (expand_intensity_per_seq, expand_integral_per_seq, mask_per_seq, time_next_per_seq) \
            in enumerate(zip(expand_intensity, expand_integral, mask_next, time_next)):
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
                                            resolution = integration_sample_rate, num_events = self.num_events,
                                            time_next = time_next_per_seq[:seq_len])
            spearman_matrix.append(spearman_matrix_per_seq)
            pearson_matrix.append(pearson_matrix_per_seq)
            L1_matrix.append(L1_matrix_per_seq)

        data['spearman_matrix'] = spearman_matrix
        data['pearson_matrix'] = pearson_matrix
        data['L1_matrix'] = L1_matrix
        
        return data, timestamp