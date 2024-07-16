import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce, pack
import numpy as np
from scipy.stats import spearmanr

from src.TPP.model.utils import approximate_integration, L1_distance_across_events, move_from_tensor_to_ndarray
from src.TPP.model.mhp.utils import softplus_ext
from src.TPP.model.mhp.mamba import mamba
from mamba_ssm import Mamba
from mamba_ssm.models.mixer_seq_simple import create_block
from mamba_ssm.ops.triton.layer_norm import layer_norm_fn, RMSNorm


class MHP(nn.Module):
    def __init__(self, device, num_events, d_input, d_mamba, n_layers,
                 dropout, kernel_size, expand, beta, mode, integration_sample_rate):
        super(MHP, self).__init__()
        self.device = device
        self.num_events = num_events
        self.integration_sample_rate = integration_sample_rate
        self.mode = mode

        # parameter for the weight of time difference
        self.alpha = nn.Parameter(torch.ones((self.num_events), dtype = torch.float32, \
                                  device = self.device, requires_grad = True))
        nn.init.normal_(self.alpha)

        # parameter for the softplus function
        self.beta = nn.Parameter(torch.ones((self.num_events), dtype = torch.float32, \
                                  device = self.device, requires_grad = True) * beta)
        nn.init.normal_(self.beta)

        # convert hidden vectors into valid intensity function values.
        self.linear = nn.Linear(d_input, num_events, device = self.device)

        # SSM+Selection (S6) history encoder
        self.event_embedding = nn.Embedding(num_embeddings = num_events + 1, embedding_dim = d_input,\
                                            padding_idx = num_events, device = self.device)
        self.weight_for_t = nn.Parameter(torch.zeros((1, d_input), device = self.device, requires_grad = True))
        
        if mode == 'pure_mamba':
            self.mamba_encoder = nn.ModuleList(
                [
                    Mamba(d_model = d_input, d_state = d_mamba, d_conv = kernel_size, \
                          expand = expand, device = self.device, layer_idx = layer_idx) for layer_idx in range(n_layers)
                ]
            )
            self.dropout = nn.Dropout(dropout)
        elif mode == 'mamba_block':
            self.mamba_encoder = nn.ModuleList(
                [
                    create_block(d_model = d_input, d_intermediate = d_input, 
                                 ssm_cfg = {'layer': 'Mamba1', 'd_state': d_mamba, 'd_conv': kernel_size, 'expand': expand},
                                 device = self.device, layer_idx = layer_idx) for layer_idx in range(n_layers)
                ]
            )
            self.norm_f = nn.LayerNorm(d_input, eps = 1e-5, device = self.device)
        else:
            # This is what the paper has claimed, however it does not work properly.
            # One possible reason is the delta time is too big for mamba to handle.
            self.mamba_encoder = mamba(num_events, d_input, d_mamba, kernel_size, n_layers, dropout, expand, device = self.device)


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


    def forward(self, time_history, time_next, events_history, mean, std):
        if self.mode == 'pure_mamba':
            events_vec = self.event_embedding(events_history)                  # [batch_size, seq_len, d_input]
            time_history = (time_history - mean) / std
            time_embedding = time_history.unsqueeze(dim = -1) * self.weight_for_t
                                                                               # [batch_size, seq_len, d_input]
            input_vecs = events_vec + time_embedding                           # [batch_size, seq_len, d_input]
    
            for mamba_layer in self.mamba_encoder:
                output_state = mamba_layer(input_vecs)                         # [batch_size, seq_len, d_input]
                output_state = self.dropout(output_state)                      # [batch_size, seq_len, d_input]
        elif self.mode == 'mamba_block':
            events_vec = self.event_embedding(events_history)                  # [batch_size, seq_len, d_input]
            time_history = (time_history - mean) / std
            time_embedding = time_history.unsqueeze(dim = -1) * self.weight_for_t
                                                                               # [batch_size, seq_len, d_input]
            output_state = events_vec + time_embedding                         # [batch_size, seq_len, d_input]

            residual = None
            for layer in self.mamba_encoder:
                output_state, residual = layer(output_state, residual)         # [batch_size, seq_len, d_input]
            # Set prenorm = False here since we don't need the residual
            output_state = layer_norm_fn(
                output_state,
                self.norm_f.weight,
                self.norm_f.bias,
                eps = self.norm_f.eps,
                residual = residual,
                prenorm = False,
                residual_in_fp32 = False,
                is_rms_norm = isinstance(self.norm_f, RMSNorm)
            )                                                                  # [batch_size, seq_len, d_input]
        else:
            output_state = self.mamba_encoder(events_history, time_history, mean, std)
                                                                               # [batch_size, seq_len, d_input]
        output_state = rearrange(output_state, f'... -> {"() " * (len(time_next.shape) - len(time_history.shape))}...')
                                                                               # [..., batch_size, seq_len, d_input]

        scaled_time = time_next.unsqueeze(dim = -1)                            # [..., batch_size, seq_len, 1]
        intensity_all_events = softplus_ext(self.linear(output_state) + self.alpha * scaled_time, beta = F.softplus(self.beta))
                                                                               # [..., batch_size, seq_len, num_events]
        time_multiplier = torch.linspace(0, 1, self.integration_sample_rate, device = self.device)
        expanded_time = time_next.unsqueeze(dim = -1) * time_multiplier        # [..., batch_size, seq_len, integration_sample_rate]
        expanded_scaled_time = self.alpha * expanded_time.unsqueeze(dim = -1)
                                                                               # [..., batch_size, seq_len, integration_sample_rate, num_events]
        intensity_all_events_pre_softplus = self.linear(output_state)          # [..., batch_size, seq_len, num_events]
        intensity_all_events_pre_softplus = repeat(intensity_all_events_pre_softplus, '... ne -> ... r ne', r = self.integration_sample_rate)
                                                                               # [..., batch_size, seq_len, integration_sample_rate, num_events]
        all_lambda = softplus_ext(intensity_all_events_pre_softplus + expanded_scaled_time, F.softplus(self.beta))
                                                                               # [..., batch_size, seq_len, integration_sample_rate, num_events]
        integral_all_events = approximate_integration(all_lambda, expanded_time, dim = -2, only_integral = True)
                                                                               # [..., batch_size, seq_len, num_events]
        
        return integral_all_events, intensity_all_events


    def integral_intensity_time_next_2d(self, events_history, time_history, time_next, integration_sample_rate, mean, std):
        assert len(time_next.shape) == 2, "Wrong input time tensor shape."

        output_state = self.mamba_encoder(events_history, time_history, mean, std)
                                                                               # [batch_size, seq_len, d_input]
        output_state = repeat(output_state, 'b s di -> b s 1 di')              # [batch_size, seq_len, 1, d_input]

        time_multiplier = torch.linspace(0, 1, integration_sample_rate, device = self.device)
        expanded_time = time_next.unsqueeze(dim = -1) * time_multiplier        # [batch_size, seq_len, integration_sample_rate]

        expanded_time = expanded_time.unsqueeze(dim = -1)     # [batch_size, seq_len, integration_sample_rate, 1]
        expanded_intensity_all_events = softplus_ext(self.linear(output_state) + self.alpha * expanded_time, beta = F.softplus(self.beta))
                                                                               # [batch_size, seq_len, integration_sample_rate, num_events]
        expanded_integral_all_events \
            = approximate_integration(expanded_intensity_all_events, expanded_time, dim = -2)
                                                                               # [batch_size, seq_len, integration_sample_rate, num_events]

        return expanded_integral_all_events, expanded_intensity_all_events, expanded_time
        

    def integral_intensity_time_next_3d(self, events_history, time_history, time_next, integration_sample_rate, mean, std):
        assert len(time_next.shape) == 3, "Wrong input time tensor shape."
        output_state = self.mamba_encoder(events_history, time_history, mean, std)
                                                                               # [batch_size, seq_len, d_input]
        # Intensity and integral estimation
        time_multiplier = torch.linspace(0, 1, integration_sample_rate, device = self.device)
                                                                               # [integration_sample_rate]
        original_expanded_time = time_next.unsqueeze(dim = -1) * time_multiplier
                                                                               # [..., batch_size, seq_len, num_event, integration_sample_rate]
        expanded_time = original_expanded_time.unsqueeze(dim = -1)             # [..., batch_size, seq_len, num_event, integration_sample_rate, 1]
        
        output_state = rearrange(output_state, f'... -> {"() " * (len(time_next.shape) - len(time_history.shape) - 1)}...')
                                                                               # [..., batch_size, seq_len, d_input]
        intensity_for_each_event = self.linear(output_state)                   # [..., batch_size, seq_len, num_events]
        intensity_for_each_event = rearrange(intensity_for_each_event, '... ne -> ... () () ne')
                                                                               # [..., batch_size, seq_len, 1, 1, num_events]
        expanded_intensity_across_all_events = softplus_ext(self.alpha * expanded_time + intensity_for_each_event, F.softplus(self.beta))
                                                                               # [..., batch_size, seq_len, num_events, integration_sample_rate, num_events]
                                                                               # [..., batch_size, seq_len, num_events, integration_sample_rate, num_events]
        expanded_integral_across_all_events \
            = approximate_integration(expanded_intensity_across_all_events, original_expanded_time, dim = -2)
                                                                               # [..., batch_size, seq_len, num_events, integration_sample_rate, num_events]

        return expanded_integral_across_all_events, expanded_intensity_across_all_events, original_expanded_time
    

    def model_probe_function(self, events_history, time_history, time_next, mask_next, integration_sample_rate, mean, std):
        output_state = self.mamba_encoder(events_history, time_history, mean, std)
                                                                               # [batch_size, seq_len, d_input]
        output_state = repeat(output_state, 'b s di -> b s 1 di')              # [batch_size, seq_len, 1, d_input]

        time_multiplier = torch.linspace(0, 1, integration_sample_rate, device = self.device)
        expanded_time = time_next.unsqueeze(dim = -1) * time_multiplier        # [batch_size, seq_len, integration_sample_rate]

        expanded_intensity_all_events = softplus_ext(self.linear(output_state) + self.alpha * expanded_time, beta = F.softplus(self.beta))
                                                                               # [batch_size, seq_len, integration_sample_rate, num_events]
        expanded_integral_all_events \
            = approximate_integration(expanded_intensity_all_events, expanded_time, dim = -2)
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
        for _, (expand_intensity_per_seq, expand_integral_per_seq, mask_per_seq, time_next_per_seq) \
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