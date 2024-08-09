import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce, pack
import numpy as np
from scipy.stats import spearmanr

from src.toolbox.misc import move_from_tensor_to_ndarray
from src.toolbox.metrics import L1_distance_across_events

import src.TPP.model.naive.naive_tpp as naive_tpp


class NaiveModule(nn.Module):
    def __init__(self, device, num_events, process_name):
        '''
        This is the Naive Module.
        In this module, we set up some simple MTPPs, like the homogenous possion, hawkes process, and self correct process to learn
        the MTPP from the dataset.
        They are some simple baselines.
        '''
        super(NaiveModule, self).__init__()
        self.num_events = num_events
        self.device = device
        self.naive_tpp = getattr(naive_tpp, process_name)(num_events = num_events, device = device)


    def forward(self, time_history, time_next, events_history):
        integral, intensity = self.naive_tpp(events_history, time_history, time_next)
                                                                               # [batch_size, seq_len, num_events]

        return integral, intensity


    def integral_intensity_time_next_2d(self, events_history, time_history, time_next, integration_sample_rate):
        time_multiplier = torch.linspace(0, 1, integration_sample_rate, device = self.device)
        expanded_time = time_next.unsqueeze(dim = -1) * time_multiplier        # [batch_size, seq_len, integration_sample_rate]

        expanded_integral_all_events, expanded_intensity_all_events = \
            self.naive_tpp.forward_time_next_2d(events_history, time_history, expanded_time, integration_sample_rate)
                                                                               # [batch_size, seq_len, integration_sample_rate]

        return expanded_integral_all_events, expanded_intensity_all_events, expanded_time


    def integral_intensity_time_next_3d(self, events_history, time_history, time_next, integration_sample_rate):
        time_multiplier = torch.linspace(0, 1, integration_sample_rate, device = self.device)
        expanded_time = time_next.unsqueeze(dim = -1) * time_multiplier        # [..., batch_size, seq_len, num_events, integration_sample_rate]
        
        expanded_integral_all_events, expanded_intensity_all_events = \
            self.naive_tpp.forward_time_next_3d(events_history, time_history, expanded_time, integration_sample_rate)
                                                                               # [..., batch_size, seq_len, num_events, integration_sample_rate, num_events]

        return expanded_integral_all_events, expanded_intensity_all_events, expanded_time


    def model_probe_function(self, events_history, time_history, time_next, mask_next, integration_sample_rate):
        time_multiplier = torch.linspace(0, 1, integration_sample_rate, device = self.device)
        expanded_time = time_next.unsqueeze(dim = -1) * time_multiplier        # [batch_size, seq_len, integration_sample_rate]

        expanded_integral_all_events, expanded_intensity_all_events = \
            self.naive_tpp.forward_time_next_2d(events_history, time_history, expanded_time, integration_sample_rate)
                                                                               # [batch_size, seq_len, integration_sample_rate]
        
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