import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from scipy.stats import spearmanr

from src.toolbox.metrics import L1_distance_across_marks
from src.toolbox.misc import move_from_tensor_to_ndarray

from . import naive_tpp


class NaiveModule(nn.Module):
    def __init__(self, device, num_marks, process_name):
        """
        This function creates a NaiveMTPP model.

        ### Args
            * ```str``` process_name
              Which classic MTPP process do you want?
              Available classic MTPPs: 1. Poisson process 2. Hawkes process.
            * ```torch.device``` device
              Running models on GPU or CPU?
        """
        super().__init__()
        self.num_marks = num_marks
        self.device = device
        self.naive_tpp = getattr(naive_tpp, process_name)(num_marks=num_marks, device=device)

    def forward(self, time_history, time_next, marks_history):
        """
        SAHP's forwardpropagation function for training.

        ### Args
            * ```torch.tensor``` marks_history
              shape: ```[batch_size, seq_len]```
              Historical mark sequence.
            * ```torch.tensor``` time_history
              shape: ```[batch_size, seq_len]```
              Historical time sequence.
            * ```torch.tensor``` time_next
              shape: ```[..., batch_size, seq_len]```
              Guessed or real time when the next mark will happen.
        ### Outputs
            * ```torch.tensor``` integral
              shape: ```[..., batch_size, seq_len, num_marks]```
              The value of \\Lambda^*(m, t) on [t_{i-1}, t_i).
            * ```torch.tensor``` intensity
              shape: ```[..., batch_size, seq_len, num_marks]```
              The value of \\lambda^*(m, t) on at t_i.
        """
        integral, intensity = self.naive_tpp(time_history, time_next, marks_history)
        # [batch_size, seq_len, num_marks]

        return integral, intensity

    def integral_intensity_time_next_2d(self, time_history, time_next, marks_history, integration_sample_rate):
        """
        Probe the value of the intensity function and its integral at sampled timestamps.
        In this function, all marks share the sampled timestmaps, so the dimension of time_next does not include num_mark.

        ### Args
            * ```torch.tensor``` marks_history
              shape: ```[batch_size, seq_len]```
              Historical mark sequence.
            * ```torch.tensor``` time_history
              shape: ```[batch_size, seq_len]```
              Historical time sequence.
            * ```torch.tensor``` time_next
              shape: ```[..., batch_size, seq_len]```
              Guessed or real time when the next mark will happen.
            * ```int``` integration_sample_rate
              The number of interpolated points in a time interval between two adjoint marks for integration estimation.
              The number of interpolated points counts the start and end point of the interval.
        ### Outputs
            * ```torch.tensor``` expanded_integral_all_marks
              shape: ```[..., batch_size, seq_len, resolution, num_marks]```
              The value of \\Lambda^*(m, t) at sampled times.
            * ```torch.tensor``` expanded_intensity_all_marks
              shape: ```[..., batch_size, seq_len, resolution, num_marks]```
              The value of \\lambda^*(m, t) at sampled times.
            * ```torch.tensor``` expanded_time
              shape: ```[..., batch_size, seq_len, resolution]```
              The value of sampled times.
        """
        time_multiplier = torch.linspace(0, 1, integration_sample_rate, device=self.device)
        expanded_time = time_next.unsqueeze(dim=-1) * time_multiplier  # [batch_size, seq_len, integration_sample_rate]

        expanded_integral_all_marks, expanded_intensity_all_marks = self.naive_tpp.forward_time_next_2d(
            time_history, expanded_time, marks_history, integration_sample_rate
        )
        # [batch_size, seq_len, integration_sample_rate]

        return expanded_integral_all_marks, expanded_intensity_all_marks, expanded_time

    def integral_intensity_time_next_3d(self, time_history, time_next, marks_history, integration_sample_rate):
        """
        Probe the value of the intensity function and its integral at sampled timestamps.
        In this function, all marks can have their sampled timestmaps, so the dimension of time_next is ```[..., batch_size, seq_len, num_marks]```.
        This function is supposed to be much slower than integral_intensity_time_next_2d().

        ### Args
            * ```torch.tensor``` marks_history
              shape: ```[batch_size, seq_len]```
              Historical mark sequence.
            * ```torch.tensor``` time_history
              shape: ```[batch_size, seq_len]```
              Historical time sequence.
            * ```torch.tensor``` time_next
              shape: ```[..., batch_size, seq_len, num_marks]```
              Guessed or real time when the next mark will happen.
            * ```int``` integration_sample_rate
              The number of interpolated points in a time interval between two adjoint marks for integration estimation.
              The number of interpolated points counts the start and end point of the interval.
        ### Outputs
            * ```torch.tensor``` expanded_integral_all_marks
              shape: ```[..., batch_size, seq_len, resolution, num_marks]```
              The value of \\Lambda^*(m, t) at sampled times.
            * ```torch.tensor``` expanded_intensity_all_marks
              shape: ```[..., batch_size, seq_len, resolution, num_marks]```
              The value of \\lambda^*(m, t) at sampled times.
            * ```torch.tensor``` expanded_time
              shape: ```[..., batch_size, seq_len, resolution]```
              The value of sampled times.
        """
        time_multiplier = torch.linspace(0, 1, integration_sample_rate, device=self.device)
        expanded_time = (
            time_next.unsqueeze(dim=-1) * time_multiplier
        )  # [..., batch_size, seq_len, num_marks, integration_sample_rate]

        expanded_integral_all_marks, expanded_intensity_all_marks = self.naive_tpp.forward_time_next_3d(
            time_history, expanded_time, marks_history, integration_sample_rate
        )
        # [..., batch_size, seq_len, num_marks, integration_sample_rate, num_marks]

        return expanded_integral_all_marks, expanded_intensity_all_marks, expanded_time

    def model_probe_function(self, time_history, time_next, marks_history, mask_next, integration_sample_rate):
        """
        Probe the value of the intensity function and its integral at sampled timestamps.
        In this function, all marks can have their sampled timestmaps, so the dimension of time_next is ```[..., batch_size, seq_len, num_marks]```.
        This function is supposed to be much slower than integral_intensity_time_next_2d().

        ### Args
            * ```torch.tensor``` marks_history
              shape: ```[batch_size, seq_len]```
              Historical mark sequence.
            * ```torch.tensor``` time_history
              shape: ```[batch_size, seq_len]```
              Historical time sequence.
            * ```torch.tensor``` time_next
              shape: ```[..., batch_size, seq_len]```
              Guessed or real time when the next mark will happen.
            * ```torch.tensor``` mask_next
              shape: ```[..., batch_size, seq_len]```
              Tell which mark in *_next is the real mark so should be considered in metric calculation.
            * ```int``` integration_sample_rate
              The number of interpolated points in a time interval between two adjoint marks for integration estimation.
              The number of interpolated points counts the start and end point of the interval.
        ### Outputs
            * ```dict``` data
              Probed data used for plot drawing.
            * ```torch.tensor``` expanded_time
              shape: ```[batch_size, seq_len, resolution]```
              The value of sampled times.
        """
        time_multiplier = torch.linspace(0, 1, integration_sample_rate, device=self.device)
        expanded_time = time_next.unsqueeze(dim=-1) * time_multiplier  # [batch_size, seq_len, integration_sample_rate]

        expanded_integral_all_marks, expanded_intensity_all_marks = self.naive_tpp.forward_time_next_2d(
            time_history, expanded_time, marks_history, integration_sample_rate
        )
        # [batch_size, seq_len, integration_sample_rate]
        # construct the plot dict
        data = {}
        data["expand_intensity_for_each_mark"] = (
            expanded_intensity_all_marks  # [batch_size, seq_len, integration_sample_rate, num_marks]
        )
        data["expand_integral_for_each_mark"] = (
            expanded_integral_all_marks  # [batch_size, seq_len, integration_sample_rate, num_marks]
        )

        expand_intensity = rearrange(expanded_intensity_all_marks, "b s r ne -> b (s r) ne")
        # [batch_size, seq_len * integration_sample_rate, num_mark]
        expand_integral = rearrange(expanded_integral_all_marks, "b s r ne -> b (s r) ne")
        # [batch_size, seq_len * integration_sample_rate, num_mark]
        spearman_matrix = []
        pearson_matrix = []
        l1_matrix = []
        for idx, (expand_intensity_per_seq, expand_integral_per_seq, mask_per_seq, expanded_time_per_seq) in enumerate(
            zip(expand_intensity, expand_integral, mask_next, expanded_time)
        ):
            seq_len = mask_per_seq.sum()
            probability_distribution = expand_intensity_per_seq * torch.exp(-expand_integral_per_seq)
            probability_distribution = move_from_tensor_to_ndarray(probability_distribution)

            # rho: spearman coefficient
            if self.num_marks == 1:
                spearman_matrix_per_seq = np.array([[1.0]])
            else:
                spearman_matrix_per_seq = spearmanr(probability_distribution[: seq_len * integration_sample_rate])[0]
                if self.num_marks == 2:
                    spearman_matrix_per_seq = np.array([[1, spearman_matrix_per_seq], [spearman_matrix_per_seq, 1]])

            # r: pearson coefficient
            pearson_matrix_per_seq = np.corrcoef(
                probability_distribution[: seq_len * integration_sample_rate], rowvar=False
            )
            if self.num_marks == 1:
                pearson_matrix_per_seq = rearrange(np.array(pearson_matrix_per_seq), " -> () ()")

            # L^1 metric
            l1_matrix_per_seq = L1_distance_across_marks(
                probability_distribution[: seq_len * integration_sample_rate],
                time_next=expanded_time_per_seq[:seq_len],
                has_flatten=True,
            )
            spearman_matrix.append(spearman_matrix_per_seq)
            pearson_matrix.append(pearson_matrix_per_seq)
            l1_matrix.append(l1_matrix_per_seq)

        data["spearman_matrix"] = spearman_matrix
        data["pearson_matrix"] = pearson_matrix
        data["L1_matrix"] = l1_matrix
        data["model_parameter"] = self.naive_tpp.get_model_parameter()

        return data, expanded_time
