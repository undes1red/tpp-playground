import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce, pack
import numpy as np
from scipy.stats import spearmanr

from src.toolbox.misc import move_from_tensor_to_ndarray
from src.toolbox.metrics import L1_distance_across_marks

from src.toolbox.algorithms import approximate_integration
from .transformers import TransformerTPP


class AttNHP(nn.Module):
    def __init__(
        self,
        training,
        device,
        num_marks,
        d_input,
        d_hidden,
        n_layers,
        n_head,
        d_qkv,
        dropout,
        integration_sample_rate,
    ):
        super().__init__()
        self.num_marks = num_marks
        self.device = device
        self.integration_sample_rate = integration_sample_rate
        self.training = training

        # This layer translates decayed hidden states into intensity function values.
        self.intensity_layer = nn.Sequential(
            nn.Linear(d_input, 1, bias=True, device=self.device), nn.Softplus(beta=1.0)
        )

        # History encoder. AttNHP employs a plain transformer to encode every marks.
        self.attn_model = TransformerTPP(
            training=training,
            num_marks=num_marks,
            device=self.device,
            d_input=d_input,
            d_hidden=d_hidden,
            n_layers=n_layers,
            n_head=n_head,
            d_qkv=d_qkv,
            dropout=dropout,
        )

    def forward(
        self,
        time_history,
        time_next,
        marks_history,
        mask_history,
        custom_marks_history=False,
        num_dimension_prior_batch=0,
    ):
        # calculate the integral
        time_multiplier = torch.linspace(0, 1, self.integration_sample_rate, device=self.device)
        expanded_time = (
            time_next.unsqueeze(dim=-1) * time_multiplier
        )  # [..., batch_size, seq_len, integration_sample_rate]

        hidden_state = self.attn_model(
            time_history,
            expanded_time,
            marks_history,
            mask_history,
            custom_marks_history=custom_marks_history,
            integration_sample_rate=self.integration_sample_rate,
        )
        # [..., integration_sample_rate, num_mark, batch_size, seq_len * 2, d_input]
        _, hidden_state_all_marks_at_expanded_time = hidden_state.chunk(2, dim=-2)
        # [..., integration_sample_rate, num_mark, batch_size, seq_len, d_input]
        intensity_all_marks = self.intensity_layer(hidden_state_all_marks_at_expanded_time)
        # [..., integration_sample_rate, num_mark, batch_size, seq_len, 1]
        # Rearrage the intensity tensor.
        intensity_all_marks = rearrange(intensity_all_marks, "... isr ne bs sl () -> ... bs sl isr ne")
        # [..., batch_size, seq_len, integration_sample_rate, num_mark]

        integral_all_marks = approximate_integration(intensity_all_marks, expanded_time, dim=-2, only_integral=True)
        # [..., batch_size, seq_len, num_marks]

        return integral_all_marks, intensity_all_marks[..., -1, :]

    def sample_for_tm(
        self,
        time_history,
        time_next,
        marks_history,
        mask_history,
        custom_marks_history=False,
        num_dimension_prior_batch=0,
    ):
        # calculate the integral
        time_multiplier = torch.linspace(0, 1, self.integration_sample_rate, device=self.device)
        expanded_time = (
            time_next.unsqueeze(dim=-1) * time_multiplier
        )  # [number_of_sampled_sequences, seq_len, integration_sample_rate]

        hidden_state = self.attn_model(time_history, expanded_time, marks_history, mask_history)
        # [integration_sample_rate, num_mark, number_of_sampled_sequences, seq_len * 2, d_input]
        _, hidden_state_all_marks_at_expanded_time = hidden_state.chunk(2, dim=-2)
        # [integration_sample_rate, num_mark, number_of_sampled_sequences, seq_len, d_input]
        intensity_all_marks = self.intensity_layer(hidden_state_all_marks_at_expanded_time)
        # [integration_sample_rate, num_mark, number_of_sampled_sequences, seq_len, 1]
        # Rearrage the intensity tensor.
        intensity_all_marks = rearrange(intensity_all_marks, "isr ne nss sl () -> nss sl isr ne")
        # [number_of_sampled_sequences, seq_len, integration_sample_rate, num_mark]

        integral_all_marks = approximate_integration(intensity_all_marks, expanded_time, dim=-2, only_integral=True)
        # [number_of_sampled_sequences, seq_len, num_marks]

        return integral_all_marks, intensity_all_marks[..., -1, :]

    def sample_for_mt(
        self,
        time_history,
        time_next,
        marks_history,
        mask_history,
        custom_marks_history=False,
        num_dimension_prior_batch=0,
    ):
        # calculate the integral
        time_multiplier = torch.linspace(0, 1, self.integration_sample_rate, device=self.device)
        expanded_time = (
            time_next.unsqueeze(dim=-1) * time_multiplier
        )  # [..., batch_size, seq_len, integration_sample_rate]

        hidden_state = self.attn_model(time_history, expanded_time, marks_history, mask_history)
        # [..., integration_sample_rate, num_mark, batch_size, seq_len * 2, d_input]
        _, hidden_state_all_marks_at_expanded_time = hidden_state.chunk(2, dim=-2)
        # [..., integration_sample_rate, num_mark, batch_size, seq_len, d_input]
        intensity_all_marks = self.intensity_layer(hidden_state_all_marks_at_expanded_time)
        # [..., integration_sample_rate, num_mark, batch_size, seq_len, 1]
        # Rearrage the intensity tensor.
        intensity_all_marks = rearrange(intensity_all_marks, "... isr ne bs sl () -> ... bs sl isr ne")
        # [..., batch_size, seq_len, integration_sample_rate, num_mark]

        integral_all_marks = approximate_integration(intensity_all_marks, expanded_time, dim=-2, only_integral=True)
        # [..., batch_size, seq_len, num_marks]

        return integral_all_marks, intensity_all_marks[..., -1, :]

    def get_mark_embedding(self, input_mark):
        return self.history_encoder.get_mark_embedding(input_mark)  # [batch_size, seq_len, d_history]

    def integral_intensity_time_next_2d(
        self, marks_history, time_history, time_next, mask_history, integration_sample_rate, time_next_start=None
    ):
        if time_next_start is None:
            time_next_start = torch.zeros_like(time_next)  # [..., batch_size, seq_len]

        # calculate the integral
        time_multiplier = torch.linspace(0, 1, integration_sample_rate, device=self.device)
        expanded_time = (time_next - time_next_start).unsqueeze(dim=-1) * time_multiplier + time_next_start.unsqueeze(
            dim=-1
        )
        # [..., batch_size, seq_len, integration_sample_rate]

        hidden_state = self.attn_model(time_history, expanded_time, marks_history, mask_history)
        # [..., integration_sample_rate, num_mark, batch_size, seq_len * 2, d_input]
        _, hidden_state_all_marks_at_expanded_time = hidden_state.chunk(2, dim=-2)
        # [..., integration_sample_rate, num_mark, batch_size, seq_len, d_input]
        intensity_all_marks = self.intensity_layer(hidden_state_all_marks_at_expanded_time)
        # [..., integration_sample_rate, num_mark, batch_size, seq_len, 1]
        # Rearrage the intensity tensor.
        intensity_all_marks = rearrange(intensity_all_marks, "... isr ne bs sl () -> ... bs sl isr ne")
        # [..., batch_size, seq_len, integration_sample_rate, num_mark]

        integral_all_marks = approximate_integration(intensity_all_marks, expanded_time, dim=-2)
        # [..., batch_size, seq_len, integration_sample_rate, num_marks]

        return integral_all_marks, intensity_all_marks, expanded_time

    def integral_intensity_time_next_3d(
        self,
        marks_history,
        time_history,
        time_next,
        mask_history,
        integration_sample_rate,
        num_dimension_prior_batch=0,
    ):
        # calculate the integral
        time_multiplier = torch.linspace(0, 1, integration_sample_rate, device=self.device)
        original_expanded_time = time_next.unsqueeze(dim=-1) * time_multiplier
        # [..., batch_size, seq_len, num_mark, integration_sample_rate]
        expanded_time = rearrange(original_expanded_time, "... b s ne isr -> ... ne b s isr")
        # [..., num_mark, batch_size, seq_len, integration_sample_rate]
        hidden_state = self.attn_model(
            time_history, expanded_time, marks_history, mask_history, sample_time_with_mark=True
        )
        # [..., num_mark, integration_sample_rate, num_mark, batch_size, seq_len * 2, d_input]
        _, hidden_state_all_marks_at_expanded_time = hidden_state.chunk(2, dim=-2)
        # [..., num_mark, integration_sample_rate, num_mark, batch_size, seq_len, d_input]
        intensity_all_marks = self.intensity_layer(hidden_state_all_marks_at_expanded_time)
        # [..., num_mark, integration_sample_rate, num_mark, batch_size, seq_len, 1]
        # Rearrage the intensity tensor.
        intensity_all_marks = rearrange(intensity_all_marks, "... ne isr ne1 bs sl () -> ... bs sl ne isr ne1")
        # [..., batch_size, seq_len, num_mark, integration_sample_rate, num_mark]
        integral_all_marks = approximate_integration(intensity_all_marks, original_expanded_time, dim=-2)
        # [..., batch_size, seq_len, num_marks, integration_sample_rate, num_mark]

        return integral_all_marks, intensity_all_marks, expanded_time

    def model_probe_function(
        self, marks_history, time_history, time_next, mask_history, mask_next, integration_sample_rate
    ):
        # calculate the integral
        time_multiplier = torch.linspace(0, 1, integration_sample_rate, device=self.device)
        expanded_time = time_next.unsqueeze(dim=-1) * time_multiplier  # [batch_size, seq_len, integration_sample_rate]

        hidden_state = self.attn_model(time_history, expanded_time, marks_history, mask_history)
        # [integration_sample_rate, num_mark, batch_size, seq_len * 2, d_input]
        _, hidden_state_all_marks_at_expanded_time = hidden_state.chunk(2, dim=-2)
        # [integration_sample_rate, num_mark, batch_size, seq_len, d_input]
        intensity_all_marks = self.intensity_layer(hidden_state_all_marks_at_expanded_time)
        # [integration_sample_rate, num_mark, batch_size, seq_len, 1]
        # Rearrage the intensity tensor.
        expanded_intensity_all_marks = rearrange(intensity_all_marks, "... isr ne bs sl () -> ... bs sl isr ne")
        # [batch_size, seq_len, integration_sample_rate, num_mark]

        expanded_integral_all_marks = approximate_integration(intensity_all_marks, expanded_time, dim=-2)
        # [batch_size, seq_len, integration_sample_rate, num_marks]

        # construct the plot dict
        data = {}
        data["expand_intensity_for_each_mark"] = (
            expanded_intensity_all_marks  # [batch_size, seq_len, integration_sample_rate, num_marks]
        )
        data["expand_integral_for_each_mark"] = (
            expanded_integral_all_marks  # [batch_size, seq_len, integration_sample_rate, num_marks]
        )

        # THP always assumes that the mark information is present.
        # So model_probe_function() always provides spearman, pearson coefficient and L1 distance.

        expand_intensity = rearrange(expanded_intensity_all_marks, "b s r ne -> b (s r) ne")
        # [batch_size, seq_len * integration_sample_rate, num_mark]
        expand_integral = rearrange(expanded_integral_all_marks, "b s r ne -> b (s r) ne")
        # [batch_size, seq_len * integration_sample_rate, num_mark]

        spearman_matrix = []
        pearson_matrix = []
        L1_matrix = []
        for idx, (expand_intensity_per_seq, expand_integral_per_seq, mask_per_seq, expanded_time_per_seq) in enumerate(
            zip(expand_intensity, expand_integral, mask_next, expanded_time)
        ):
            seq_len = mask_per_seq.sum()
            probability_distribution = expand_intensity_per_seq * torch.exp(-expand_integral_per_seq)
            probability_distribution = move_from_tensor_to_ndarray(probability_distribution)

            # rho: spearman coefficient
            if self.num_marks == 1:
                spearman_matrix_per_seq = np.array(
                    [
                        [
                            1.0,
                        ],
                    ]
                )
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
            L1_matrix_per_seq = L1_distance_across_marks(
                probability_distribution[: seq_len * integration_sample_rate],
                time_next=expanded_time_per_seq[:seq_len],
                has_flatten=True,
            )
            spearman_matrix.append(spearman_matrix_per_seq)
            pearson_matrix.append(pearson_matrix_per_seq)
            L1_matrix.append(L1_matrix_per_seq)

        data["spearman_matrix"] = spearman_matrix
        data["pearson_matrix"] = pearson_matrix
        data["L1_matrix"] = L1_matrix

        return data, expanded_time
