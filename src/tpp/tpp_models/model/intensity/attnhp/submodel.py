import torch
import torch.nn as nn

from src.toolbox.algorithms import approximate_integration, evaluate_on_one_batch
from src.toolbox.misc import move_from_tensor_to_ndarray

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
        resolution,
    ):
        super().__init__()
        self.num_marks = num_marks
        self.device = device
        self.resolution = resolution
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
        mask_next,
        custom_marks_history=False,
    ):
        # calculate the integral
        time_multiplier = torch.linspace(0, 1, self.resolution, device=self.device)
        expanded_time = time_next.unsqueeze(dim=-1) * time_multiplier  # [..., batch_size, seq_len, resolution]

        hidden_state = self.attn_model(
            time_history,
            expanded_time,
            marks_history,
            mask_history,
            mask_next,
            custom_marks_history=custom_marks_history,
        )
        # [..., batch_size, seq_len, resolution, num_marks, d_input]

        intensity_all_marks = self.intensity_layer(hidden_state).squeeze(dim=-1)
        # [..., batch_size, seq_len, resolution, num_marks]

        integral_all_marks = approximate_integration(intensity_all_marks, expanded_time, dim=-2, only_integral=True)
        # [..., batch_size, seq_len, num_marks]

        return integral_all_marks, intensity_all_marks[..., -1, :]

    def get_mark_embedding(self, input_mark):
        return self.history_encoder.get_mark_embedding(input_mark)  # [batch_size, seq_len, d_history]

    def integral_intensity_time_next_2d(
        self,
        time_history,
        time_next,
        marks_history,
        mask_history,
        mask_next,
        resolution,
        time_next_start=None,
        time_next_with_resolution_dim=False,
    ):
        if time_next_start is None:
            time_next_start = torch.zeros_like(time_next)  # [..., batch_size, seq_len]

        if time_next_with_resolution_dim:
            expanded_time = time_next
            # [..., batch_size, seq_len, resolution]
        else:
            time_multiplier = torch.linspace(0, 1, resolution, device=self.device)
            expanded_time = (time_next - time_next_start).unsqueeze(
                dim=-1
            ) * time_multiplier + time_next_start.unsqueeze(dim=-1)
            # [..., batch_size, seq_len, resolution]

        hidden_state = self.attn_model(time_history, expanded_time, marks_history, mask_history, mask_next)
        # [..., batch_size, seq_len, resolution, num_marks, d_input]

        intensity_all_marks = self.intensity_layer(hidden_state).squeeze(dim=-1)
        # [..., batch_size, seq_len, resolution, num_marks]
        integral_all_marks = approximate_integration(intensity_all_marks, expanded_time, dim=-2)
        # [..., batch_size, seq_len, resolution, num_marks]

        # integral offset
        if expanded_time[..., 0].max() == 0:
            integral_all_marks_from_zero_to_interval_start = torch.zeros_like(integral_all_marks)
        else:
            time_multiplier = torch.linspace(0, 1, resolution, device=self.device)
            zero_to_interval_start_time = (
                expanded_time[..., 0].unsqueeze(dim=-1) * time_multiplier
            )  # [..., batch_size, seq_len, resolution]
            hidden_state = self.attn_model(
                time_history, zero_to_interval_start_time, marks_history, mask_history, mask_next
            )
            # [..., batch_size, seq_len, resolution, num_marks, d_input]

            intensity_all_marks = self.intensity_layer(hidden_state).squeeze(dim=-1)
            # [..., batch_size, seq_len, resolution, num_marks]
            integral_all_marks = approximate_integration(intensity_all_marks, expanded_time, dim=-2)
            # [..., batch_size, seq_len, resolution, num_marks]

            integral_all_marks_from_zero_to_interval_start = approximate_integration(
                intensity_all_marks, zero_to_interval_start_time, dim=-2, only_integral=True
            )
            # [..., batch_size, seq_len, num_marks]
            integral_all_marks_from_zero_to_interval_start = integral_all_marks_from_zero_to_interval_start.unsqueeze(
                dim=-2
            )
            # [..., batch_size, seq_len, resolution, num_marks]

        return integral_all_marks + integral_all_marks_from_zero_to_interval_start, intensity_all_marks, expanded_time

    def integral_intensity_time_next_3d(
        self,
        time_history,
        time_next,
        marks_history,
        mask_history,
        resolution,
        time_next_start=None,
        time_next_with_resolution_dim=False,
    ):
        if time_next_start is None:
            time_next_start = torch.zeros_like(time_next)  # [..., batch_size, seq_len]

        # calculate the integral
        if time_next_with_resolution_dim:
            expanded_time = time_next
            # [..., batch_size, seq_len, num_marks, resolution]
        else:
            time_multiplier = torch.linspace(0, 1, resolution, device=self.device)
            # [resolution]
            expanded_time = (time_next - time_next_start).unsqueeze(
                dim=-1
            ) * time_multiplier + time_next_start.unsqueeze(dim=-1)
            # [..., batch_size, seq_len, num_marks, resolution]

        hidden_state = self.attn_model(time_history, expanded_time, marks_history, mask_history)
        # [..., batch_size, seq_len, num_marks, resolution, num_marks, d_input]
        intensity_all_marks = self.intensity_layer(hidden_state).squeeze(dim=-1)
        # [..., batch_size, seq_len, num_marks, resolution, num_marks]
        integral_all_marks = approximate_integration(intensity_all_marks, expanded_time, dim=-2)
        # [..., batch_size, seq_len, num_marks, resolution, num_marks]

        # integral offset
        if expanded_time[..., 0].max() == 0:
            integral_all_marks_from_zero_to_interval_start = torch.zeros_like(integral_all_marks)
        else:
            time_multiplier = torch.linspace(0, 1, resolution, device=self.device)
            zero_to_interval_start_time = (
                expanded_time[..., 0].unsqueeze(dim=-1) * time_multiplier
            )  # [..., batch_size, seq_len, num_marks, resolution]
            hidden_state = self.attn_model(time_history, zero_to_interval_start_time, marks_history, mask_history)
            # [..., batch_size, seq_len, num_marks, resolution, num_marks, d_input]
            expanded_intensity_from_zero_to_interval_start = self.intensity_layer(hidden_state).squeeze(dim=-1)
            # [..., batch_size, seq_len, num_marks, resolution, num_marks]

            integral_all_marks_from_zero_to_interval_start = approximate_integration(
                expanded_intensity_from_zero_to_interval_start, zero_to_interval_start_time, dim=-2, only_integral=True
            )
            # [..., batch_size, seq_len, num_marks, num_marks]
            integral_all_marks_from_zero_to_interval_start = integral_all_marks_from_zero_to_interval_start.unsqueeze(
                dim=-2
            )
            # [..., batch_size, seq_len, num_marks, resolution, num_marks]

        return integral_all_marks + integral_all_marks_from_zero_to_interval_start, intensity_all_marks, expanded_time

    def model_probe_function(self, time_history, time_next, marks_history, mask_history, mask_next, resolution):
        # calculate the integral
        time_multiplier = torch.linspace(0, 1, resolution, device=self.device)
        expanded_time = time_next.unsqueeze(dim=-1) * time_multiplier  # [batch_size, seq_len, resolution]

        hidden_state = self.attn_model(time_history, expanded_time, marks_history, mask_history)
        # [batch_size, seq_len, resolution, num_marks, d_input]
        expanded_intensity_all_marks = self.intensity_layer(hidden_state).squeeze(dim=-1)
        # [batch_size, seq_len, resolution, num_marks]

        expanded_integral_all_marks = approximate_integration(expanded_intensity_all_marks, expanded_time, dim=-2)
        # [batch_size, seq_len, resolution, num_marks]

        # construct the plot dict
        data = {}
        data["expand_intensity_for_each_mark"] = (
            expanded_intensity_all_marks  # [batch_size, seq_len, resolution, num_marks]
        )
        data["expand_integral_for_each_mark"] = (
            expanded_integral_all_marks  # [batch_size, seq_len, resolution, num_marks]
        )

        probability_distribution = expanded_intensity_all_marks * torch.exp(
            -expanded_integral_all_marks.sum(dim=-1, keepdim=True)
        )
        # [batch_size, seq_len, resolution, num_mark]

        results = evaluate_on_one_batch(
            probability_distribution,
            dim_input=-3,
            mask=mask_next,
            evaluate_func=["spearman_self", "pearson_self", "l1_self"],
            additional_inputs=[
                expanded_time,
            ],
        )

        data["spearman_matrix"] = move_from_tensor_to_ndarray(results["spearman_self"])
        data["pearson_matrix"] = move_from_tensor_to_ndarray(results["pearson_self"])
        data["L1_matrix"] = move_from_tensor_to_ndarray(results["l1_self"])

        return data, expanded_time
