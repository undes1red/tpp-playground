import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

from src.toolbox.algorithms import approximate_integration, evaluate_on_one_batch
from src.toolbox.misc import move_from_tensor_to_ndarray
from src.toolbox.modules import softplus_ext

from .his_coder.transformers import TransformerTPP


class THP(nn.Module):
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
        history_time_offset,
    ):
        """
        This function creates a SAHP model.

        ### Args
            * ```int``` d_input
            The dimension of the Transformer input tensor.
            * ```int``` d_hidden
              The dimension of the FFN module in the Transformer.
            * ```int``` n_layers
              The number of self attention + FFN layers in the Transformer.
            * ```int``` n_head
              The number of head in self attention.
            * ```int``` d_qkv
              The dimension of matrices Q and K.
            * ```int``` d_v
              The dimension of metrix V.
            * ```float``` dropout
              Dropout rate for the history encoder.
            * ```int``` d_rnn
              The dimension of RNN's hidden state.
            * ```float``` history_time_offset
              THP scales the input time by dividing it with the time interval from start to the latest mark in history.
              This can cause issues when there is no mark in history-the input time will be divided by 0.
              So, we add this offset to the time interval to avoid this divided-by-0 case.
            * ```torch.device``` device
              Running models on GPU or CPU?
            * ```int``` integration_sample_rate
              The number of interpolated points in a time interval between two adjoint mark for integration estimation.
              The number of interpolated points counts the start and end point of the interval.
        """
        super().__init__()
        self.device = device
        self.num_marks = num_marks
        self.training = training
        self.integration_sample_rate = integration_sample_rate
        self.history_time_offset = history_time_offset

        # parameter for the weight of time difference
        self.alpha = nn.Parameter(torch.ones((self.num_marks), device=self.device, requires_grad=True))
        nn.init.normal_(self.alpha)

        # parameter for the softplus function
        self.beta = nn.Parameter(torch.zeros((self.num_marks), device=self.device, requires_grad=True))
        nn.init.normal_(self.beta)

        # convert hidden vectors into valid intensity function values.
        self.linear = nn.Linear(d_input, num_marks, device=self.device)

        # the history encoder
        self.history_encoder = TransformerTPP(
            self.training,
            num_marks,
            device=self.device,
            d_input=d_input,
            d_hidden=d_hidden,
            n_layers=n_layers,
            n_head=n_head,
            d_qkv=d_qkv,
            dropout=dropout,
        )

    def extract_history_embeddings(self, time, mark, mask):
        """
        Extract history from the provided mark sequence and encode it into history representations.

        ### Args
            * ```torch.tensor``` time
              shape: ```[batch_size, seq_len]```
              Historical time sequence.
            * ```torch.tensor``` mark
              shape: ```[..., batch_size, seq_len]```
              Guessed or real time when the next mark will happen.
            * ```torch.tensor``` mask
              shape: ```[..., batch_size, seq_len]```
              Used to mask out padding mark from the attention map.
        ### Outputs
            * ```torch.tensor``` integral_all_mark
              shape: ```[..., batch_size, seq_len, num_marks]```
              The value of \\Lambda^*(m, t) on [t_{i-1}, t_i).
            * ```torch.tensor``` intensity_all_mark
              shape: ```[..., batch_size, seq_len, num_marks]```
              The value of \\lambda^*(m, t) on at t_i.
        """
        time_history, _ = self.divide_history_and_next(time)  # [batch_size, seq_len]
        mark_history, _ = self.divide_history_and_next(mark)  # [batch_size, seq_len]
        mask_history, _ = self.divide_history_and_next(mask)  # [batch_size, seq_len]

        return self.history_encoder(time_history, mark_history, mask_history)
        # [batch_size, seq_len, num_marks]

    def forward(self, time_history, time_next, marks_history, masks_history):
        """
        THP's forwardpropagation function for training.

        ### Args
            * ```torch.tensor``` mark_history
              shape: ```[batch_size, seq_len]```
              Historical mark sequence.
            * ```torch.tensor``` time_history
              shape: ```[batch_size, seq_len]```
              Historical time sequence.
            * ```torch.tensor``` time_next
              shape: ```[..., batch_size, seq_len]```
              Guessed or real time when the next mark will happen.
            * ```torch.tensor``` mask_history
              shape: ```[..., batch_size, seq_len]```
              Used to mask out padding mark from the attention map.
        ### Outputs
            * ```torch.tensor``` integral_all_mark
              shape: ```[..., batch_size, seq_len, num_marks]```
              The value of \\Lambda^*(m, t) on [t_{i-1}, t_i).
            * ```torch.tensor``` intensity_all_mark
              shape: ```[..., batch_size, seq_len, num_marks]```
              The value of \\lambda^*(m, t) on at t_i.
        """
        history = self.history_encoder(time_history, marks_history, masks_history)
        # [batch_size, seq_len, d_input]

        aggregate_time = time_history.cumsum(dim=-1)  # [batch_size, seq_len]
        # Avoid zero denominator
        aggregate_time = aggregate_time + self.history_time_offset  # [batch_size, seq_len]
        aggregate_time = rearrange(
            aggregate_time, f"... -> {'() ' * (len(time_next.shape) - len(aggregate_time.shape))}..."
        )
        # [..., batch_size, seq_len]
        history = rearrange(history, f"... -> {'() ' * (len(time_next.shape) - len(aggregate_time.shape))}...")
        # [..., batch_size, seq_len, d_input]

        scaled_time = (time_next / aggregate_time).unsqueeze(dim=-1)  # [..., batch_size, seq_len, 1]
        intensity_all_mark = softplus_ext(self.linear(history) + self.alpha * scaled_time, beta=F.softplus(self.beta))
        # [..., batch_size, seq_len, num_marks]

        reshaped_aggregate_time = rearrange(
            time_history.cumsum(dim=-1), f"... -> {'() ' * (len(time_next.shape) - len(aggregate_time.shape))}... () ()"
        )
        # [..., batch_size, seq_len, 1, 1]
        reshaped_aggregate_time = reshaped_aggregate_time + self.history_time_offset
        # [..., batch_size, seq_len, 1, 1]
        time_multiplier = torch.linspace(0, 1, self.integration_sample_rate, device=self.device)
        expanded_time = (
            time_next.unsqueeze(dim=-1) * time_multiplier
        )  # [..., batch_size, seq_len, integration_sample_rate]
        expanded_scaled_time = self.alpha * expanded_time.unsqueeze(dim=-1) / reshaped_aggregate_time
        # [..., batch_size, seq_len, integration_sample_rate, num_marks]
        intensity_all_mark_pre_softplus = self.linear(history)  # [..., batch_size, seq_len, num_marks]
        intensity_all_mark_pre_softplus = rearrange(
            intensity_all_mark_pre_softplus, "... ne -> ... () ne"
        )
        # [..., batch_size, seq_len, integration_sample_rate, num_marks]
        all_lambda = softplus_ext(intensity_all_mark_pre_softplus + expanded_scaled_time, F.softplus(self.beta))
        # [..., batch_size, seq_len, integration_sample_rate, num_marks]
        integral_all_mark = approximate_integration(all_lambda, expanded_time, dim=-2, only_integral=True)
        # [..., batch_size, seq_len, num_marks]

        return integral_all_mark, intensity_all_mark

    def integral_intensity_time_next_2d(
        self,
        time_history,
        time_next,
        marks_history,
        mask_history,
        integration_sample_rate,
        time_next_start=None,
        time_next_with_resolution_dim=False,
    ):
        """
        Probe the value of the intensity function and its integral at sampled timestamps.
        In this function, all marks share the sampled timestmaps, so the dimension of time_next does not include num_mark.

        ### Args
            * ```torch.tensor``` mark_history
              shape: ```[batch_size, seq_len]```
              Historical mark sequence.
            * ```torch.tensor``` time_history
              shape: ```[batch_size, seq_len]```
              Historical time sequence.
            * ```torch.tensor``` time_next
              shape: ```[..., batch_size, seq_len]```
              Guessed or real time when the next mark will happen.
            * ```torch.tensor``` mask_history
              shape: ```[..., batch_size, seq_len]```
              Used to mask out padding mark from the attention map.
            * ```int``` integration_sample_rate
              The number of interpolated points in a time interval between two adjoint mark for integration estimation.
              The number of interpolated points counts the start and end point of the interval.
            * ```torch,tensor``` time_next_start
              shape: ```[..., batch_size, seq_len]``` if not None
              When given, this function computes the integral between [time_next_start, t_i]. time_next_start are expected to be non-negative.
              This affects the integral, intensity, and timestamp.
        ### Outputs
            * ```torch.tensor``` expanded_integral_all_mark
              shape: ```[..., batch_size, seq_len, resolution, num_marks]```
              The value of \\Lambda^*(m, t) at sampled times.
            * ```torch.tensor``` expanded_intensity_all_mark
              shape: ```[..., batch_size, seq_len, resolution, num_marks]```
              The value of \\lambda^*(m, t) at sampled times.
            * ```torch.tensor``` expanded_time
              shape: ```[..., batch_size, seq_len, resolution]```
              The value of sampled times.
        """
        if time_next_start is None:
            time_next_start = torch.zeros_like(time_next)  # [..., batch_size, seq_len]

        time_multiplier = torch.linspace(0, 1, integration_sample_rate, device=self.device)
        if time_next_with_resolution_dim:
            original_expanded_time = time_next
        else:
            time_multiplier = torch.linspace(0, 1, integration_sample_rate, device=self.device)
            original_expanded_time = (time_next - time_next_start).unsqueeze(
                dim=-1
            ) * time_multiplier + time_next_start.unsqueeze(dim=-1)
            # [..., batch_size, seq_len, integration_sample_rate]

        history = self.history_encoder(time_history, marks_history, mask_history)
        # [batch_size, seq_len, d_input]
        einop = f"b s di -> {'() ' * (len(original_expanded_time.shape) - 3)} b s () di"
        history = rearrange(history, einop)  # [..., batch_size, seq_len, 1, d_input]

        aggregate_time = time_history.cumsum(dim=-1)  # [batch_size, seq_len]
        # Avoid zero denominator
        aggregate_time = aggregate_time + self.history_time_offset  # [batch_size, seq_len]
        einop = f"b s -> {'() ' * (len(original_expanded_time.shape) - 3)} b s ()"
        aggregate_time = rearrange(aggregate_time, einop)  # [..., batch_size, seq_len, 1]

        scaled_time = (original_expanded_time / aggregate_time).unsqueeze(
            dim=-1
        )  # [..., batch_size, seq_len, integration_sample_rate, 1]
        expanded_intensity_all_mark = softplus_ext(
            self.linear(history) + self.alpha * scaled_time, beta=F.softplus(self.beta)
        )
        # [..., batch_size, seq_len, integration_sample_rate, num_marks]
        expanded_integral_all_mark = approximate_integration(expanded_intensity_all_mark, original_expanded_time, dim=-2)
        # [..., batch_size, seq_len, integration_sample_rate, num_marks]

        # integral offset
        if original_expanded_time[..., 0].max() == 0:
            integral_all_marks_from_zero_to_interval_start = torch.zeros_like(expanded_integral_all_mark)
        else:
            time_multiplier = torch.linspace(0, 1, integration_sample_rate, device=self.device)
            zero_to_interval_start_time = (
                original_expanded_time[..., 0].unsqueeze(dim=-1) * time_multiplier
            )  # [..., batch_size, seq_len, integration_sample_rate]

            scaled_zero_to_interval_start_time = (zero_to_interval_start_time / aggregate_time).unsqueeze(
                dim=-1
            )  # [..., batch_size, seq_len, integration_sample_rate, 1]
            intensity_all_marks_from_zero_to_interval_start = softplus_ext(
                self.linear(history) + self.alpha * scaled_zero_to_interval_start_time, beta=F.softplus(self.beta)
            )
            # [..., batch_size, seq_len, integration_sample_rate, num_marks]
            integral_all_marks_from_zero_to_interval_start = approximate_integration(
                intensity_all_marks_from_zero_to_interval_start, zero_to_interval_start_time, dim=-2, only_integral=True
            ).unsqueeze(dim=-2)
            # [..., batch_size, seq_len, integration_sample_rate, num_marks]

        return (
            expanded_integral_all_mark + integral_all_marks_from_zero_to_interval_start,
            expanded_intensity_all_mark,
            original_expanded_time,
        )

    def integral_intensity_time_next_3d(
        self,
        time_history,
        time_next,
        marks_history,
        mask_history,
        integration_sample_rate,
        time_next_start=None,
        time_next_with_resolution_dim=False,
    ):
        """
        Probe the value of the intensity function and its integral at sampled timestamps.
        In this function, all marks can have their sampled timestmaps, so the dimension of time_next is ```[..., batch_size, seq_len, num_marks]```.
        This function is supposed to be much slower than integral_intensity_time_next_2d().

        ### Args
            * ```torch.tensor``` mark_history
              shape: ```[batch_size, seq_len]```
              Historical mark sequence.
            * ```torch.tensor``` time_history
              shape: ```[batch_size, seq_len]```
              Historical time sequence.
            * ```torch.tensor``` time_next
              shape: ```[..., batch_size, seq_len, num_marks]```
              Guessed or real time when the next mark will happen.
            * ```torch.tensor``` mask_history
              shape: ```[..., batch_size, seq_len]```
              Used to mask out padding mark from the attention map.
            * ```int``` integration_sample_rate
              The number of interpolated points in a time interval between two adjoint mark for integration estimation.
              The number of interpolated points counts the start and end point of the interval.
        ### Outputs
            * ```torch.tensor``` expanded_integral_all_mark
              shape: ```[..., batch_size, seq_len, resolution, num_marks]```
              The value of \\Lambda^*(m, t) at sampled times.
            * ```torch.tensor``` expanded_intensity_all_mark
              shape: ```[..., batch_size, seq_len, resolution, num_marks]```
              The value of \\lambda^*(m, t) at sampled times.
            * ```torch.tensor``` expanded_time
              shape: ```[..., batch_size, seq_len, resolution]```
              The value of sampled times.
        """
        if time_next_start is None:
            time_next_start = torch.zeros_like(time_next)  # [..., batch_size, seq_len]

        history = self.history_encoder(time_history, marks_history, mask_history)
        # [batch_size, seq_len, d_input]

        if time_next_with_resolution_dim:
            original_expanded_time = time_next
        else:
            time_multiplier = torch.linspace(0, 1, integration_sample_rate, device=self.device)
            original_expanded_time = (time_next - time_next_start).unsqueeze(
                dim=-1
            ) * time_multiplier + time_next_start.unsqueeze(dim=-1)
            # [..., batch_size, seq_len, num_mark, integration_sample_rate]

        expanded_time = original_expanded_time.unsqueeze(
            dim=-1
        )  # [..., batch_size, seq_len, num_mark, integration_sample_rate, 1]

        history = rearrange(history, f"... -> {'() ' * (len(original_expanded_time.shape) - 4)}...")
        # [..., batch_size, seq_len, d_input]
        aggregate_time = rearrange(
            torch.cumsum(time_history, dim=-1),
            f"... -> {'() ' * (len(original_expanded_time.shape) - 4)}... () () ()",
        )
        # [..., batch_size, seq_len, 1, 1, 1]
        aggregate_time = aggregate_time + self.history_time_offset  # [..., batch_size, seq_len, 1, 1, 1]
        scaled_expanded_time = (
            expanded_time / aggregate_time
        )  # [..., batch_size, seq_len, num_mark, integration_sample_rate, 1]

        intensity_for_each_mark = self.linear(history)  # [..., batch_size, seq_len, num_marks]
        intensity_for_each_mark = rearrange(intensity_for_each_mark, "... ne -> ... () () ne")
        # [..., batch_size, seq_len, 1, 1, num_marks]
        expanded_intensity_across_all_mark = softplus_ext(
            self.alpha * scaled_expanded_time + intensity_for_each_mark, F.softplus(self.beta)
        )
        # [..., batch_size, seq_len, num_marks, integration_sample_rate, num_marks]
        # [..., batch_size, seq_len, num_marks, integration_sample_rate, num_marks]
        expanded_integral_across_all_mark = approximate_integration(
            expanded_intensity_across_all_mark, original_expanded_time, dim=-2
        )
        # [..., batch_size, seq_len, num_marks, integration_sample_rate, num_marks]

        # integral offset
        if original_expanded_time[..., 0].max() == 0:
            integral_all_marks_from_zero_to_interval_start = torch.zeros_like(expanded_integral_across_all_mark)
        else:
            time_multiplier = torch.linspace(0, 1, integration_sample_rate, device=self.device)
            zero_to_interval_start_time = (
                original_expanded_time[..., 0].unsqueeze(dim=-1) * time_multiplier
            ).unsqueeze(dim=-1)
            # [..., batch_size, seq_len, num_marks, integration_sample_rate, num_marks]

            scaled_zero_to_interval_start_time = (zero_to_interval_start_time / aggregate_time)
            # [..., batch_size, seq_len, num_marks, integration_sample_rate, 1]
            intensity_all_marks_from_zero_to_interval_start = softplus_ext(
                intensity_for_each_mark + self.alpha * scaled_zero_to_interval_start_time, beta=F.softplus(self.beta)
            )
            # [..., batch_size, seq_len, num_marks, integration_sample_rate, num_marks]
            integral_all_marks_from_zero_to_interval_start = approximate_integration(
                intensity_all_marks_from_zero_to_interval_start, zero_to_interval_start_time, dim=-2, only_integral=True, func_val_x_having_same_shape=True
            ).unsqueeze(dim=-2)
            # [..., batch_size, seq_len, num_marks, integration_sample_rate, num_marks]

        return (
            expanded_integral_across_all_mark + integral_all_marks_from_zero_to_interval_start,
            expanded_intensity_across_all_mark,
            original_expanded_time,
        )

    def model_probe_function(
        self, time_history, time_next, marks_history, mask_history, mask_next, integration_sample_rate
    ):
        """
        Probe the value of the intensity function and its integral at sampled timestamps.
        In this function, all marks can have their sampled timestmaps, so the dimension of time_next is ```[..., batch_size, seq_len, num_marks]```.
        This function is supposed to be much slower than integral_intensity_time_next_2d().

        ### Args
            * ```torch.tensor``` mark_history
              shape: ```[batch_size, seq_len]```
              Historical mark sequence.
            * ```torch.tensor``` time_history
              shape: ```[batch_size, seq_len]```
              Historical time sequence.
            * ```torch.tensor``` time_next
              shape: ```[..., batch_size, seq_len]```
              Guessed or real time when the next mark will happen.
            * ```torch.tensor``` mask_history
              shape: ```[..., batch_size, seq_len]```
              Used to mask out padding mark from the attention map.
            * ```torch.tensor``` mask_next
              shape: ```[..., batch_size, seq_len]```
              Tell which mark in *_next is the real mark so should be considered in metric calculation.
            * ```int``` integration_sample_rate
              The number of interpolated points in a time interval between two adjoint mark for integration estimation.
              The number of interpolated points counts the start and end point of the interval.
        ### Outputs
            * ```dict``` data
              Probed data used for plot drawing.
            * ```torch.tensor``` expanded_time
              shape: ```[batch_size, seq_len, resolution]```
              The value of sampled times.
        """
        history = self.history_encoder(time_history, marks_history, mask_history)
        # [batch_size, seq_len, d_input]
        history = repeat(history, "b s di -> b s 1 di")  # [batch_size, seq_len, 1, d_input]

        aggregate_time = time_history.cumsum(dim=-1).unsqueeze(dim=-1)  # [batch_size, seq_len]
        # Avoid zero denominator
        aggregate_time = aggregate_time + self.history_time_offset  # [batch_size, seq_len]

        time_multiplier = torch.linspace(0, 1, integration_sample_rate, device=self.device)
        expanded_time = time_next.unsqueeze(dim=-1) * time_multiplier  # [batch_size, seq_len, integration_sample_rate]

        scaled_time = (expanded_time / aggregate_time).unsqueeze(
            dim=-1
        )  # [batch_size, seq_len, integration_sample_rate, 1]
        expanded_intensity_all_mark = softplus_ext(
            self.linear(history) + self.alpha * scaled_time, beta=F.softplus(self.beta)
        )
        # [batch_size, seq_len, integration_sample_rate, num_marks]
        expanded_integral_all_mark = approximate_integration(expanded_intensity_all_mark, expanded_time, dim=-2)
        # [batch_size, seq_len, integration_sample_rate, num_marks]

        # construct the plot dict
        data = {}
        data["expand_intensity_for_each_mark"] = (
            expanded_intensity_all_mark  # [batch_size, seq_len, integration_sample_rate, num_marks]
        )
        data["expand_integral_for_each_mark"] = (
            expanded_integral_all_mark  # [batch_size, seq_len, integration_sample_rate, num_marks]
        )

        probability_distribution = expanded_intensity_all_mark * torch.exp(-expanded_integral_all_mark.sum(dim=-1, keepdim=True))
        # [batch_size, seq_len, integration_sample_rate, num_mark]

        results = evaluate_on_one_batch(
            probability_distribution,
            dim_input=-3,
            mask=mask_next,
            evaluate_func=["spearman_self", "pearson_self", "l1_self"],
            additional_inputs=[
                expanded_time,
            ],
        )

        data["spearman_matrix"] = move_from_tensor_to_ndarray(results['spearman_self'])
        data["pearson_matrix"] = move_from_tensor_to_ndarray(results['pearson_self'])
        data["L1_matrix"] = move_from_tensor_to_ndarray(results['l1_self'])

        return data, expanded_time
