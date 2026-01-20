import numpy as np
import torch
import torch.nn as nn
from einops import pack, rearrange, reduce, repeat
from scipy.stats import spearmanr

from src.toolbox.metrics import L1_distance_across_marks
from src.toolbox.misc import check_tensor, move_from_tensor_to_ndarray
from src.toolbox.modules import NonNegLinear, ScaledTanh


class IFN(nn.Module):
    """
    IFN (Integration-free Neural Marked Temporal Point Process)
    """

    def __init__(
        self,
        d_history,
        d_intensity,
        num_marks,
        dropout,
        history_module,
        history_module_layers,
        mlp_layers,
        removes_tail,
        tanh_parameter,
        epsilon,
        device,
    ):
        """
        This function creates a IFN model.

        ### Args
            * ```int``` d_history
              The dimension of the history representation.
            * ```float``` dropout
              Dropout rate for the history encoder. Only works when history_module_layers > 1.
            * ```int``` n_layers
              How many self attention layers our model will have?
            * ```int``` n_head
              The number of head in self attention.
            * ```int``` d_qk
              The dimension of matrices Q and K.
            * ```int``` d_v
              The dimension of metrix V.
            * ```int``` d_intensity
              The dimension of the cumulative hazard function network.
            * ```int``` mlp_layers
              The number of layers in the cumulative hazard function network.
            * ```namespace``` opt
              Model arguments.
        """
        super().__init__()
        self.device = device
        self.num_marks = num_marks
        self.epsilon = epsilon
        self.removes_tail = removes_tail
        self.tanh_parameter = tanh_parameter

        self.marks = nn.Embedding(num_marks + 1, d_history, padding_idx=num_marks, device=device)

        self.his_encoder = getattr(nn, history_module)(
            input_size=d_history + 1,
            hidden_size=d_history,
            num_layers=history_module_layers,
            batch_first=True,
            dropout=dropout,
            device=device,
        )

        self.weight_for_t = nn.Parameter(
            torch.zeros((self.num_marks, d_intensity), device=self.device, requires_grad=True)
        )
        self.time_bias = nn.Parameter(torch.ones(self.num_marks, d_intensity, device=self.device, requires_grad=True))
        nn.init.xavier_uniform_(self.weight_for_t)
        nn.init.xavier_uniform_(self.time_bias)

        self.history_mapper = nn.Linear(d_history, d_intensity, bias=True, device=device)
        self.time_mapper = NonNegLinear(d_intensity, d_intensity, device=self.device)

        self.mlp = nn.ModuleList(
            [NonNegLinear(d_intensity, d_intensity, bias=True, device=device) for _ in range(mlp_layers)]
        )

        self.aggregate = NonNegLinear(d_intensity, 1, bias=True, device=device)
        self.layer_activation = ScaledTanh(self.tanh_parameter, device=self.device)

        self.nonneg_activation = nn.Softplus()
        self.nonneg_factor = nn.ReLU()
        self.nonneg_integral = nn.Sigmoid()

    def forward(
        self,
        time_history,
        time_next,
        marks_history,
        mean,
        std,
        custom_marks_history=False,
        training=False,
        extend_input_time=True,
    ):
        """
        IFN's forwardpropagation function.

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
            * ```float``` mean
            * ```float``` std
              Used for input time scaling.
            * ```bool``` custom_marks_history
              when true, the marks_history will be the mark embedding of historical marks.
        ### Outputs
            * ```torch.tensor``` \\Gamma^*(m, t)
              shape: ```[..., batch_size, seq_len, num_marks]```
              The value of \\Gamma^*(m, t) on [t_{i-1}, t_i).
        """
        # Obtain historical embeddings.
        time_history = (time_history - mean) / std  # [batch_size, seq_len]

        marks_embeddings = (
            marks_history if custom_marks_history else self.marks(marks_history)
        )  # [batch_size, seq_len, d_history]
        # [batch_size, seq_len, d_history]
        history, _ = pack([marks_embeddings, time_history], "b s *")  # [batch_size, seq_len, d_history + 1]

        # Reshape hidden output for full connection layers.
        hidden_history, (_, _) = self.his_encoder(history)  # [batch_size, seq_len, d_history]
        hidden_history = repeat(hidden_history, "b s dh -> b s ne dh", ne=self.num_marks)
        # [batch_size, seq_len, num_marks, d_history]
        hidden_history = self.history_mapper(hidden_history)  # [batch_size, seq_len, num_marks, d_intensity]

        # Obtain timestamp embeddings.
        if extend_input_time:
            time_next = repeat(
                time_next, "... b s -> ... b s ne", ne=self.num_marks
            )  # [..., batch_size, seq_len, num_marks]
        time_next.requires_grad = True
        time_next_scaled = (time_next - mean) / std  # [..., batch_size, seq_len, num_marks]
        time_next_zero = torch.ones_like(time_next) * (-mean / std)  # [..., batch_size, seq_len, num_marks]

        time_bias = rearrange(
            self.time_bias, f"... -> {'() ' * (len(time_next.shape) + 1 - len(self.time_bias.shape))}..."
        )
        # [..., 1, 1, num_marks, d_intensity]
        time_embedding = time_next_scaled.unsqueeze(dim=-1) * self.nonneg_activation(self.weight_for_t) + time_bias
        # [..., batch_size, seq_len, num_marks, d_intensity]
        time_zero_embedding = time_next_zero.unsqueeze(dim=-1) * self.nonneg_activation(self.weight_for_t) + time_bias
        # [..., batch_size, seq_len, num_marks, d_intensity]

        time_embedding = self.time_mapper(time_embedding)  # [..., batch_size, seq_len, num_marks, d_intensity]
        time_zero_embedding = self.time_mapper(
            time_zero_embedding
        )  # [..., batch_size, seq_len, num_marks, d_intensity]

        hidden_history = rearrange(
            hidden_history, f"... -> {'() ' * (len(time_embedding.shape) - len(hidden_history.shape))}..."
        )
        # [..., batch_size, seq_len, num_marks, d_intensity]
        output = time_embedding + hidden_history  # [..., batch_size, seq_len, num_marks, d_intensity]
        output_zero = time_zero_embedding + hidden_history  # [..., batch_size, seq_len, num_marks, d_intensity]

        for layer_idx, layer in enumerate(self.mlp):
            output = layer(output)  # [..., batch_size, seq_len, num_marks, d_intensity]
            output = self.layer_activation(output)  # [..., batch_size, seq_len, num_marks, d_intensity]

            output_zero = layer(output_zero)  # [..., batch_size, seq_len, num_marks, d_intensity]
            output_zero = self.layer_activation(output_zero)  # [..., batch_size, seq_len, num_marks, d_intensity]

            if layer_idx == 0:
                output_max = (
                    torch.ones_like(output) * self.tanh_parameter
                )  # [..., batch_size, seq_len, num_marks, d_intensity]
            else:
                output_max = layer(output_max)  # [..., batch_size, seq_len, num_marks, d_intensity]
                output_max = self.layer_activation(output_max)  # [..., batch_size, seq_len, num_marks, d_intensity]

        probability_integral_from_t_to_inf = self.nonneg_integral(-self.aggregate(output))
        # [..., batch_size, seq_len, num_marks, 1]
        probability_integral_from_tl_to_inf = self.nonneg_integral(-self.aggregate(output_zero))
        # [..., batch_size, seq_len, num_marks, 1]
        probability_integral_minimal = self.nonneg_integral(-self.aggregate(output_max))
        # [..., batch_size, seq_len, num_marks, 1]

        if self.removes_tail:
            regularized_probability_integral_from_t_to_inf = (
                probability_integral_from_t_to_inf - probability_integral_minimal
            )
            # [..., batch_size, seq_len, num_marks, 1]
            regularized_probability_integral_from_tl_to_inf = (
                probability_integral_from_tl_to_inf - probability_integral_minimal
            ) + self.epsilon
        # [..., batch_size, seq_len, num_marks, 1]
        else:
            regularized_probability_integral_from_t_to_inf = probability_integral_from_t_to_inf
            # [..., batch_size, seq_len, num_marks, 1]
            regularized_probability_integral_from_tl_to_inf = probability_integral_from_tl_to_inf + self.epsilon
            # [..., batch_size, seq_len, num_marks, 1]

        probability_integral_from_t_to_inf = rearrange(regularized_probability_integral_from_t_to_inf, "... 1 -> ...")
        # [..., batch_size, seq_len, num_marks]
        probability_integral_from_tl_to_inf = reduce(
            regularized_probability_integral_from_tl_to_inf, "... ne 1 -> ... ()", "sum"
        )
        # [..., batch_size, seq_len, 1]

        probability_integral_from_t_to_infinite = (
            probability_integral_from_t_to_inf / probability_integral_from_tl_to_inf
        )
        # [..., batch_size, seq_len, num_marks]

        # the value of probability distribution at t, or p(m, t|\\mathcal{H})
        probability_for_each_mark = -torch.autograd.grad(
            outputs=probability_integral_from_t_to_infinite,
            inputs=time_next,
            grad_outputs=torch.ones_like(probability_integral_from_t_to_infinite),
            create_graph=training,
        )[0]  # [batch_size, seq_len, num_marks]
        time_next.requires_grad = False

        check_tensor(probability_for_each_mark)  # [batch_size, seq_len, num_marks]
        check_tensor(probability_integral_from_t_to_infinite)  # [batch_size, seq_len, num_marks]

        return probability_integral_from_t_to_infinite, probability_for_each_mark

    def gamma_at_t_autoregressive(
        self, sampled_time_history, tau, sampled_marks_history, mean, std, extend_input_time=True
    ):
        """
        IFN's forwardpropagation function specific for sampling the next one event conditioned on the history.

        ### Args
            * ```torch.tensor``` marks_history
              shape: ```[number_of_sampled_sequences, sampled_seq_len]```
              Historical mark sequence.
            * ```torch.tensor``` time_history
              shape: ```[number_of_sampled_sequences, sampled_seq_len]```
              Historical time sequence.
            * ```torch.tensor``` tau
              shape: ```[number_of_sampled_sequences, num_marks]```
              Guessed time when the next mark will happen.
            * ```float``` mean
            * ```float``` std
              Used for input time scaling.
        ### Outputs
            * ```torch.tensor``` \\Gamma^*(m, t)
              shape: ```# [number_of_sampled_sequences, num_marks]```
              The value of \\Gamma^*(m, t) on [t_{i-1}, t_i).
        """
        if extend_input_time:
            tau = repeat(tau, "... -> ... ne", ne=self.num_marks)  # [..., num_marks]
        # Obtain historical embeddings.
        sampled_time_history = (sampled_time_history - mean) / std  # [number_of_sampled_sequences, sampled_seq_len]

        sampled_marks_embeddings = self.marks(
            sampled_marks_history
        )  # [number_of_sampled_sequences, sampled_seq_len, d_history]
        sampled_history = torch.cat([sampled_marks_embeddings, sampled_time_history.unsqueeze(dim=-1)], dim=-1)
        # [number_of_sampled_sequences, sampled_seq_len, d_history + 1]

        # Reshape hidden output for full connection layers.
        _, (sampled_history_embedding, _) = self.his_encoder(
            sampled_history
        )  # [1, number_of_sampled_sequences, d_history]
        sampled_history_embedding = rearrange(sampled_history_embedding, "() bs dh -> bs () dh")
        # [number_of_sampled_sequences, 1, d_history]
        sampled_history_embedding = self.history_mapper(sampled_history_embedding)
        # [number_of_sampled_sequences, 1, d_intensity]
        # Obtain timestamp embeddings.
        tau = (tau - mean) / std  # [number_of_sampled_sequences, num_marks]
        time_next_zero = torch.ones_like(tau) * (-mean / std)  # [number_of_sampled_sequences, num_marks]

        time_bias = rearrange(self.time_bias, f"... -> {'() ' * (len(tau.shape) + 1 - len(self.time_bias.shape))}...")
        # [1, num_marks, d_intensity]
        time_embedding = tau.unsqueeze(dim=-1) * self.nonneg_activation(self.weight_for_t) + time_bias
        # [number_of_sampled_sequences, num_marks, d_intensity]
        time_zero_embedding = time_next_zero.unsqueeze(dim=-1) * self.nonneg_activation(self.weight_for_t) + time_bias
        # [number_of_sampled_sequences, num_marks, d_intensity]

        time_embedding = self.time_mapper(time_embedding)  # [number_of_sampled_sequences, num_marks, d_intensity]
        time_zero_embedding = self.time_mapper(
            time_zero_embedding
        )  # [number_of_sampled_sequences, num_marks, d_intensity]

        output = time_embedding + sampled_history_embedding  # [number_of_sampled_sequences, num_marks, d_intensity]
        output_zero = (
            time_zero_embedding + sampled_history_embedding
        )  # [number_of_sampled_sequences, num_marks, d_intensity]

        for layer_idx, layer in enumerate(self.mlp):
            output = layer(output)  # [number_of_sampled_sequences, num_marks, d_intensity]
            output = self.layer_activation(output)  # [number_of_sampled_sequences, num_marks, d_intensity]

            output_zero = layer(output_zero)  # [number_of_sampled_sequences, num_marks, d_intensity]
            output_zero = self.layer_activation(output_zero)  # [number_of_sampled_sequences, num_marks, d_intensity]

            if layer_idx == 0:
                output_max = (
                    torch.ones_like(output) * self.tanh_parameter
                )  # [number_of_sampled_sequences, num_marks, d_intensity]
            else:
                output_max = layer(output_max)  # [number_of_sampled_sequences, num_marks, d_intensity]
                output_max = self.layer_activation(output_max)  # [number_of_sampled_sequences, num_marks, d_intensity]

        probability_integral_from_t_to_inf = self.nonneg_integral(-self.aggregate(output))
        # [number_of_sampled_sequences, num_marks, 1]
        probability_integral_from_tl_to_inf = self.nonneg_integral(-self.aggregate(output_zero))
        # [number_of_sampled_sequences, num_marks, 1]
        probability_integral_minimal = self.nonneg_integral(-self.aggregate(output_max))
        # [number_of_sampled_sequences, num_marks, 1]

        if self.removes_tail:
            regularized_probability_integral_from_t_to_inf = (
                probability_integral_from_t_to_inf - probability_integral_minimal
            )
            # [number_of_sampled_sequences, num_marks, 1]
            regularized_probability_integral_from_tl_to_inf = (
                probability_integral_from_tl_to_inf - probability_integral_minimal
            ) + self.epsilon
        # [number_of_sampled_sequences, num_marks, 1]
        else:
            regularized_probability_integral_from_t_to_inf = probability_integral_from_t_to_inf
            # [number_of_sampled_sequences, num_marks, 1]
            regularized_probability_integral_from_tl_to_inf = probability_integral_from_tl_to_inf + self.epsilon
            # [number_of_sampled_sequences, num_marks, 1]

        probability_integral_from_t_to_inf = rearrange(regularized_probability_integral_from_t_to_inf, "... 1 -> ...")
        # [number_of_sampled_sequences, num_marks]
        probability_integral_from_tl_to_inf = reduce(
            regularized_probability_integral_from_tl_to_inf, "... ne 1 -> ... ()", "sum"
        )
        # [number_of_sampled_sequences, 1]

        return probability_integral_from_t_to_inf / probability_integral_from_tl_to_inf

    def probability_time_next_2d(self, time_history, time_next, marks_history, resolution, mean, std):
        """
        IFN's forwardpropagation function specifically for probability function probe.

        ### Args
            * ```torch.tensor``` time_history
              shape: ```[batch_size, seq_len]```
              Historical time sequence.
            * ```torch.tensor``` time_next
              shape: ```[batch_size, seq_len]```
              Guessed or real time when the next mark will happen.
            * ```torch.tensor``` marks_history
              shape: ```[batch_size, seq_len]```
              Historical mark sequence.
            * ```int``` resolution
              The number of interpolated points in a time interval between two adjoint marks for integration estimation.
              The number of interpolated points counts the start and end point of the interval.
            * ```float``` mean
            * ```float``` std
              Used for input time scaling.
        ### Outputs
            * ```torch.tensor``` expand_probability
              shape: ```[batch_size, seq_len, resolution, num_marks]```
              The value of the probability distribution at interpolated timestamps.
            * ```torch.tensor``` original_time_expand
              shape: ```[batch_size, seq_len, resolution]```
              What all interpolated timestamps are.
        """
        # History embeddings
        time_history = (time_history - mean) / std  # [batch_size, seq_len]

        marks_embeddings = self.marks(marks_history)  # [batch_size, seq_len, d_history]
        history, history_ps = pack([marks_embeddings, time_history], "b s *")  # [batch_size, seq_len, d_history + 1]

        hidden_history, (_, _) = self.his_encoder(history)  # [batch_size, seq_len, d_history]
        hidden_history = self.history_mapper(hidden_history)  # [batch_size, seq_len, d_intensity]

        hidden_history = repeat(hidden_history, "b s di -> b s r ne di", r=resolution, ne=self.num_marks)
        # [batch_size, seq_len, resolution, num_marks, d_intensity]

        # Expanded time embedding
        time_multiplier = torch.linspace(0, 1, resolution, device=self.device)
        # [resolution]
        original_time_expand = time_multiplier * time_next.unsqueeze(dim=-1)  # [batch_size, seq_len, resolution]
        time_expand = original_time_expand.clone()  # [batch_size, seq_len, resolution]
        time_expand = repeat(time_expand, "b s r -> b s r ne", ne=self.num_marks)
        # [batch_size, seq_len, resolution, num_marks]

        time_expand.requires_grad = True
        time_expand_norm = (time_expand - mean) / std  # [batch_size, seq_len, resolution, num_marks]

        time_bias = rearrange(
            self.time_bias, f"... -> {'() ' * (len(time_expand_norm.shape) + 1 - len(self.time_bias.shape))}..."
        )
        # [1, 1, 1, num_marks, d_intensity]
        emb_time_expand = time_expand_norm.unsqueeze(dim=-1) * self.nonneg_activation(self.weight_for_t) + time_bias
        # [batch_size, seq_len, resolution, num_marks, d_intensity]
        emb_time_expand = self.time_mapper(emb_time_expand)  # [batch_size, seq_len, resolution, num_marks, d_intensity]
        output = emb_time_expand + hidden_history  # [batch_size, seq_len, resolution, num_marks, d_intensity]

        for layer_idx, layer in enumerate(self.mlp):
            output = layer(output)  # [batch_size, seq_len, resolution, num_marks, d_intensity]
            output = self.layer_activation(output)  # [batch_size, seq_len, resolution, num_marks, d_intensity]

            if layer_idx == 0:
                output_max = (
                    torch.ones((*output.shape[:2], *output.shape[3:]), device=self.device) * self.tanh_parameter
                )
            # [batch_size, seq_len, num_marks, d_intensity]
            else:
                output_max = layer(output_max)  # [batch_size, seq_len, num_marks, d_intensity]
                output_max = self.layer_activation(output_max)  # [batch_size, seq_len, num_marks, d_intensity]

        expand_integral = self.nonneg_integral(
            -self.aggregate(output)
        )  # [batch_size, seq_len, resolution, num_marks, 1]
        expand_integral_minimal = self.nonneg_integral(-self.aggregate(output_max))
        # [batch_size, seq_len, num_marks, 1]
        expand_integral_minimal = rearrange(expand_integral_minimal, "b s ne last -> b s () ne last")
        # [batch_size, seq_len, 1, num_marks, 1]
        if self.removes_tail:
            expand_integral = (
                expand_integral - expand_integral_minimal
            )  # [batch_size, seq_len, resolution, num_marks, 1]
        integral_from_zero_to_inf = expand_integral[:, :, 0, :, :].detach() + self.epsilon
        # [batch_size, seq_len, num_marks, 1]
        integral_sum = reduce(integral_from_zero_to_inf, "b s ne 1 -> b s 1 1 1", "sum")
        # [batch_size, seq_len, 1, 1, 1]
        expand_integral = expand_integral / integral_sum  # [batch_size, seq_len, resolution, num_marks, 1]

        expand_probability = -torch.autograd.grad(
            outputs=expand_integral,
            inputs=time_expand,
            grad_outputs=torch.ones_like(expand_integral),
        )[0]  # [batch_size, seq_len, resolution, num_marks]
        time_expand.requires_grad = False

        expand_probability = expand_probability.detach()  # [batch_size, seq_len, resolution, num_marks]

        return expand_probability, original_time_expand

    def get_mark_embedding(self, input_mark):
        """
        Get mark embeddings for input_mark.

        ### Args
            * ```torch.tensor``` input_mark
              shape: ```[batch_size, seq_len]```
              Input mark sequence.
        ### Outputs
            * ```torch.tensor```
              shape: ```[batch_size, seq_len, d_history]```
              The output embeddings.
        """
        return self.marks(input_mark)  # [batch_size, seq_len, d_history]

    def model_probe_function(self, time_history, time_next, marks_history, mask_next, resolution, mean, std):
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
            * ```int``` resolution
              The number of interpolated points in a time interval between two adjoint marks for integration estimation.
              The number of interpolated points counts the start and end point of the interval.
            * ```float``` mean
            * ```float``` std
              Used for input time scaling.
        ### Outputs
            * ```dict``` data
              Probed data used for plot drawing.
            * ```torch.tensor``` original_time_expand
              shape: ```[batch_size, seq_len, resolution]```
              The value of sampled times.
        """
        # History embeddings
        time_history = (time_history - mean) / std  # [batch_size, seq_len]

        marks_embeddings = self.marks(marks_history)  # [batch_size, seq_len, d_history]
        history, history_ps = pack([marks_embeddings, time_history], "b s *")  # [batch_size, seq_len, d_history + 1]

        hidden_history, (_, _) = self.his_encoder(history)  # [batch_size, seq_len, d_history]
        hidden_history = self.history_mapper(hidden_history)  # [batch_size, seq_len, d_intensity]

        hidden_history = repeat(hidden_history, "b s di -> b s r ne di", r=resolution, ne=self.num_marks)
        # [batch_size, seq_len, resolution, num_marks, d_intensity]

        # Expanded time embedding
        time_multiplier = torch.linspace(0, 1, resolution, device=self.device)
        # [resolution]
        original_time_expand = time_multiplier * rearrange(time_next, "... -> ... 1")
        # [batch_size, seq_len, resolution]
        time_expand = original_time_expand.clone()  # [batch_size, seq_len, resolution]
        time_expand = repeat(original_time_expand, "b s r -> b s r ne", ne=self.num_marks)
        # [batch_size, seq_len, resolution, num_marks]

        time_expand.requires_grad = True
        time_expand_norm = (time_expand - mean) / std  # [batch_size, seq_len, resolution, num_marks]

        time_bias = rearrange(
            self.time_bias, f"... -> {'() ' * (len(time_expand_norm.shape) + 1 - len(self.time_bias.shape))}..."
        )
        # [1, 1, 1, num_marks, d_intensity]
        emb_time_expand = time_expand_norm.unsqueeze(dim=-1) * self.nonneg_activation(self.weight_for_t) + time_bias
        # [batch_size, seq_len, resolution, num_marks, d_intensity]
        emb_time_expand = self.time_mapper(emb_time_expand)  # [batch_size, seq_len, resolution, num_marks, d_intensity]
        output = emb_time_expand + hidden_history  # [batch_size, seq_len, resolution, num_marks, d_intensity]

        for layer_idx, layer in enumerate(self.mlp):
            output = layer(output)  # [batch_size, seq_len, resolution, num_marks, d_intensity]
            output = self.layer_activation(output)  # [batch_size, seq_len, resolution, num_marks, d_intensity]

            if layer_idx == 0:
                output_max = (
                    torch.ones((*output.shape[:2], *output.shape[3:]), device=self.device) * self.tanh_parameter
                )
            # [batch_size, seq_len, num_marks, d_intensity]
            else:
                output_max = layer(output_max)  # [batch_size, seq_len, num_marks, d_intensity]
                output_max = self.layer_activation(output_max)  # [batch_size, seq_len, num_marks, d_intensity]

        expand_integral = self.nonneg_activation(
            -self.aggregate(output)
        )  # [batch_size, seq_len, resolution, num_marks, 1]
        expand_integral_minimal = self.nonneg_integral(-self.aggregate(output_max))
        # [batch_size, seq_len, num_marks, 1]
        expand_integral_minimal = rearrange(expand_integral_minimal, "b s ne last -> b s () ne last")
        # [batch_size, seq_len, 1, num_marks, 1]
        if self.removes_tail:
            expand_integral = (
                expand_integral - expand_integral_minimal
            )  # [batch_size, seq_len, resolution, num_marks, 1]
        expand_integral = expand_integral.squeeze(dim=-1)  # [batch_size, seq_len, resolution, num_marks]

        integral_from_zero_to_inf = expand_integral[:, :, 0, :].detach() + self.epsilon
        # [batch_size, seq_len, num_marks]
        integral_sum = reduce(integral_from_zero_to_inf, "b s ne -> b s ()", "sum")
        # [batch_size, seq_len, 1]
        integral_sum = rearrange(integral_sum, "b s 1 -> b s 1 1")  # [batch_size, seq_len, 1, 1]
        expand_integral = expand_integral / integral_sum  # [batch_size, seq_len, resolution, num_marks]

        # Gradient 1: Integral -> time
        marks_probability_at_each_interpolated_timestamp = -torch.autograd.grad(
            outputs=expand_integral,
            inputs=time_expand,
            grad_outputs=torch.ones_like(expand_integral),
            retain_graph=True,
        )[0]  # [batch_size, seq_len, resolution, num_marks]

        time_expand.requires_grad = False

        # The data dict is defined here.
        # This dict should pack all data required by plot().
        data = {}
        data["expand_probability_for_each_mark"] = marks_probability_at_each_interpolated_timestamp
        # [batch_size, seq_len, resolution, num_marks]

        probability_for_each_mark = rearrange(
            marks_probability_at_each_interpolated_timestamp, "b s r ne -> b (s r) ne"
        )
        # [batch_size, seq_len * resolution, num_marks]

        spearman_matrix = []
        pearson_matrix = []
        l1_matrix = []
        for _, (expand_probability_per_seq, mask_per_seq, original_time_expand_per_seq) in enumerate(
            zip(probability_for_each_mark, mask_next, original_time_expand)
        ):
            seq_len = mask_per_seq.sum()
            expand_probability_per_seq = move_from_tensor_to_ndarray(expand_probability_per_seq)

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
                spearman_matrix_per_seq = spearmanr(expand_probability_per_seq[: seq_len * resolution])[0]
                if self.num_marks == 2:
                    spearman_matrix_per_seq = np.array([[1, spearman_matrix_per_seq], [spearman_matrix_per_seq, 1]])

            # r: pearson coefficient
            pearson_matrix_per_seq = np.corrcoef(expand_probability_per_seq[: seq_len * resolution], rowvar=False)
            if self.num_marks == 1:
                pearson_matrix_per_seq = rearrange(np.array(pearson_matrix_per_seq), " -> () ()")

            # L^1 metric
            l1_matrix_per_seq = L1_distance_across_marks(
                expand_probability_per_seq[: seq_len * resolution],
                time_next=original_time_expand_per_seq[:seq_len],
                has_flatten=True,
            )

            spearman_matrix.append(spearman_matrix_per_seq)
            pearson_matrix.append(pearson_matrix_per_seq)
            l1_matrix.append(l1_matrix_per_seq)

        data["spearman_matrix"] = spearman_matrix
        data["pearson_matrix"] = pearson_matrix
        data["L1_matrix"] = l1_matrix

        return data, original_time_expand
