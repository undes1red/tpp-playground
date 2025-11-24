import numpy as np
import torch
import torch.nn as nn
from einops import pack, rearrange, reduce, repeat, unpack
from scipy.stats import spearmanr

from src.toolbox.metrics import L1_distance_across_marks
from src.toolbox.misc import check_tensor, move_from_tensor_to_ndarray
from src.toolbox.modules import NonNegLinear


class FullyNN(nn.Module):
    def __init__(
        self, d_history, d_intensity, num_marks, dropout, history_module, history_module_layers, mlp_layers, device
    ):
        """
        This function creates a FullyNN model.

        ### Args
            * ```str``` history_module
              Which RNN model do we use to encode the history? Default is LSTM. We don't recommend to change it to something else.
            * ```int``` d_history
              The dimension of the history representation.
            * ```float``` dropout
              Dropout rate for the history encoder. Only works when history_module_layers > 1.
            * ```int``` history_module_layers
              How many layer of RNN our model will have?
            * ```int``` d_intensity
              The dimension of the cumulative hazard function network.
            * ```int``` mlp_layers
              The number of layers in the cumulative hazard function network.
            * ```int``` num_marks
              The number of available marks in the sequence.
        """
        super().__init__()
        self.device = device
        self.num_marks = num_marks

        # FullyNN can not distinguish different markers because of computation graph overlap.
        # It is expected that the original FullyNN achieves very bad marker prediction performance regardless the model size.
        self.marks = nn.Embedding(num_marks + 1, d_history, padding_idx=num_marks, device=device)

        self.his_encoder = getattr(nn, history_module)(
            input_size=d_history + 1,
            hidden_size=d_history,
            num_layers=history_module_layers,
            batch_first=True,
            dropout=dropout,
            device=device,
        )

        # Map the time number into a vector.
        self.weight_for_t = nn.Parameter(torch.zeros((1, d_intensity), device=self.device, requires_grad=True))
        nn.init.xavier_uniform_(self.weight_for_t)

        # Map history and time embeddings into the same hidden space.
        self.history_mapper = nn.Linear(d_history, d_intensity, bias=True, device=device)
        self.time_mapper = NonNegLinear(d_intensity, d_intensity, device=self.device)

        # IEM module featuring non-negative fully connected layers.
        self.mlp = nn.ModuleList(
            [NonNegLinear(d_intensity, d_intensity, bias=True, device=device) for _ in range(mlp_layers)]
        )
        self.layer_activation = nn.Tanh()
        self.aggregate = NonNegLinear(d_intensity, 1, bias=True, device=device)
        self.nonneg_activation = nn.Softplus()

    def forward(self, time_history, time_next, marks_history, mean, std, custom_marks_history=False, training=False):
        """
        FENNModel's forwardpropagation function for training.

        ### Args
            * ```torch.tensor``` marks_history
              shape: ```[batch_size, seq_len]```
              Historical event sequence.
            * ```torch.tensor``` time_history
              shape: ```[batch_size, seq_len]```
              Historical time sequence.
            * ```torch.tensor``` time_next
              shape: ```[..., batch_size, seq_len]```
              Guessed or real time when the next event will happen.
            * ```float``` mean
            * ```float``` std
              Used for input time scaling.
            * ```bool``` custom_marks_history
              when true, the marks_history will be the mark embedding of historical marks.
        ### Outputs
            * ```torch.tensor``` integral
              shape: ```[..., batch_size, seq_len, num_marks]```
              The value of \\Lambda^*(m, t) on [t_{i-1}, t_i).
        """
        time_history = (time_history - mean) / std  # [batch_size, seq_len]

        marks_embeddings = marks_history if custom_marks_history else self.marks(marks_history)
        # [batch_size, seq_len, d_history]
        history = torch.cat([marks_embeddings, time_history.unsqueeze(dim=-1)], dim=-1)
        # [batch_size, seq_len, d_history + 1]
        # Reshape hidden output for full connection layers.
        hidden_history, (_, _) = self.his_encoder(history)  # [batch_size, seq_len, d_history]

        hidden_history = repeat(hidden_history, "b s dh -> b s ne dh", ne=self.num_marks)
        # [batch_size, seq_len, num_marks, d_history]

        time_next = repeat(time_next, "... -> ... ne", ne=self.num_marks)  # [..., batch_size, seq_len, num_marks]
        time_next.requires_grad = True
        time_next_scaled = (time_next - mean) / std  # [..., batch_size, seq_len, num_marks]
        time_embedding = time_next_scaled.unsqueeze(dim=-1) * self.nonneg_activation(self.weight_for_t)
        # [..., batch_size, seq_len, num_marks, d_intensity]

        hidden_history = self.history_mapper(hidden_history)  # [batch_size, seq_len, num_marks, d_intensity]
        time_embedding = self.time_mapper(time_embedding)  # [..., batch_size, seq_len, num_marks, d_intensity]
        hidden_history = rearrange(
            hidden_history, f"... -> {'() ' * (len(time_embedding.shape) - len(hidden_history.shape))}..."
        )
        # [..., batch_size, seq_len, num_marks, d_intensity]
        output = self.layer_activation(
            time_embedding + hidden_history
        )  # [..., batch_size, seq_len, num_marks, d_intensity]

        for nonneg_layer in self.mlp:
            output = nonneg_layer(output)  # [..., batch_size, seq_len, num_marks, d_intensity]
            output = self.layer_activation(output)  # [..., batch_size, seq_len, num_marks, d_intensity]

        integral_for_each_mark = self.nonneg_activation(
            self.aggregate(output)
        )  # [..., batch_size, seq_len, num_marks, 1]
        integral_for_each_mark = integral_for_each_mark.squeeze(dim=-1)  # [..., batch_size, seq_len, num_marks]

        intensity_for_each_mark = torch.autograd.grad(
            outputs=integral_for_each_mark,
            inputs=time_next,
            grad_outputs=torch.ones_like(integral_for_each_mark),
            create_graph=training,
        )[0]
        check_tensor(intensity_for_each_mark)  # [batch_size, seq_len, num_marks]
        time_next.requires_grad = False

        return integral_for_each_mark, intensity_for_each_mark

    def get_event_embedding(self, input_event):
        """
        Get mark embeddings for input_event.

        ### Args
            * ```torch.tensor``` input_event
              shape: ```[batch_size, seq_len]```
              Input event sequence.
        ### Outputs
            * ```torch.tensor```
              shape: ```[batch_size, seq_len, d_history]```
              The output embeddings.
        """
        return self.marks(input_event)  # [batch_size, seq_len, d_history]

    def integral_intensity_time_next_2d(
        self, time_history, time_next, marks_history, integration_sample_rate, mean, std,
    ):
        """
        Probe the value of the intensity function and its integral at sampled timestamps.
        In this function, all marks share the sampled timestmaps, so the dimension of time_next does not include num_event.

        ### Args
            * ```torch.tensor``` marks_history
              shape: ```[batch_size, seq_len]```
              Historical event sequence.
            * ```torch.tensor``` time_history
              shape: ```[batch_size, seq_len]```
              Historical time sequence.
            * ```torch.tensor``` time_next
              shape: ```[..., batch_size, seq_len]```
              Guessed or real time when the next event will happen.
            * ```int``` resolution
              The number of interpolated points in a time interval between two adjoint marks for integration estimation.
              The number of interpolated points counts the start and end point of the interval.
            * ```float``` mean
            * ```float``` std
              Used for input time scaling.
            * ```torch,tensor``` time_next_start
              shape: ```[..., batch_size, seq_len]``` if not None
              When given, this function computes the integral between [time_next_start, t_i]. time_next_start are expected to be non-negative.
              This affects the integral, intensity, and timestamp.
        ### Outputs
            * ```torch.tensor``` expand_integral
              shape: ```[..., batch_size, seq_len, resolution, num_marks]```
              The value of \\Lambda^*(m, t) at sampled times.
            * ```torch.tensor``` expand_intensity
              shape: ```[..., batch_size, seq_len, resolution, num_marks]```
              The value of \\lambda^*(m, t) at sampled times.
            * ```torch.tensor``` original_time_expand
              shape: ```[..., batch_size, seq_len, resolution]```
              The value of sampled times.
        """
        # Prepare the history embedding.
        time_history = (time_history - mean) / std  # [batch_size, seq_len]

        marks_embeddings = self.marks(marks_history)  # [batch_size, seq_len, d_history]
        history = torch.cat([marks_embeddings, time_history.unsqueeze(dim=-1)], dim=-1)
        # [batch_size, seq_len, d_history + 1]

        hidden_history, (_, _) = self.his_encoder(history)  # [batch_size, seq_len, d_history]
        hidden_history = self.history_mapper(hidden_history)  # [batch_size, seq_len, d_intensity]

        hidden_history = repeat(hidden_history, "b s di -> b s r ne di", r=integration_sample_rate, ne=self.num_marks)
        # [batch_size, seq_len, resolution, num_marks, d_intensity]

        # Prepare the time embedding.
        time_multiplier = torch.linspace(0, 1, integration_sample_rate, device=self.device)
        # [resolution]
        original_time_expand = time_next.unsqueeze(dim=-1) * time_multiplier  # [batch_size, seq_len, resolution]
        time_expand = original_time_expand.clone()  # [batch_size, seq_len, resolution]
        time_expand = repeat(original_time_expand, "b s r -> b s r ne", ne=self.num_marks)
        # [batch_size, seq_len, resolution, num_marks]
        time_expand.requires_grad = True
        normed_time_expand = (time_expand - mean) / std  # [batch_size, seq_len, resolution, num_marks]

        emb_normed_time_expand = normed_time_expand.unsqueeze(dim=-1) * self.nonneg_activation(self.weight_for_t)
        # [batch_size, seq_len, resolution, num_marks, d_intensity]

        emb_normed_time_expand = self.time_mapper(
            emb_normed_time_expand
        )  # [batch_size, seq_len, resolution, num_marks, d_intensity]
        output = self.layer_activation(
            emb_normed_time_expand + hidden_history
        )  # [batch_size, seq_len, resolution, num_marks, d_intensity]

        # Get intensity integrals.
        for nonneg_layer in self.mlp:
            output = nonneg_layer(output)  # [batch_size, seq_len, resolution, num_marks, d_intensity]
            output = self.layer_activation(output)  # [batch_size, seq_len, resolution, num_marks, d_intensity]

        expand_integral = self.nonneg_activation(
            self.aggregate(output)
        )  # [batch_size, seq_len, resolution, num_marks, 1]

        # Get intensity values at every sampled $ t $.
        expand_intensity = torch.autograd.grad(
            outputs=expand_integral,
            inputs=time_expand,
            grad_outputs=torch.ones_like(expand_integral),
        )[0]  # [batch_size, seq_len, resolution, num_marks]
        time_expand.requires_grad = False

        expand_integral = expand_integral.squeeze(dim=-1).detach()  # [batch_size, seq_len, resolution, num_marks]
        expand_intensity = expand_intensity.detach()  # [batch_size, seq_len, resolution, num_marks]

        return expand_integral, expand_intensity, original_time_expand

    def integral_intensity_time_next_3d(self, time_history, time_next, marks_history, integration_sample_rate, mean, std):
        """
        Probe the value of the intensity function and its integral at sampled timestamps.
        In this function, all marks can have their sampled timestmaps, so the dimension of time_next is ```[..., batch_size, seq_len, num_marks]```.
        This function is supposed to be much slower than integral_intensity_time_next_2d().

        ### Args
            * ```torch.tensor``` marks_history
              shape: ```[batch_size, seq_len]```
              Historical event sequence.
            * ```torch.tensor``` time_history
              shape: ```[batch_size, seq_len]```
              Historical time sequence.
            * ```torch.tensor``` time_next
              shape: ```[..., batch_size, seq_len, num_marks]```
              Guessed or real time when the next event will happen.
            * ```int``` resolution
              The number of interpolated points in a time interval between two adjoint marks for integration estimation.
              The number of interpolated points counts the start and end point of the interval.
            * ```float``` mean
            * ```float``` std
              Used for input time scaling.
        ### Outputs
            * ```torch.tensor``` expand_integral
              shape: ```[..., batch_size, seq_len, resolution, num_marks]```
              The value of \\Lambda^*(m, t) at sampled times.
            * ```torch.tensor``` expand_intensity
              shape: ```[..., batch_size, seq_len, resolution, num_marks]```
              The value of \\lambda^*(m, t) at sampled times.
            * ```torch.tensor``` original_time_expand
              shape: ```[..., batch_size, seq_len, resolution]```
              The value of sampled times.
        """
        # Prepare the history embedding.
        time_history = (time_history - mean) / std  # [batch_size, seq_len]

        marks_embeddings = self.marks(marks_history)  # [batch_size, seq_len, d_history]
        history = torch.cat([marks_embeddings, time_history.unsqueeze(dim=-1)], dim=-1)
        # [batch_size, seq_len, d_history + 1]

        hidden_history, (_, _) = self.his_encoder(history)  # [batch_size, seq_len, d_history]
        hidden_history = self.history_mapper(hidden_history)  # [batch_size, seq_len, d_intensity]

        hidden_history = repeat(
            hidden_history, "b s di -> b s r ne ne1 di", r=integration_sample_rate, ne=self.num_marks, ne1=self.num_marks
        )
        # [batch_size, seq_len, resolution, num_marks, num_marks, d_intensity]

        # Prepare the time embedding.
        time_multiplier = torch.linspace(0, 1, integration_sample_rate, device=self.device)
        # [resolution]
        original_time_expand = time_next.unsqueeze(dim=-2) * rearrange(
            time_multiplier, f"r -> {'() ' * (len(time_next.shape) - 1)}r 1"
        )
        # [..., batch_size, seq_len, resolution, num_marks]
        time_expand = repeat(original_time_expand.clone(), "... -> ... ne", ne=self.num_marks)
        # [..., batch_size, seq_len, resolution, num_marks, num_marks]
        time_expand.requires_grad = True
        normed_time_expand = (time_expand - mean) / std  # [..., batch_size, seq_len, resolution, num_marks, num_marks]

        emb_normed_time_expand = normed_time_expand.unsqueeze(dim=-1) * self.nonneg_activation(self.weight_for_t)
        # [..., batch_size, seq_len, resolution, num_marks, num_marks, d_intensity]
        emb_normed_time_expand = self.time_mapper(
            emb_normed_time_expand
        )  # [..., batch_size, seq_len, resolution, num_marks, num_marks, d_intensity]
        hidden_history = rearrange(
            hidden_history, f"... -> {'() ' * (len(emb_normed_time_expand.shape) - len(hidden_history.shape))} ..."
        )
        # [..., batch_size, seq_len, resolution, num_marks, num_marks, d_intensity]
        output = self.layer_activation(
            emb_normed_time_expand + hidden_history
        )  # [..., batch_size, seq_len, resolution, num_marks, num_marks, d_intensity]

        # Get intensity integrals.
        for nonneg_layer in self.mlp:
            output = nonneg_layer(output)  # [..., batch_size, seq_len, resolution, num_marks, num_marks, d_intensity]
            output = self.layer_activation(
                output
            )  # [..., batch_size, seq_len, resolution, num_marks, num_marks, d_intensity]

        expand_integral = self.nonneg_activation(
            self.aggregate(output)
        )  # [..., batch_size, seq_len, resolution, num_marks, num_marks, 1]

        # Get intensity values at every sampled $ t $.
        expand_intensity = torch.autograd.grad(
            outputs=expand_integral,
            inputs=time_expand,
            grad_outputs=torch.ones_like(expand_integral),
        )[0]  # [..., batch_size, seq_len, resolution, num_marks, num_marks]
        time_expand.requires_grad = False

        expand_integral = expand_integral.squeeze(
            dim=-1
        ).detach()  # [..., batch_size, seq_len, resolution, num_marks, num_marks]
        expand_intensity = expand_intensity.detach()  # [..., batch_size, seq_len, resolution, num_marks, num_marks]

        return expand_integral, expand_intensity, original_time_expand

    def model_probe_function(self, time_history, time_next, marks_history, mask_next, resolution, mean, std):
        """
        Probe the value of the intensity function and its integral at sampled timestamps.
        In this function, all marks can have their sampled timestmaps, so the dimension of time_next is ```[..., batch_size, seq_len, num_marks]```.
        This function is supposed to be much slower than integral_intensity_time_next_2d().

        ### Args
            * ```torch.tensor``` marks_history
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
        # Prepare the history embedding.
        time_history = (time_history - mean) / std  # [batch_size, seq_len]

        marks_embeddings = self.marks(marks_history)  # [batch_size, seq_len, d_history]
        history = torch.cat([marks_embeddings, time_history.unsqueeze(dim=-1)], dim=-1)
        # [batch_size, seq_len, d_history + 1]

        hidden_history, (_, _) = self.his_encoder(history)  # [batch_size, seq_len, d_history]
        hidden_history = self.history_mapper(hidden_history)  # [batch_size, seq_len, d_intensity]

        hidden_history = repeat(hidden_history, "b s di -> b s r ne di", r=resolution, ne=self.num_marks)
        # [batch_size, seq_len, resolution, num_marks, d_intensity]

        # Prepare the time embedding.
        time_multiplier = torch.linspace(0, 1, resolution, device=self.device)
        # [resolution]
        original_time_expand = time_next.unsqueeze(dim=-1) * time_multiplier  # [batch_size, seq_len, resolution]
        time_expand = original_time_expand.clone()  # [batch_size, seq_len, resolution]
        time_expand = repeat(original_time_expand, "b s r -> b s r ne", ne=self.num_marks)
        # [batch_size, seq_len, resolution, num_marks]

        time_expand.requires_grad = True
        normed_time_expand = (time_expand - mean) / std  # [batch_size, seq_len, resolution, num_marks]

        emb_normed_time_expand = normed_time_expand.unsqueeze(dim=-1) * self.nonneg_activation(self.weight_for_t)
        # [batch_size, seq_len, resolution, num_marks, d_intensity]

        emb_normed_time_expand = self.time_mapper(
            emb_normed_time_expand
        )  # [batch_size, seq_len, resolution, num_marks, d_intensity]
        output = self.layer_activation(
            emb_normed_time_expand + hidden_history
        )  # [batch_size, seq_len, resolution, num_marks, d_intensity]

        # Get intensity integrals.
        for nonneg_layer in self.mlp:
            output = nonneg_layer(output)  # [batch_size, seq_len, resolution, num_marks, d_intensity]
            output = self.layer_activation(output)  # [batch_size, seq_len, resolution, num_marks, d_intensity]

        expand_integral = self.nonneg_activation(
            self.aggregate(output)
        )  # [batch_size, seq_len, resolution, num_marks, 1]

        expand_intensity = torch.autograd.grad(
            outputs=expand_integral,
            inputs=time_expand,
            grad_outputs=torch.ones_like(expand_integral),
            retain_graph=True,
        )[0]  # [batch_size, seq_len, resolution, num_marks]

        time_expand.requires_grad = False

        expand_integral = expand_integral.squeeze(dim=-1)  # [batch_size, seq_len, resolution, num_marks]

        # The data dict is defined here.
        # This dict should pack all data required by plot().
        data = {}
        data["expand_intensity_for_each_mark"] = expand_intensity  # [batch_size, seq_len, resolution, num_marks]
        data["expand_integral_for_each_mark"] = expand_integral  # [batch_size, seq_len, resolution, num_marks]

        expand_intensity = rearrange(expand_intensity, "b s r ne -> b (s r) ne")
        # [batch_size, seq_len * resolution, num_event]
        expand_integral = rearrange(
            expand_integral, "b s r ne -> b (s r) ne"
        )  # [batch_size, seq_len * resolution, num_event]

        spearman_matrix = []
        pearson_matrix = []
        l1_matrix = []
        for idx, (
            expand_intensity_per_seq,
            expand_integral_per_seq,
            mask_per_seq,
            original_time_expand_per_seq,
        ) in enumerate(zip(expand_intensity, expand_integral, mask_next, original_time_expand)):
            seq_len = mask_per_seq.sum()
            probability_distribution = expand_intensity_per_seq * torch.exp(-expand_integral_per_seq)
            probability_distribution = move_from_tensor_to_ndarray(probability_distribution)

            # rho: spearman coefficient
            if self.num_marks == 1:
                spearman_matrix_per_seq = np.array([[1.0]])
            else:
                spearman_matrix_per_seq = spearmanr(probability_distribution[: seq_len * resolution])[0]
                if self.num_marks == 2:
                    spearman_matrix_per_seq = np.array([[1, spearman_matrix_per_seq], [spearman_matrix_per_seq, 1]])

            # r: pearson coefficient
            pearson_matrix_per_seq = np.corrcoef(probability_distribution[: seq_len * resolution], rowvar=False)
            if self.num_marks == 1:
                pearson_matrix_per_seq = rearrange(np.array(pearson_matrix_per_seq), " -> () ()")

            # L^1 metric
            l1_matrix_per_seq = L1_distance_across_marks(
                probability_distribution[: seq_len * resolution],
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
