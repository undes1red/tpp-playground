import torch
import torch.nn as nn
from einops import rearrange, repeat

from src.toolbox.algorithms import evaluate_on_one_batch
from src.toolbox.misc import check_tensor, compile_model, move_from_tensor_to_ndarray
from src.toolbox.modules import NonNegLinear


class FENN(nn.Module):
    def __init__(
        self,
        use_compile,
        compile_backend,
        d_intensity,
        num_marks,
        dropout,
        history_module,
        history_module_layers,
        mlp_layers,
        device,
    ):
        """
        This function creates a FENN model.

        ### Args
            * ```str``` history_module
              Which RNN model do we use to encode the history? Default is LSTM. We don't recommend to change it to something else.
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
        self.use_compile = use_compile
        self.compile_backend = compile_backend
        self.num_marks = num_marks

        self.marks = nn.Embedding(num_marks + 1, d_intensity, padding_idx=num_marks, device=device)

        self.his_encoder = getattr(nn, history_module)(
            input_size=d_intensity,
            hidden_size=d_intensity,
            num_layers=history_module_layers,
            batch_first=True,
            dropout=dropout,
            device=device,
        )
        self.his_encoder = compile_model(self.his_encoder, self.use_compile, self.compile_backend)

        # Map the time number into a vector.
        self.weight_for_t = nn.Parameter(
            torch.zeros((self.num_marks, d_intensity), device=self.device, requires_grad=True)
        )
        nn.init.xavier_uniform_(self.weight_for_t)

        # Map history and time embeddings into the same hidden space.
        self.history_mapper = nn.Linear(d_intensity, d_intensity, bias=True, device=device)
        self.time_mapper = NonNegLinear(d_intensity, d_intensity, device=self.device)

        # IEM module featuring non-negative fully connected layers.
        self.mlp = nn.ModuleList(
            [NonNegLinear(d_intensity, d_intensity, bias=True, device=device) for _ in range(mlp_layers)]
        )
        self.layer_activation = nn.Tanh()
        self.aggregate = NonNegLinear(d_intensity, 1, bias=True, device=device)
        self.nonneg_activation = nn.Softplus()

    def history_seq_encoding(self, time_history, marks_history, mean, std, custom_marks_history=False):
        time_history = (time_history - mean) / std  # [batch_size, seq_len]

        marks_embeddings = marks_history if custom_marks_history else self.marks(marks_history)
        # [batch_size, seq_len, d_intensity]
        time_embeddings = torch.zeros_like(marks_embeddings)
        # [batch_size, seq_len, d_intensity]

        time_history = repeat(time_history, "... -> ... ne", ne=self.num_marks)  # [batch_size, seq_len, num_marks]
        time_history_emb = time_history.unsqueeze(dim=-1) * self.nonneg_activation(self.weight_for_t)
        # [batch_size, seq_len, num_marks, d_intensity]
        time_history_emb = (
            time_history_emb[..., 1:, :, :]
            * nn.functional.one_hot(marks_history[..., 1:], num_classes=self.num_marks).unsqueeze(dim=-1)
        ).sum(dim=-2)
        # [batch_size, seq_len-1, d_intensity]
        time_embeddings[..., 1:, :] = time_history_emb
        # [batch_size, seq_len, d_intensity]
        return marks_embeddings + time_embeddings  # [batch_size, seq_len, d_intensity]

    def forward(self, time_history, time_next, marks_history, mean, std, custom_marks_history=False, training=False):
        """
        FENNModel's forwardpropagation function for training.

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
            * ```torch.tensor``` integral
              shape: ```[..., batch_size, seq_len, num_marks]```
              The value of \\Lambda^*(m, t) on [t_{i-1}, t_i).
        """
        history = self.history_seq_encoding(time_history, marks_history, mean, std, custom_marks_history)
        # [batch_size, seq_len, d_intensity]

        # Reshape hidden output for full connection layers.
        hidden_history, (_, _) = self.his_encoder(history)  # [batch_size, seq_len, d_intensity]

        hidden_history = repeat(hidden_history, "b s dh -> b s ne dh", ne=self.num_marks)
        # [batch_size, seq_len, num_marks, d_intensity]

        time_next = repeat(time_next, "... -> ... ne", ne=self.num_marks)  # [..., batch_size, seq_len, num_marks]
        time_next_requires_grad = time_next.requires_grad
        if not time_next_requires_grad:
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
        if not time_next_requires_grad:
            time_next.requires_grad = False

        return integral_for_each_mark, intensity_for_each_mark

    def get_mark_embedding(self, input_mark):
        """
        Get mark embeddings for input_mark.

        ### Args
            * ```torch.tensor``` input_mark
              shape: ```[batch_size, seq_len]```
              Input mark sequence.
        ### Outputs
            * ```torch.tensor```
              shape: ```[batch_size, seq_len, d_intensity]```
              The output embeddings.
        """
        return self.marks(input_mark)  # [batch_size, seq_len, d_intensity]

    def integral_intensity_next_one_event_time_next_1d(
        self,
        time_history,
        time_next,
        marks_history,
        mean,
        std,
        resolution=None,
        time_next_start=None,
        only_value_at_time_next=False,
        time_next_with_resolution_dim=False,
    ):
        """
        FENN's forwardpropagation function specific for sampling time first then mark.

        ### Args
            * ```torch.tensor``` marks_history
              shape: ```[number_of_sampled_sequences, sampled_seq_len]```
              Historical mark sequence.
            * ```torch.tensor``` time_history
              shape: ```[number_of_sampled_sequences, sampled_seq_len]```
              Historical time sequence.
            * ```torch.tensor``` time_next
              shape: ```[number_of_sampled_sequences, num_marks]```
              Guessed time when the next mark will happen.
        ### Outputs
            * ```torch.tensor``` integral_all_marks
              shape: ```[number_of_sampled_sequences, num_marks]```
              The value of \\Lambda^*(m, t) on [t_{i-1}, t_i).
            * ```torch.tensor``` intensity_all_marks
              shape: ```[number_of_sampled_sequences, num_marks]```
              The value of \\lambda^*(m, t) on at t_i.
        """
        if only_value_at_time_next:
            resolution = 2

        # Prepare the history embedding.
        if time_next_start is None:
            time_next_start = torch.zeros_like(time_next)  # [..., batch_size, seq_len]

        history = self.history_seq_encoding(time_history, marks_history, mean, std)
        # [number_of_sampled_sequences, seq_len, d_intensity]

        # Reshape hidden output for full connection layers.
        _, (hidden_history, _) = self.his_encoder(history)  # [1, number_of_sampled_sequences, d_hidden]

        hidden_history = rearrange(hidden_history, "a nss dh -> nss a dh")
        # [number_of_sampled_sequences, 1, d_hidden]

        # Prepare the time embedding.
        if time_next_with_resolution_dim:
            original_time_expand = time_next
            # [number_of_sampled_sequences, resolution]
        else:
            time_multiplier = torch.linspace(0, 1, resolution, device=self.device)
            # [resolution]
            original_time_expand = (time_next - time_next_start).unsqueeze(
                dim=-1
            ) * time_multiplier + time_next_start.unsqueeze(dim=-1)
            # [number_of_sampled_sequences, resolution]

        expanded_time_next = repeat(
            original_time_expand, "... -> ... ne", ne=self.num_marks
        )  # [number_of_sampled_sequences, resolution, num_marks]

        expanded_time_next.requires_grad = True
        expanded_time_next_scaled = (
            expanded_time_next - mean
        ) / std  # [number_of_sampled_sequences, resolution, num_marks]
        time_embedding = expanded_time_next_scaled.unsqueeze(dim=-1) * self.nonneg_activation(self.weight_for_t)
        # [number_of_sampled_sequences, resolution, num_marks, d_intensity]

        hidden_history = self.history_mapper(hidden_history)  # [number_of_sampled_sequences, 1, d_intensity]
        time_embedding = self.time_mapper(
            time_embedding
        )  # [number_of_sampled_sequences, resolution, num_marks, d_intensity]
        output = self.layer_activation(
            time_embedding + hidden_history.unsqueeze(dim=-2)
        )  # [number_of_sampled_sequences, resolution, num_marks, d_intensity]

        for nonneg_layer in self.mlp:
            output = nonneg_layer(output)  # [number_of_sampled_sequences, resolution, num_marks, d_intensity]
            output = self.layer_activation(output)  # [number_of_sampled_sequences, resolution, num_marks, d_intensity]

        integral_for_each_mark = self.nonneg_activation(
            self.aggregate(output)
        )  # [number_of_sampled_sequences, resolution, num_marks, 1]
        integral_for_each_mark = integral_for_each_mark.squeeze(
            dim=-1
        )  # [number_of_sampled_sequences, resolution, num_marks]

        intensity_for_each_mark = torch.autograd.grad(
            outputs=integral_for_each_mark,
            inputs=expanded_time_next,
            grad_outputs=torch.ones_like(integral_for_each_mark),
        )[0]
        check_tensor(intensity_for_each_mark)  # [number_of_sampled_sequences, resolution, num_marks]
        expanded_time_next.requires_grad = False

        if only_value_at_time_next:
            return (
                integral_for_each_mark[:, -1, :].detach(),
                intensity_for_each_mark[:, -1, :].detach(),
                expanded_time_next,
            )

        return integral_for_each_mark.detach(), intensity_for_each_mark.detach(), original_time_expand

    def integral_intensity_next_one_event_time_next_2d(
        self,
        time_history,
        time_next,
        marks_history,
        mean,
        std,
        resolution=None,
        time_next_start=None,
        time_next_with_resolution_dim=False,
    ):
        """
        FENN's forwardpropagation function specific for sampling mark first then time.

        ### Args
            * ```torch.tensor``` marks_history
              shape: ```[number_of_sampled_sequences, sampled_seq_len]```
              Historical mark sequence.
            * ```torch.tensor``` time_history
              shape: ```[number_of_sampled_sequences, sampled_seq_len]```
              Historical time sequence.
            * ```torch.tensor``` time_next
              shape: ```[number_of_sampled_sequences, num_marks]```
              Guessed time when the next mark will happen.
        ### Outputs
            * ```torch.tensor``` integral_all_marks
              shape: ```[number_of_sampled_sequences, num_marks]```
              The value of \\Lambda^*(m, t) on [t_{i-1}, t_i).
            * ```torch.tensor``` intensity_all_marks
              shape: ```[number_of_sampled_sequences, num_marks]```
              The value of \\lambda^*(m, t) on at t_i.
        """
        # Prepare the history embedding.
        if time_next_start is None:
            time_next_start = torch.zeros_like(time_next)  # [..., batch_size, seq_len]

        # Prepare the history embedding.
        history = self.history_seq_encoding(time_history, marks_history, mean, std)
        # [number_of_sampled_sequences, seq_len, d_intensity]

        _, (hidden_history, _) = self.his_encoder(history)  # [1, number_of_sampled_sequences, d_hidden]

        hidden_history = rearrange(hidden_history, "a nss dh -> nss a dh")
        # [number_of_sampled_sequences, 1, d_hidden]
        hidden_history = self.history_mapper(hidden_history)  # [number_of_sampled_sequences, 1, d_intensity]
        hidden_history = rearrange(
            hidden_history,
            "b () di -> b () () () di",
        )
        # [number_of_sampled_sequences, num_marks, resolution, num_marks, d_intensity]

        # Prepare the time embedding.
        if time_next_with_resolution_dim:
            original_time_expand = time_next
            # [number_of_sampled_sequences, num_marks, resolution]
        else:
            time_multiplier = torch.linspace(0, 1, resolution, device=self.device)
            # [resolution]
            original_time_expand = (time_next - time_next_start).unsqueeze(
                dim=-1
            ) * time_multiplier + time_next_start.unsqueeze(dim=-1)
            # [number_of_sampled_sequences, num_marks, resolution]

        time_expand = repeat(original_time_expand.clone(), "... -> ... ne", ne=self.num_marks)
        # [number_of_sampled_sequences, num_marks, resolution, num_marks]
        time_expand.requires_grad = True
        normed_time_expand = (
            time_expand - mean
        ) / std  # [number_of_sampled_sequences, num_marks, resolution, num_marks]

        emb_normed_time_expand = normed_time_expand.unsqueeze(dim=-1) * self.nonneg_activation(self.weight_for_t)
        # [number_of_sampled_sequences, num_marks, resolution, num_marks, d_intensity]
        emb_normed_time_expand = self.time_mapper(
            emb_normed_time_expand
        )  # [number_of_sampled_sequences, num_marks, resolution, num_marks, d_intensity]
        output = self.layer_activation(
            emb_normed_time_expand + hidden_history
        )  # [number_of_sampled_sequences, num_marks, resolution, num_marks, d_intensity]

        # Get intensity integrals.
        for nonneg_layer in self.mlp:
            output = nonneg_layer(
                output
            )  # [number_of_sampled_sequences, num_marks, resolution, num_marks, d_intensity]
            output = self.layer_activation(
                output
            )  # [number_of_sampled_sequences, num_marks, resolution, num_marks, d_intensity]

        expand_integral = self.nonneg_activation(
            self.aggregate(output)
        )  # [number_of_sampled_sequences, num_marks, resolution, num_marks, 1]

        # Get intensity values at every sampled $ t $.
        expand_intensity = torch.autograd.grad(
            outputs=expand_integral,
            inputs=time_expand,
            grad_outputs=torch.ones_like(expand_integral),
        )[0]  # [number_of_sampled_sequences, num_marks, resolution, num_marks]
        time_expand.requires_grad = False

        expand_integral = expand_integral.squeeze(
            dim=-1
        ).detach()  # [number_of_sampled_sequences, num_marks, resolution, num_marks]
        expand_intensity = expand_intensity.detach()  # [number_of_sampled_sequences, num_marks, resolution, num_marks]

        return expand_integral, expand_intensity, original_time_expand

    @torch.inference_mode()
    def integral_time_next_2d(
        self,
        time_history,
        time_next,
        marks_history,
        mean,
        std,
        resolution=None,
        time_next_start=None,
        time_next_with_resolution_dim=False,
    ):
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
        if time_next_start is None:
            time_next_start = torch.zeros_like(time_next)  # [..., batch_size, seq_len]

        history = self.history_seq_encoding(time_history, marks_history, mean, std)
        # [batch_size, seq_len, d_intensity]

        hidden_history, (_, _) = self.his_encoder(history)  # [batch_size, seq_len, d_intensity]
        hidden_history = self.history_mapper(hidden_history)  # [batch_size, seq_len, d_intensity]

        einop = f"b s di -> {'() ' * (len(time_next.shape) - (3 if time_next_with_resolution_dim else 2))}b s () () di"
        hidden_history = rearrange(
            hidden_history, einop
        )  # [..., batch_size, seq_len, resolution, num_marks, d_intensity]

        # Prepare the time embedding.
        if time_next_with_resolution_dim:
            original_time_expand = time_next
        else:
            time_multiplier = torch.linspace(0, 1, resolution, device=self.device)
            # [resolution]
            original_time_expand = (time_next - time_next_start).unsqueeze(
                dim=-1
            ) * time_multiplier + time_next_start.unsqueeze(dim=-1)
            # [..., batch_size, seq_len, resolution]

        time_expand = original_time_expand.clone()  # [..., batch_size, seq_len, resolution]
        time_expand = repeat(original_time_expand, "... -> ... ne", ne=self.num_marks)
        # [..., batch_size, seq_len, resolution, num_marks]
        normed_time_expand = (time_expand - mean) / std  # [..., batch_size, seq_len, resolution, num_marks]

        emb_normed_time_expand = normed_time_expand.unsqueeze(dim=-1) * self.nonneg_activation(self.weight_for_t)
        # [..., batch_size, seq_len, resolution, num_marks, d_intensity]

        emb_normed_time_expand = self.time_mapper(
            emb_normed_time_expand
        )  # [..., batch_size, seq_len, resolution, num_marks, d_intensity]
        output = self.layer_activation(
            emb_normed_time_expand + hidden_history
        )  # [..., batch_size, seq_len, resolution, num_marks, d_intensity]

        # Get intensity integrals.
        for nonneg_layer in self.mlp:
            output = nonneg_layer(output)  # [..., batch_size, seq_len, resolution, num_marks, d_intensity]
            output = self.layer_activation(output)  # [..., batch_size, seq_len, resolution, num_marks, d_intensity]

        expand_integral = self.nonneg_activation(
            self.aggregate(output)
        )  # [..., batch_size, seq_len, resolution, num_marks, 1]

        expand_integral = expand_integral.squeeze(dim=-1).detach()  # [..., batch_size, seq_len, resolution, num_marks]

        return expand_integral, original_time_expand

    def integral_intensity_time_next_2d(
        self,
        time_history,
        time_next,
        marks_history,
        mean,
        std,
        resolution=None,
        time_next_start=None,
        time_next_with_resolution_dim=False,
    ):
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
        if time_next_start is None:
            time_next_start = torch.zeros_like(time_next)  # [..., batch_size, seq_len]

        history = self.history_seq_encoding(time_history, marks_history, mean, std)
        # [batch_size, seq_len, d_intensity]

        hidden_history, (_, _) = self.his_encoder(history)  # [batch_size, seq_len, d_intensity]
        hidden_history = self.history_mapper(hidden_history)  # [batch_size, seq_len, d_intensity]

        einop = f"b s di -> {'() ' * (len(time_next.shape) - (3 if time_next_with_resolution_dim else 2))}b s () () di"
        hidden_history = rearrange(
            hidden_history, einop
        )  # [..., batch_size, seq_len, resolution, num_marks, d_intensity]

        # Prepare the time embedding.
        if time_next_with_resolution_dim:
            original_time_expand = time_next
        else:
            time_multiplier = torch.linspace(0, 1, resolution, device=self.device)
            # [resolution]
            original_time_expand = (time_next - time_next_start).unsqueeze(
                dim=-1
            ) * time_multiplier + time_next_start.unsqueeze(dim=-1)
            # [..., batch_size, seq_len, resolution]

        time_expand = original_time_expand.clone()  # [..., batch_size, seq_len, resolution]
        time_expand = repeat(original_time_expand, "... -> ... ne", ne=self.num_marks)
        # [..., batch_size, seq_len, resolution, num_marks]
        time_expand.requires_grad = True
        normed_time_expand = (time_expand - mean) / std  # [..., batch_size, seq_len, resolution, num_marks]

        emb_normed_time_expand = normed_time_expand.unsqueeze(dim=-1) * self.nonneg_activation(self.weight_for_t)
        # [..., batch_size, seq_len, resolution, num_marks, d_intensity]

        emb_normed_time_expand = self.time_mapper(
            emb_normed_time_expand
        )  # [..., batch_size, seq_len, resolution, num_marks, d_intensity]
        output = self.layer_activation(
            emb_normed_time_expand + hidden_history
        )  # [..., batch_size, seq_len, resolution, num_marks, d_intensity]

        # Get intensity integrals.
        for nonneg_layer in self.mlp:
            output = nonneg_layer(output)  # [..., batch_size, seq_len, resolution, num_marks, d_intensity]
            output = self.layer_activation(output)  # [..., batch_size, seq_len, resolution, num_marks, d_intensity]

        expand_integral = self.nonneg_activation(
            self.aggregate(output)
        )  # [..., batch_size, seq_len, resolution, num_marks, 1]

        # Get intensity values at every sampled $ t $.
        expand_intensity = torch.autograd.grad(
            outputs=expand_integral,
            inputs=time_expand,
            grad_outputs=torch.ones_like(expand_integral),
        )[0]  # [..., batch_size, seq_len, resolution, num_marks]
        time_expand.requires_grad = False

        expand_integral = expand_integral.squeeze(dim=-1).detach()  # [..., batch_size, seq_len, resolution, num_marks]
        expand_intensity = expand_intensity.detach()  # [..., batch_size, seq_len, resolution, num_marks]

        return expand_integral, expand_intensity, original_time_expand

    def integral_intensity_time_next_3d(
        self,
        time_history,
        time_next,
        marks_history,
        mean,
        std,
        resolution=None,
        time_next_start=None,
        time_next_with_resolution_dim=False,
    ):
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
        if time_next_start is None:
            time_next_start = torch.zeros_like(time_next)  # [..., batch_size, seq_len]

        # Prepare the history embedding.
        history = self.history_seq_encoding(time_history, marks_history, mean, std)
        # [batch_size, seq_len, d_intensity]

        hidden_history, (_, _) = self.his_encoder(history)  # [batch_size, seq_len, d_intensity]
        hidden_history = self.history_mapper(hidden_history)  # [batch_size, seq_len, d_intensity]

        hidden_history = repeat(
            hidden_history,
            "b s di -> b s ne r ne1 di",
            r=resolution,
            ne=self.num_marks,
            ne1=self.num_marks,
        )
        # [batch_size, seq_len, num_marks, resolution, num_marks, d_intensity]

        # Prepare the time embedding.
        if time_next_with_resolution_dim:
            original_time_expand = time_next
            # [..., batch_size, seq_len, num_marks, resolution]
        else:
            time_multiplier = torch.linspace(0, 1, resolution, device=self.device)
            # [resolution]
            original_time_expand = (time_next - time_next_start).unsqueeze(
                dim=-1
            ) * time_multiplier + time_next_start.unsqueeze(dim=-1)
            # [..., batch_size, seq_len, num_marks, resolution]

        time_expand = repeat(original_time_expand.clone(), "... -> ... ne", ne=self.num_marks)
        # [..., batch_size, seq_len, num_marks, resolution, num_marks]
        time_expand.requires_grad = True
        normed_time_expand = (time_expand - mean) / std  # [..., batch_size, seq_len, num_marks, resolution, num_marks]

        emb_normed_time_expand = normed_time_expand.unsqueeze(dim=-1) * self.nonneg_activation(self.weight_for_t)
        # [..., batch_size, seq_len, num_marks, resolution, num_marks, d_intensity]
        emb_normed_time_expand = self.time_mapper(
            emb_normed_time_expand
        )  # [..., batch_size, seq_len, resolution, num_marks, num_marks, d_intensity]
        hidden_history = rearrange(
            hidden_history, f"... -> {'() ' * (len(emb_normed_time_expand.shape) - len(hidden_history.shape))} ..."
        )
        # [..., batch_size, seq_len, num_marks, resolution, num_marks, d_intensity]
        output = self.layer_activation(
            emb_normed_time_expand + hidden_history
        )  # [..., batch_size, seq_len, num_marks, resolution, num_marks, d_intensity]

        # Get intensity integrals.
        for nonneg_layer in self.mlp:
            output = nonneg_layer(output)  # [..., batch_size, seq_len, num_marks, resolution, num_marks, d_intensity]
            output = self.layer_activation(
                output
            )  # [..., batch_size, seq_len, num_marks, resolution, num_marks, d_intensity]

        expand_integral = self.nonneg_activation(
            self.aggregate(output)
        )  # [..., batch_size, seq_len, num_marks, resolution, num_marks, 1]

        # Get intensity values at every sampled $ t $.
        expand_intensity = torch.autograd.grad(
            outputs=expand_integral,
            inputs=time_expand,
            grad_outputs=torch.ones_like(expand_integral),
        )[0]  # [..., batch_size, seq_len, num_marks, resolution, num_marks]
        time_expand.requires_grad = False

        expand_integral = expand_integral.squeeze(
            dim=-1
        ).detach()  # [..., batch_size, seq_len, num_marks, resolution, num_marks]
        expand_intensity = expand_intensity.detach()  # [..., batch_size, seq_len, num_marks, resolution, num_marks]

        return expand_integral, expand_intensity, original_time_expand

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

        # Prepare the history embedding.
        time_history = (time_history - mean) / std  # [batch_size, seq_len]

        marks_embeddings = self.marks(marks_history)
        # [batch_size, seq_len, d_intensity]
        time_history = repeat(time_history, "... -> ... ne", ne=self.num_marks)  # [batch_size, seq_len, num_marks]
        time_history_emb = time_history.unsqueeze(dim=-1) * self.nonneg_activation(self.weight_for_t)
        # [batch_size, seq_len, num_marks, d_intensity]
        time_history_emb = (
            time_history_emb[..., 1:, :, :]
            * nn.functional.one_hot(marks_history[..., 1:], num_classes=self.num_marks).unsqueeze(dim=-1)
        ).sum(dim=-2)
        # [batch_size, seq_len-1, d_intensity]
        time_history_emb = torch.cat(
            (
                torch.zeros(time_history_emb.shape[0], 1, time_history_emb.shape[-1], device=self.device),
                time_history_emb,
            ),
            dim=-2,
        )
        # [batch_size, seq_len, d_intensity]
        history = marks_embeddings + time_history_emb  # [batch_size, seq_len, d_intensity]

        hidden_history, (_, _) = self.his_encoder(history)  # [batch_size, seq_len, d_intensity]
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

        probability_distribution = expand_intensity * torch.exp(-expand_integral.sum(dim=-1, keepdim=True))
        # [batch_size, seq_len, integration_sample_rate, num_mark]

        results = evaluate_on_one_batch(
            probability_distribution,
            dim_input=-3,
            mask=mask_next,
            evaluate_func=["spearman_self", "pearson_self", "l1_self"],
            additional_inputs=[
                original_time_expand,
            ],
        )

        data["spearman_matrix"] = move_from_tensor_to_ndarray(results["spearman_self"])
        data["pearson_matrix"] = move_from_tensor_to_ndarray(results["pearson_self"])
        data["L1_matrix"] = move_from_tensor_to_ndarray(results["l1_self"])

        return data, original_time_expand
