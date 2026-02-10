from typing import TYPE_CHECKING, Self

import torch
import torch.nn.functional as F
from einops import rearrange

from src.toolbox.algorithms import approximate_integration, evaluate_on_one_batch
from src.toolbox.misc import (
    argument_check,
    check_tensor,
    compile_func,
    compile_model,
    pack_one_value_to_dict,
)
from src.tpp.tpp_models.model.basic_tpp_model import (
    BasicModel,
    memory_ceiling,
)
from src.tpp.tpp_models.model.utils import (
    BalancedSamplingFromDistributionMixin,
    GetWhichEventFirstMixin,
    NextEventPredictionMarkTimeMixin,
    NextEventPredictionTimeMarkMixin,
    SpearmanL1EvaluationMixin,
    decide_resolution_inf_and_resolution_between_events,
)

from .plot import (
    generate_debug_figure,
    generate_integral_figure,
    generate_intensity_figure,
    generate_probability_figure,
)
from .sample import sample_time
from .submodel import AttNHP

if TYPE_CHECKING:
    import argparse


class AttNHPWrapper(
    BasicModel,
    BalancedSamplingFromDistributionMixin,
    GetWhichEventFirstMixin,
    NextEventPredictionMarkTimeMixin,
    NextEventPredictionTimeMarkMixin,
    SpearmanL1EvaluationMixin,
):
    def __init__(
        self,
        opt,
        training,
        device,
        d_input=64,
        d_hidden=256,
        n_layers=3,
        n_head=3,
        d_qkv=64,
        dropout=0.1,
        epsilon=1e-20,
        sample_rate=32,
        mae_step=8,
        mae_e_step=8,
        resolution=100,
        survival_loss_during_training=True,
    ) -> Self:
        super().__init__()
        self.device = device
        self.training = training
        self.device_type = "cuda" if opt.cuda else "cpu"
        self.model_dtype = opt.dtype
        self.use_compile = opt.compile
        self.compile_backend = opt.compile_backend
        self.num_marks = opt.info_dict["num_marks"]
        self.start_time = opt.info_dict["t_0"]
        self.end_time = opt.info_dict["T"]
        self.resolution = resolution
        self.epsilon = epsilon
        self.survival_loss_during_training = survival_loss_during_training
        self.sample_rate = sample_rate
        self.mae_step = mae_step
        self.mae_e_step = mae_e_step
        self.bisect_early_stop_threshold = 1e-5
        self.max_step = 50

        self.model = AttNHP(
            training=training,
            num_marks=self.num_marks,
            d_input=d_input,
            d_hidden=d_hidden,
            n_layers=n_layers,
            n_head=n_head,
            d_qkv=d_qkv,
            dropout=dropout,
            device=device,
            resolution=resolution,
        )

        self.model = compile_model(self.model, self.use_compile, self.compile_backend)

    def divide_history_and_next(self, input_data):
        """
        What divide_history_and_next should do?
        [a, b, c, d, e, pad, pad, pad]
        [1, 1, 1, 1, 1, 0,   0,   0]
                    |
                    |
                    |
                   \\/
        [a, b, c, d, e, pad, pad], [b, c, d, e, pad, pad, pad]
        [1, 1, 1, 1, 1, 0,   0  ], [1, 1, 1, 1, 0,   0,   0  ]
        """
        input_history, input_next = input_data[:, :-1].clone(), input_data[:, 1:].clone()
        return input_history, input_next

    def remove_dummy_events_from_mask(self: Self, mask: torch.Tensor) -> torch.Tensor:
        """Remove the dummy marks by altering the mask.

        Args:
            self (Self): the SAHP model
            mask (torch.Tensor): the input mask tensor. shape: [batch, seq_len]

        Returns:
            torch.Tensor: The input mask tensor with the dummy marks at the end removed. shape: [batch, seq_len]
        """
        dummy_indices = mask.sum(dim=1, dtype=torch.long) - 1  # [batch_size]
        mask_without_dummy = mask.clone()  # [batch_size, seq_len]
        batch_indices = torch.arange(mask.size(0), device=mask.device)  # [batch_size]
        mask_without_dummy[batch_indices, dummy_indices] = 0  # [batch_size, seq_len]

        return mask_without_dummy

    def forward(self, task_name, *args, **kwargs):
        """
        The entrance of the FullyNN wrapper.

        Args:
        * input_time    type: torch.tensor shape: [batch_size, seq_len + 1]
                        The original time sequence. We should extract the history and target sequence from it
                        by divide_history_and_next().
        * input_marks  type: torch.tensor shape: [batch_size, seq_len + 1]
                        The original mark sequence. We should extract the history and target sequence from it
                        by divide_history_and_next().
        * mask          type: torch.tensor shape: [batch_size, seq_len + 1]
                        We use mask to mask out unneeded outputs.
        * mean          type: float shape: N/A
                        The mean of all $ t_i - t_{i - 1} $ in the entire dataset. Dataloader is responsible to provide
                        this value if needed.
        * std           type: float shape: N/A
                        The mean of all $ t_i - t_{i - 1} $ in the entire dataset. Dataloader is responsible to provide
                        this value if needed.
        * evaluate      type: bool shape: N/A
                        perform a model training step when evaluate == False
                        perform a model evaluate step when evaluate == True

        Outputs:
        Refers to train() and evaluate()'s documentation for detailed information.

        """
        task_mapper = {
            "train": self.train_procedure,
            "evaluate": self.evaluate_procedure,
            "spearman_and_l1": self.get_spearman_and_l1,
            "mae_and_f1": self.get_mae_and_f1,
            "mae_e_and_f1": self.get_mae_e_and_f1,
            "which_mark_occurs_first": self.get_which_event_first,
            "balanced_sampling_from_distribution": self.balanced_sampling_from_distribution,
            # Figure Drawing.
            "intensity": self.figure_intensity,
            "integral": self.figure_integral,
            "probability": self.figure_probability,
            "debug": self.figure_debug,
        }

        return task_mapper[task_name](*args, **kwargs)

    """
    Functions for model training.
    """

    def train_procedure(self, time, marks, mask, mean, std):
        """
        Check if marks data is present.
        Now, we assume that no mark data is available.
        Args:
        1. time: the sequence containing marks' timestamps. shape: [batch_size, seq_len + 1]
        2. marks: the sequence containing information about marks. shape: [batch_size, seq_len + 1]
        3. mask: filter out the padding marks in the mark batches. shape: [batch_size, seq_len + 1]
        """
        time_history, time_next = self.divide_history_and_next(time)  # [batch_size, seq_len] * 2
        marks_history, marks_next = self.divide_history_and_next(marks)  # [batch_size, seq_len] * 2
        mask_history, mask_next = self.divide_history_and_next(mask)  # [batch_size, seq_len] * 2

        integral_all_marks, intensity_all_marks = self.model(
            time_history, time_next, marks_history, mask_history, mask_next
        )
        # 2 * [..., batch_size, seq_len, num_marks]

        mask_next_without_dummy = self.remove_dummy_events_from_mask(mask_next)  # [batch_size, seq_len]
        mark_next_without_dummy = (mask_next_without_dummy * marks_next).long()
        # [batch_size, seq_len]
        the_number_of_marks = mask_next_without_dummy.sum().item()

        # L = \\sum_{i}{\\lambda^_k*(t_i)} + \\int_{t_0}^{t_n}{\\sum_{k}{\\lambda^*_k(\\tau)}d\\tau}
        log_likeli_loss_without_dummy, marker_loss_without_dummy = self.loss_function(
            integral_all_marks=integral_all_marks,
            intensity_all_marks=intensity_all_marks,
            marks_next=mark_next_without_dummy,
            mask_next=mask_next_without_dummy,
        )

        loss_survival = 0
        if self.survival_loss_during_training:
            # survival_loss = \\int_{t_n}^{T}{\\sum_{k}{\\lambda^*_k(\\tau)}d\\tau}
            dummy_mark_index = mask_next.sum(dim=-1) - 1  # [batch_size]
            integral_survival = integral_all_marks.sum(dim=-1).gather(index=dummy_mark_index.unsqueeze(dim=-1), dim=-1)
            # [batch_size, 1]
            loss_survival = integral_survival.sum()

        loss = log_likeli_loss_without_dummy + loss_survival

        return loss, log_likeli_loss_without_dummy, marker_loss_without_dummy, the_number_of_marks

    """
    Functions for model evaluation
    """

    @torch.inference_mode()
    def evaluate_procedure(self, time, marks, mask, mean, std):
        """
        Check if marks data is present.
        Now, we assume that no mark data is available.
        Args:
        1. time: the sequence containing marks' timestamps. shape: [batch_size, seq_len + 1]
        2. marks: the sequence containing information about marks. shape: [batch_size, seq_len + 1]
        3. mask: filter out the padding marks in the mark batches. shape: [batch_size, seq_len + 1]
        """
        time_history, time_next = self.divide_history_and_next(time)  # [batch_size, seq_len] * 2
        marks_history, marks_next = self.divide_history_and_next(marks)  # [batch_size, seq_len] * 2
        mask_history, mask_next = self.divide_history_and_next(mask)  # [batch_size, seq_len] * 2

        mask_next_without_dummy = self.remove_dummy_events_from_mask(mask_next)  # [batch_size, seq_len]
        mark_next_without_dummy = (mask_next_without_dummy * marks_next).long()
        # [batch_size, seq_len]
        the_number_of_marks = mask_next_without_dummy.sum().item()

        pred_time, mark_dist = self.next_event_prediction_time_mark(
            time_history,
            time_next,
            marks_history,
            mask_history,
            mask_next,
            mean=mean,
            std=std,
            sample_rate=self.sample_rate,
            mae_step=self.mae_step,
        )

        mae = torch.abs(pred_time - time_next) * mask_next_without_dummy  # [batch_size, seq_len]
        mae = mae.sum().item() / the_number_of_marks

        pred_mark = mark_dist.argmax(dim=-1)  # [batch_size, seq_len]

        results = evaluate_on_one_batch(
            pred_mark, marks_next, mask_next_without_dummy, ["acc", "macro-f1", "micro-f1"], num_classes=self.num_marks
        )

        acc = results["acc"].mean().item()
        macro_f1 = results["macro-f1"].mean().item()
        micro_f1 = results["micro-f1"].mean().item()

        integral_all_marks_time_next, intensity_all_marks_time_next = self.model(
            time_history, time_next, marks_history, mask_history, mask_next
        )
        # 2 * [batch_size, seq_len, num_marks]

        # NLL loss and mark loss at time_next
        # L = \\sum_{i}{\\lambda^_k*(t_i)} + \\int_{t_0}^{t_n}{\\sum_{k}{\\lambda^*_k(\\tau)}d\\tau}
        log_likeli_loss_time_next_without_dummy, marker_loss_time_next_without_dummy = self.loss_function(
            integral_all_marks=integral_all_marks_time_next,
            intensity_all_marks=intensity_all_marks_time_next,
            marks_next=mark_next_without_dummy,
            mask_next=mask_next_without_dummy,
        )
        # Survival probability: \\int_{t_N}^{T}{\\sum_{k}\\lambda_k^(\\tau)d\\tau}
        dummy_events_index = mask_next.sum(dim=-1) - 1  # [batch_size]
        integral_survival = integral_all_marks_time_next.sum(dim=-1).gather(
            index=dummy_events_index.unsqueeze(dim=-1), dim=-1
        )
        # [batch_size, 1]
        loss_survival = integral_survival.mean()

        return (
            log_likeli_loss_time_next_without_dummy / the_number_of_marks,
            loss_survival,
            marker_loss_time_next_without_dummy / the_number_of_marks,
            mae,
            acc,
            macro_f1,
            micro_f1,
            the_number_of_marks,
        )

    """
    Loss functions
    """

    @compile_func(compile_or_not="use_compile", backend="compile_backend", fullgraph=True)
    def loss_function(self, integral_all_marks, intensity_all_marks, marks_next, mask_next):
        """Log-likelihood of sequence."""
        type_mask = F.one_hot(marks_next, num_classes=self.num_marks)  # [batch_size, seq_len, num_marks]

        # MTPP loss function
        selected_intensity = (intensity_all_marks * type_mask).sum(dim=-1)  # [batch_size, seq_len]
        log_intensity = torch.log(selected_intensity + self.epsilon)  # [batch_size, seq_len]
        nll = -log_intensity + integral_all_marks.sum(dim=-1)  # [batch_size, seq_len]

        mtpp_loss = torch.sum(nll * mask_next)

        # mark loss function. Only for evaluation, do NOT use this loss as a part of the training loss.
        marks_prediction_probability = torch.log(intensity_all_marks + self.epsilon)
        # [batch_size, seq_len, num_marks]
        marks_prediction_probability = F.softmax(marks_prediction_probability, dim=-1)
        # [batch_size, seq_len, num_marks]
        reshaped_marks_prediction_probability = rearrange(marks_prediction_probability, "b s ne -> b ne s")
        # [batch_size, num_marks, seq_len]
        marks_loss = F.cross_entropy(input=reshaped_marks_prediction_probability, target=marks_next, reduction="none")
        # [batch_size, seq_len]
        marks_loss = (marks_loss * mask_next).sum()

        return mtpp_loss, marks_loss

    sample_time = sample_time

    @torch.inference_mode()
    def next_event_prediction_time_mark(
        self: Self,
        time_history: torch.Tensor,
        time_next: torch.Tensor,
        marks_history: torch.Tensor,
        mask_history: torch.Tensor,
        mask_next: torch.Tensor,
        mean: float,
        std: float,
        sample_rate: int,
        mae_step: int,
        get_time_sample: bool = False,
        evaluation: bool = True,
        resolution: int = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the next mark prediction from the SAHP by MAE and F1.
        This function first predict the time of the next mark then the mark of the next mark given the TRUE time.
        This can ensure the mark accuracy is only influenced by the model, not by the predicted time.

        Args:
            self (Self): the SAHP model.
            time_history (torch.Tensor): the time of historical marks
            time_next (torch.Tensor): the time of the next true marks
            marks_history (torch.Tensor): the mark of historical marks
            marks_next (torch.Tensor): the mark of the next true marks
            mask_history (torch.Tensor): the mask of historical sequences, 1 meaning a true mark, and 0 meaning a fake mark.
            mask_next (torch.Tensor): the mask of the next true marks
            mean (float): the mean of all time intervals
            std (float): the standard variance of all time intervals
            sample_rate (int): how many samples are needed for one time prediction
            mae_step (int): how many samples for one mark are generated in one shot
            evaluation (bool): If true, we are in the evaluation mode, the mark distribution is at the time_next.
                               If false, we are in the prediction mode, the mark distribution is at the pred_time

        Returns:
            tuple[torch.Tensor, torch.Tensor]: predicted time and mark distribution
        """
        inf_val = mean + 10 * std
        pred_time = self.sample_time(
            sampling_approach="its",
            task="tm",
            time_history=time_history,
            marks_history=marks_history,
            mask_history=mask_history,
            mask_next=mask_next,
            resolution=self.resolution if resolution is None else resolution,
            number_of_total_samples=sample_rate,
            step=mae_step,
            inf_val=inf_val,
            mean=mean,
            std=std,
        )  # [sample_rate, batch_size, seq_len]

        if not get_time_sample:
            pred_time = pred_time.mean(dim=0)  # [batch_size, seq_len]

        if evaluation:
            _, intensity_all_marks = self.model(time_history, time_next, marks_history, mask_history, mask_next)
            # [batch_size, seq_len, num_marks]
        else:
            _, intensity_all_marks = self.model(time_history, pred_time, marks_history, mask_history, mask_next)
            # [batch_size, seq_len, num_marks]

        mark_distribution = intensity_all_marks / intensity_all_marks.sum(dim=-1, keepdim=True)
        # [batch_size, seq_len, num_marks]
        return pred_time, mark_distribution

    @torch.inference_mode()
    def next_event_prediction_mark_time(
        self: Self,
        time_history: torch.Tensor,
        marks_history: torch.Tensor,
        marks_next: torch.Tensor,
        mask_history: torch.Tensor,
        mean: float,
        std: float,
        sample_rate: int,
        mae_e_step: int,
        get_time_sample: bool = False,
        evaluation: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the next mark prediction from the SAHP by MAE and F1.
        This function first predict the time of the next mark then the mark of the next mark given the TRUE time.
        This can ensure the mark accuracy is only influenced by the model, not by the predicted time.

        Args:
            self (Self): the SAHP model.
            time_history (torch.Tensor): the time of historical marks
            time_next (torch.Tensor): the time of the next true marks
            marks_history (torch.Tensor): the mark of historical marks
            marks_next (torch.Tensor): the mark of the next true marks
            mask_history (torch.Tensor): the mask of historical sequences, 1 meaning a true mark, and 0 meaning a fake mark.
            mask_next (torch.Tensor): the mask of the next true marks
            mean (float): the mean of all time intervals
            std (float): the standard variance of all time intervals
            sample_rate (int): how many samples are needed for one time prediction
            mae_e_step (int): how many samples for one mark are generated in one shot
            evaluation (bool): If true, we are in the evaluation mode, the mark distribution is at the time_next.
                               If false, we are in the prediction mode, the mark distribution is at the pred_time

        Returns:
            tuple[torch.Tensor, torch.Tensor]: predicted time and mark distribution
        """
        inf_val = mean + 10 * std
        batch_size, seq_len = time_history.shape
        resolution_inf, resolution_between_marks = decide_resolution_inf_and_resolution_between_events(
            batch_size, seq_len, memory_ceiling, self.num_marks
        )
        mark_distribution = self.get_pm_next_event(time_history, marks_history, mask_history, inf_val, resolution_inf)
        # [batch_size, seq_len, num_marks]

        tau_sampled_all_mark = self.sample_time(
            sampling_approach="its",
            task="mt",
            marks_history=marks_history,
            time_history=time_history,
            mask_history=mask_history,
            p_m=mark_distribution,
            resolution=resolution_between_marks,
            number_of_total_samples=sample_rate,
            step=mae_e_step,
            inf_val=inf_val,
            mean=mean,
            std=std,
        )  # [sample_rate, batch_size, seq_len, num_marks]

        if not get_time_sample:
            tau_sampled_all_mark = tau_sampled_all_mark.mean(dim=0)  # [batch_size, seq_len, num_marks]
        else:
            evaluation = False

        if evaluation:
            marks_next_mask = torch.nn.functional.one_hot(marks_next, num_classes=self.num_marks)
            # [batch_size, seq_len, num_marks]
            pred_time = (tau_sampled_all_mark * marks_next_mask).sum(dim=-1)  # [batch_size, seq_len]
        else:
            pred_time = tau_sampled_all_mark  # [batch_size, seq_len, num_marks]

        return pred_time, mark_distribution

    @torch.inference_mode()
    def get_pm_next_event(
        self: Self,
        time_history: torch.Tensor,
        marks_history: torch.Tensor,
        mask_history: torch.Tensor,
        inf_val: int,
        resolution_inf: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the next mark prediction from the SAHP by MAE and F1.
        This function first predict the time of the next mark then the mark of the next mark given the TRUE time.
        This can ensure the mark accuracy is only influenced by the model, not by the predicted time.

        Args:
            self (Self): the SAHP model.
            time_history (torch.Tensor): the time of historical marks.
            marks_history (torch.Tensor): the mark of historical marks.
            mask_history (torch.Tensor): the mask of historical sequences, 1 meaning a true mark, and 0 meaning a fake mark.
            inf_val (float): the number treated as positive infinity.
            resolution_inf (int): how many samples are needed for mark distribution estimation.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: predicted mark distribution
        """
        time_next_inf = torch.ones_like(time_history, device=self.device) * inf_val
        # [batch_size, seq_len]
        (
            expanded_integral_all_marks_to_inf,
            expanded_intensity_all_marks_to_inf,
            timestamp,
        ) = self.model.integral_intensity_time_next_2d(
            time_history, time_next_inf, marks_history, mask_history, resolution_inf
        )
        # 2 * [batch_size, seq_len, resolution, num_marks]
        expanded_probability_inf = (
            torch.exp(-expanded_integral_all_marks_to_inf.sum(dim=-1, keepdim=True))
            * expanded_intensity_all_marks_to_inf
        )
        # [batch_size, seq_len, resolution, num_marks]
        return approximate_integration(expanded_probability_inf, timestamp, dim=-2, only_integral=True)
        # [batch_size, seq_len, num_marks]

    @torch.inference_mode()
    def probability_time_next_2d(self, time_history, time_next, marks_history, mask_history, resolution, mean, std):
        expand_integral, expand_intensity, timestamp = self.model.integral_intensity_time_next_2d(
            time_history, time_next, marks_history, mask_history, resolution
        )
        return expand_intensity * torch.exp(-expand_integral.sum(dim=-1, keepdim=True)), timestamp

    def extract_plot_data(self, minibatch):
        """
        This function extracts input_time, input_marks, input_intensity, mask, mean, and std from the minibatch.

        Args:
        * minibatch  type: list shape: [[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]
                     data structure: [[input_time, input_marks, score, mask], (mean, std)]

        Outputs:
        * input_time    type: torch.tensor shape: [batch_size, seq_len + 1]
                        Raw mark timestamp sequence.
        * input_marks  type: torch.tensor shape: [batch_size, seq_len + 1]
                        Raw mark marks sequence.
        * mask          type: torch.tensor shape: [batch_size, seq_len + 1]
                        Raw mask sequence.
        * mean          type: int shape: N/A
                        The mean of all $ t_i - t_{i - 1} $ in the entire dataset. Dataloader is responsible to provide
                        this value if needed.
        * std           type: int shape: N/A
                        The mean of all $ t_i - t_{i - 1} $ in the entire dataset. Dataloader is responsible to provide
                        this value if needed.
        """
        input_time, input_marks, _, mask, input_intensity = minibatch[0]
        mean, std = minibatch[1]

        return input_time, input_marks, input_intensity, mask, mean, std

    @torch.inference_mode()
    def figure_intensity(self: Self, input_data: list, opt: argparse.Namespace) -> None:
        """Function prober, used by evaluator to draw plots of the intensity function.

        You should declare the following arguments in your config file:
        1. ```int``` resolution: The number of interpolated points in a time interval between two adjoint marks for integration estimation.
                                 The number of interpolated points counts the start and end point of the interval.
        Args:
            self (Self): the model
            input_data (list): the minibatch from the dataloader.
            opt (argparse.Namespace): the input arguments.
        """
        argument_check(opt, **{"resolution": int})

        input_time, input_marks, input_intensity, mask, mean, std = self.extract_plot_data(input_data)

        time_history, time_next = self.divide_history_and_next(input_time)  # [batch_size, seq_len]
        marks_history, marks_next = self.divide_history_and_next(input_marks)
        # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)  # [batch_size, seq_len]

        expand_integral, expand_intensity, timestamp = self.model.integral_intensity_time_next_2d(
            time_history, time_next, marks_history, mask_history, opt.resolution
        )
        # 3 * [batch_size, seq_len, resolution, num_marks]
        check_tensor(expand_integral)
        check_tensor(expand_intensity)
        if expand_intensity.shape != expand_integral.shape:
            raise ValueError("Why expand_intensity and expand_integral have different shapes?")

        data = {
            "time_next": time_next,
            "marks_next": marks_next,
            "mask_next": mask_next,
            "expand_intensity": expand_intensity,
            "input_intensity": input_intensity,
            "timestamp": timestamp,
        }
        generate_intensity_figure(data, opt)

    @torch.inference_mode()
    def figure_integral(self: Self, input_data: list, opt: argparse.Namespace) -> None:
        """Function prober, used by evaluator to draw plots of the integral of the intensity function.

          You should declare the following arguments in your config file:
          1. ```int``` resolution: The number of interpolated points in a time interval between two adjoint marks for integration estimation.
                                   The number of interpolated points counts the start and end point of the interval.

        Args:
            self (Self): the model
            input_data (list): the minibatch from the dataloader.
            opt (argparse.Namespace): the input arguments.
        """
        argument_check(opt, **{"resolution": int})

        input_time, input_marks, input_intensity, mask, mean, std = self.extract_plot_data(input_data)

        time_history, time_next = self.divide_history_and_next(input_time)  # [batch_size, seq_len]
        marks_history, marks_next = self.divide_history_and_next(input_marks)
        # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)  # [batch_size, seq_len]

        expand_integral, expand_intensity, timestamp = self.model.integral_intensity_time_next_2d(
            time_history, time_next, marks_history, mask_history, opt.resolution
        )
        # 3 * [batch_size, seq_len, resolution, num_marks]
        check_tensor(expand_integral)
        check_tensor(expand_intensity)
        if expand_intensity.shape != expand_integral.shape:
            raise ValueError("Why expand_intensity and expand_integral have different shapes?")

        data = {
            "time_next": time_next,
            "marks_next": marks_next,
            "mask_next": mask_next,
            "expand_integral": expand_integral,
            "input_intensity": input_intensity,
            "timestamp": timestamp,
        }
        generate_integral_figure(data, opt)

    @torch.inference_mode()
    def figure_probability(self: Self, input_data: list, opt: argparse.Namespace) -> None:
        """Function prober, used by evaluator to draw plots of the probability distribution.

          You should declare the following arguments in your config file:
          1. ```int``` resolution: The number of interpolated points in a time interval between two adjoint marks for integration estimation.
                                   The number of interpolated points counts the start and end point of the interval.


        Args:
            self (Self): the model
            input_data (list): the minibatch from the dataloader.
            opt (argparse.Namespace): the input arguments.
        """
        argument_check(opt, **{"resolution": int})

        input_time, input_marks, input_intensity, mask, mean, std = self.extract_plot_data(input_data)

        time_history, time_next = self.divide_history_and_next(input_time)  # [batch_size, seq_len]
        marks_history, marks_next = self.divide_history_and_next(input_marks)
        # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)  # [batch_size, seq_len]

        expand_integral, expand_intensity, timestamp = self.model.integral_intensity_time_next_2d(
            time_history, time_next, marks_history, mask_history, opt.resolution
        )
        # 3 * [batch_size, seq_len, resolution, num_marks]
        check_tensor(expand_integral)
        check_tensor(expand_intensity)
        if expand_intensity.shape != expand_integral.shape:
            raise ValueError("Why expand_intensity and expand_integral have different shapes?")
        expand_probability = expand_intensity * torch.exp(-expand_integral.sum(dim=-1, keepdim=True))
        # [batch_size, seq_len, resolution, num_marks]
        data = {
            "time_next": time_next,
            "marks_next": marks_next,
            "mask_next": mask_next,
            "expand_probability": expand_probability,
            "input_intensity": input_intensity,
            "timestamp": timestamp,
        }

        generate_probability_figure(data, opt)

    @torch.inference_mode()
    def figure_debug(self: Self, input_data: list, opt: argparse.Namespace) -> None:
        """Function prober, used by evaluator to draw plots for deeper insight of intensity functions and other metrics.

          You should declare the following arguments in your config file:
          1. ```int``` resolution: The number of interpolated points in a time interval between two adjoint marks for integration estimation.
                                   The number of interpolated points counts the start and end point of the interval.
          2. ```int``` sample_rate: The number of interpolated points in a time interval between two adjoint marks for integration estimation.
                                    The number of interpolated points counts the start and end point of the interval.
          3. ```int``` mae_step: This parameter controls how many samples are generated in one shot when sampling from p(t).
          4. ```int``` mae_e_step: This parameter controls how many samples are generated in one shot when sampling from all p(t|m)s at the same time.


        Args:
            self (Self): the model
            input_data (list): the minibatch from the dataloader.
            opt (argparse.Namespace): the input arguments.
        """
        argument_check(
            opt,
            **{
                "resolution": int,
                "sample_rate": int,
                "mae_step": int,
                "mae_e_step": int,
            },
        )

        input_time, input_marks, input_intensity, mask, mean, std = self.extract_plot_data(input_data)

        time_history, time_next = self.divide_history_and_next(input_time)  # [batch_size, seq_len]
        marks_history, marks_next = self.divide_history_and_next(input_marks)
        # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)  # [batch_size, seq_len]

        data, timestamp = self.model.model_probe_function(
            time_history,
            time_next,
            marks_history,
            mask_history,
            mask_next,
            opt.resolution,
        )

        pred_time, _ = self.next_event_prediction_time_mark(
            time_history,
            time_next,
            marks_history,
            mask_history,
            mean=mean,
            std=std,
            sample_rate=self.sample_rate,
            mae_step=self.mae_step,
        )
        # [batch_size, seq_len] + [batch_size, seq_len]
        mae_tm = torch.abs(pred_time - time_next) * mask_next  # [batch_size, seq_len]

        pred_time_all_marks, mark_dist = self.next_event_prediction_mark_time(
            time_history,
            marks_history,
            marks_next,
            mask_history,
            mean=mean,
            std=std,
            sample_rate=self.sample_rate,
            mae_e_step=opt.mae_e_step,
            evaluation=False,
        )
        # [batch_size, seq_len, num_marks] + [batch_size, seq_len, num_marks]
        marks_next_mask = torch.nn.functional.one_hot(marks_next, num_classes=self.num_marks)
        # [batch_size, seq_len, num_marks]
        pred_time = (pred_time_all_marks * marks_next_mask).sum(dim=-1)  # [batch_size, seq_len, num_marks]
        maes_ptm = torch.abs(pred_time - time_next) * mask_next  # [batch_size, seq_len]
        top_k = evaluate_on_one_batch(
            mark_dist, marks_next, mask_next, "top_k", dim_input=-2, num_classes=self.num_marks
        )
        # [batch_size, num_marks]
        probability_sum = mark_dist.sum(dim=-1)  # [batch_size, seq_len]

        # Append additional info into the data dict.
        data.update(
            {
                "marks_next": marks_next,
                "time_next": time_next,
                "mask_next": mask_next,
                "mae_pt": mae_tm,
                "maes_ptm": maes_ptm,
                "top_k": top_k,
                "tau_pred_all_mark": pred_time_all_marks,
                "probability_sum": probability_sum,
                "timestamp": timestamp,
            }
        )

        generate_debug_figure(data, opt)

    def train_step(self, minibatch, scaler):
        """Epoch operation in training phase"""
        self.train()

        time, marks, score, mask = minibatch[0]  # 3 * [batch_size, seq_len + 1, 1] & [batch_size, seq_len, 1]
        mean, std = minibatch[1]

        with torch.autocast(device_type=self.device_type, dtype=self.model_dtype):
            loss, time_loss_without_dummy, marks_loss, the_number_of_marks = self.forward(
                "train", time, marks, mask, mean, std
            )

        scaler.scale(loss).backward()

        time_loss_without_dummy = time_loss_without_dummy.item() / the_number_of_marks
        marks_loss = marks_loss.item() / the_number_of_marks
        fact = score.sum().item() / the_number_of_marks

        return time_loss_without_dummy, fact, marks_loss

    def evaluation_step(self, minibatch):
        """Epoch operation in evaluation phase"""
        self.eval()

        time, marks, score, mask = minibatch[0]  # 3 * [batch_size, seq_len + 1, 1] & [batch_size, seq_len, 1]
        mean, std = minibatch[1]

        with torch.autocast(device_type=self.device_type, dtype=self.model_dtype):
            (
                time_loss,
                loss_survival,
                marks_loss,
                mae,
                acc,
                macro_f1,
                micro_f1,
                the_number_of_marks,
            ) = self.forward("evaluate", time, marks, mask, mean, std)

        time_loss = time_loss.item() / the_number_of_marks
        loss_survival = loss_survival.item()
        marks_loss = marks_loss.item() / the_number_of_marks
        fact = score.sum().item() / the_number_of_marks

        return time_loss, loss_survival, fact, marks_loss, mae, acc, macro_f1, micro_f1

    def postprocess(self, input_data, procedure):
        def train_postprocess(input_data):
            """
            Training process
            [absolute loss, relative loss, marks loss]
            """
            return [input_data[0], input_data[0] - input_data[1], input_data[2]]

        def test_postprocess(input_data):
            """
            Evaluation process
            [absolute loss, relative loss, marks loss, mae value]
            """
            return [
                input_data[0],
                input_data[1],
                input_data[0] - input_data[2],
                input_data[3],
                input_data[4],
                input_data[5],
                input_data[6],
                input_data[7],
            ]

        return train_postprocess(input_data) if procedure == "training" else test_postprocess(input_data)

    def log_print_format(self, input_data, procedure):
        def train_log_print_format(input_data):
            format_dict = {}
            format_dict["absolute_loss"] = pack_one_value_to_dict(input_data[0])
            format_dict["relative_loss"] = pack_one_value_to_dict(input_data[1])
            format_dict["marks_loss"] = pack_one_value_to_dict(input_data[2])
            return format_dict

        def test_log_print_format(input_data):
            format_dict = {}
            format_dict["absolute_NLL_loss"] = pack_one_value_to_dict(input_data[0])
            format_dict["avg_survival_loss"] = pack_one_value_to_dict(input_data[1])
            format_dict["relative_NLL_loss"] = pack_one_value_to_dict(input_data[2])
            format_dict["marks_loss"] = pack_one_value_to_dict(input_data[3])
            format_dict["mae"] = pack_one_value_to_dict(input_data[4], "2.8f")
            format_dict["acc"] = pack_one_value_to_dict(input_data[5], "2.8f")
            format_dict["macro-f1"] = pack_one_value_to_dict(input_data[6], "2.8f")
            format_dict["micro-f1"] = pack_one_value_to_dict(input_data[7], "2.8f")
            return format_dict

        return train_log_print_format(input_data) if procedure == "training" else test_log_print_format(input_data)

    # The maximum length of the format_dict in different procedures.
    format_dict_length = 8

    def choose_metric(self, evaluation_report_format_dict, test_report_format_dict):
        """
        [relative loss on evaluation dataset, relative loss on test dataset, mark loss on test dataset]
        """
        return [evaluation_report_format_dict["absolute_NLL_loss"], test_report_format_dict["absolute_NLL_loss"]], [
            "evaluation_absolute_loss",
            "test_absolute_loss",
        ]

    metric_number = 2  # metric number is the length of the output of choose_metric
