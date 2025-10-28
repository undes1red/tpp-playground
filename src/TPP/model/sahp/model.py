import argparse
from typing import Any, Self

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from sklearn.metrics import roc_auc_score

from src.toolbox.algorithms import approximate_integration
from src.toolbox.metrics import evaluate_func, evaluate_on_one_batch
from src.toolbox.misc import (
    argument_check,
    break_batched_inputs_into_seqs,
    check_tensor,
    move_from_tensor_to_ndarray,
    pack_one_value_to_dict,
)
from src.TPP.model.basic_tpp_model import BasicModel, memory_ceiling
from src.TPP.model.sahp.plot import (
    generate_debug_figure,
    generate_integral_figure,
    generate_intensity_figure,
    generate_probability_figure,
)
from src.TPP.model.sahp.sample import sample_time
from src.TPP.model.sahp.submodel import SAHP
from src.TPP.model.utils import (
    decide_resolution_inf_and_resolution_between_events,
    get_f1_and_top_k_acc_in_mae_e,
    pick_log_probability,
)
from src.TPP.resources import expand_true_probability


class SAHPWrapper(BasicModel):
    """
    Self-attentive Hawkes Process (SAHP), proposed by Zhang et al. @ ICML 2020.
    """

    def __init__(
        self: Self,
        opt: argparse.Namespace,
        device: torch.device,
        d_input: int = 64,
        d_rnn: int = 64,
        d_hidden: int = 256,
        n_layers: int = 3,
        n_head: int = 3,
        d_qk: int = 64,
        d_v: int = 64,
        dropout: int = 0.1,
        epsilon: float = 1e-20,
        sample_rate: int = 32,
        mae_step: int = 4,
        mae_e_step: int = 4,
        integration_sample_rate: int = 100,
        survival_loss_during_training: bool = True,
    ):
        """Create a SAHP model.

        Args:
            self (Self): the SAHP model.
            opt (argparse.Namespace): all input arguments.
            device (torch.device): where we run this model.
            d_input (int, optional): the dimension of the Transformer input tensor. Defaults to 64.
            d_rnn (int, optional): the dimension of the RNN output tensor. Defaults to 64.
            n_layers (int, optional): how many Transformer layers the history encoder has. Defaults to 3.
            n_head (int, optional): the number of heads one Transformer layer has. Defaults to 3.
            d_qk (int, optional): the dimension of Q and K. Defaults to 64.
            d_v (int, optional): the dimension of V. Defaults to 64.
            epsilon (float, optional): Shift the calculated intensity function and probability distribution so ```torch.log()``` won't fail. Defaults to 1e-20.
            sample_rate (int, optional): the number of time samples for one time prediction. Defaults to 32.
            mae_step (int, optional): the number of samples generated in one shot when sampling from all p(t). Defaults to 4.
            mae_e_step (int, optional): the number of samples generated in one shot when sampling from all p(t|m)s. Defaults to 4.
            survival_loss_during_training (bool, optional): When true, the training loss includes the integral between the last observed event to the end time T. Defaults to True.
        """
        super().__init__()
        self.device = device
        self.compile_or_not = opt.compile
        self.num_events = opt.info_dict["num_events"]
        self.start_time = opt.info_dict["t_0"]
        self.end_time = opt.info_dict["T"]
        self.integration_sample_rate = integration_sample_rate
        self.epsilon = epsilon
        self.survival_loss_during_training = survival_loss_during_training
        self.sample_rate = sample_rate
        self.mae_step = mae_step
        self.mae_e_step = mae_e_step
        self.bisect_early_stop_threshold = 1e-4
        self.max_step = 50

        self.model = SAHP(
            num_events=self.num_events,
            d_input=d_input,
            d_rnn=d_rnn,
            d_hidden=d_hidden,
            n_layers=n_layers,
            n_head=n_head,
            d_qk=d_qk,
            d_v=d_v,
            dropout=dropout,
            device=device,
            integration_sample_rate=integration_sample_rate,
        )

    def divide_history_and_next(self: Self, input_data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract the history and prediction sequences from the input sequence.

        Args:
            self (Self): the SAHP model
            input (torch.Tensor): the input tensor. shape: [batch, seq_len + 1]

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The history sequence and the target event extracted from the original input. shape: [batch, seq_len]
        """
        input_history, input_next = input_data[:, :-1].clone(), input_data[:, 1:].clone()
        return input_history, input_next

    def remove_dummy_event_from_mask(self: Self, mask: torch.Tensor) -> torch.Tensor:
        """Remove the dummy event by altering the mask.

        Args:
            self (Self): the SAHP model
            mask (torch.Tensor): the input mask tensor. shape: [batch, seq_len]

        Returns:
            torch.Tensor: The input mask tensor with the dummy event at the end removed. shape: [batch, seq_len]
        """
        dummy_indices = mask.sum(dim=1, dtype=torch.long) - 1  # [batch_size]
        mask_without_dummy = mask.clone()  # [batch_size, seq_len]
        batch_indices = torch.arange(mask.size(0), device=mask.device)  # [batch_size]
        mask_without_dummy[batch_indices, dummy_indices] = 0  # [batch_size, seq_len]

        return mask_without_dummy

    def forward(self: Self, task_name: str, *args, **kwargs) -> Any:
        """The entrance of the SAHP model.

        Args:
            self (Self): the SAHP model
            task_name (str): the task name

        Returns:
            Any: the output
        """
        task_mapper = {
            "train": self.train_procedure,
            "evaluate": self.evaluate_procedure,
            "spearman_and_l1": self.get_spearman_and_l1,
            "mae_and_f1": self.get_mae_and_f1,
            "mae_e_and_f1": self.get_mae_e_and_f1,
            "which_event_occurs_first": self.get_which_event_first,
            "samples_from_et": self.samples_from_et,
            # Functions for the EHD task.
            "ehd_perplexity": self.ehd_perplexity,
            "ehd_event_emb": self.get_event_embedding,
            # Figure Drawing.
            "intensity": self.figure_intensity,
            "integral": self.figure_integral,
            "probability": self.figure_probability,
            "debug": self.figure_debug,
            # For CPPOD, should be used with the od_generic dataloader.
            "cppod_evaluation": self.cppod_evaluation,
            "cppod_commission_evaluation": self.cppod_commission_evaluation,
        }

        return task_mapper[task_name](*args, **kwargs)

    def train_procedure(
        self: Self,
        time: torch.Tensor,
        events: torch.Tensor,
        mask: torch.Tensor,
        mean: float,
        std: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """SAHP's forwardpropagation function for training.

        Args:
            self (Self): the SAHP model
            time (torch.Tensor): input time sequence
            events (torch.Tensor): input event sequence
            mask (torch.Tensor): input mask sequence. 1 means keeping this event, and 0 means this event should be masked.
            mean (float): the mean of all time intervals.
            std (float): the standard variance of all time intervals.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]: the training loss, the NLL at true events, the cross entropy loss of the mark, the number of true events.
        """
        time_history, time_next = self.divide_history_and_next(time)  # [batch_size, seq_len] * 2
        events_history, events_next = self.divide_history_and_next(events)  # [batch_size, seq_len] * 2
        mask_history, mask_next = self.divide_history_and_next(mask)  # [batch_size, seq_len] * 2

        integral_all_events, intensity_all_events = self.model(time_history, time_next, events_history, mask_history)
        # 2 * [batch_size, seq_len, num_events]

        mask_next_without_dummy = self.remove_dummy_event_from_mask(mask_next)  # [batch_size, seq_len]
        event_next_without_dummy = (mask_next_without_dummy * events_next).long()
        # [batch_size, seq_len]
        the_number_of_events = mask_next_without_dummy.sum().item()

        # L = \\sum_{i}{\\lambda^_k*(t_i)} + \\int_{t_0}^{t_n}{\\sum_{k}{\\lambda^*_k(\\tau)}d\\tau}
        log_likeli_loss_without_dummy, marker_loss_without_dummy = self.loss_function(
            integral_all_events=integral_all_events,
            intensity_all_events=intensity_all_events,
            events_next=event_next_without_dummy,
            mask_next=mask_next_without_dummy,
        )

        loss_survival = 0
        if self.survival_loss_during_training:
            # survival_loss = \\int_{t_n}^{T}{\\sum_{k}{\\lambda^*_k(\\tau)}d\\tau}
            dummy_event_index = mask_next.sum(dim=-1) - 1  # [batch_size]
            integral_survival = integral_all_events.sum(dim=-1).gather(
                index=dummy_event_index.unsqueeze(dim=-1), dim=-1
            )
            # [batch_size, 1]
            loss_survival = integral_survival.sum()

        loss = log_likeli_loss_without_dummy + loss_survival

        return (
            loss / the_number_of_events,
            log_likeli_loss_without_dummy / the_number_of_events,
            marker_loss_without_dummy / the_number_of_events,
            the_number_of_events,
        )

    @torch.inference_mode()
    def evaluate_procedure(
        self: Self,
        time: torch.Tensor,
        events: torch.Tensor,
        mask: torch.Tensor,
        mean: float,
        std: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float, int]:
        """SAHP's forwardpropagation function for evaluation.

        Args:
            self (Self): the SAHP model
            time (torch.Tensor): input time sequence
            events (torch.Tensor): input event sequence
            mask (torch.Tensor): input mask sequence. 1 means keeping this event, and 0 means this event should be masked.
            mean (float): the mean of all time intervals.
            std (float): the standard variance of all time intervals.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float, int]: the NLL at true events, the sum of the survival loss, the cross entropy loss of the mark, MAE to evaluate time predictions, F1 to evaluate mark predictions, the number of true events.
        """
        time_history, time_next = self.divide_history_and_next(time)  # [batch_size, seq_len] * 2
        events_history, events_next = self.divide_history_and_next(events)  # [batch_size, seq_len] * 2
        mask_history, mask_next = self.divide_history_and_next(mask)  # [batch_size, seq_len] * 2

        mask_next_without_dummy = self.remove_dummy_event_from_mask(mask_next)  # [batch_size, seq_len]
        event_next_without_dummy = (mask_next_without_dummy * events_next).long()
        # [batch_size, seq_len]
        the_number_of_events = mask_next_without_dummy.sum().item()

        integral_all_events_time_next, intensity_all_events_time_next = self.model(
            time_history, time_next, events_history, mask_history
        )
        # 2 * [batch_size, seq_len, num_events]

        pred_time, mark_dist = self.next_event_prediction_time_mark(
            time_history,
            time_next,
            events_history,
            mask_history,
            mean=mean,
            std=std,
            sample_rate=self.sample_rate,
            mae_step=self.mae_step,
        )

        mae = torch.abs(pred_time - time_next) * mask_next  # [batch_size, seq_len]
        mae = mae.sum().item() / the_number_of_events

        pred_mark = mark_dist.argmax(dim=-1)  # [batch_size, seq_len]
        results = evaluate_on_one_batch(pred_mark, events_next, mask_next, ["acc", "macro-f1", "micro-f1"])
        acc = results["acc"].mean()
        macro_f1 = results["macro-f1"].mean()
        micro_f1 = results["micro-f1"].mean()

        # NLL loss and event loss at time_next
        # L = \\sum_{i}{\\lambda^_k*(t_i)} + \\int_{t_0}^{t_n}{\\sum_{k}{\\lambda^*_k(\\tau)}d\\tau}
        log_likeli_loss_time_next_without_dummy, marker_loss_time_next_without_dummy = self.loss_function(
            integral_all_events=integral_all_events_time_next,
            intensity_all_events=intensity_all_events_time_next,
            events_next=event_next_without_dummy,
            mask_next=mask_next_without_dummy,
        )
        # Survival probability: \\int_{t_N}^{T}{\\sum_{k}\\lambda_k^(\\tau)d\\tau}
        dummy_event_index = mask_next.sum(dim=-1) - 1  # [batch_size]
        integral_survival = integral_all_events_time_next.sum(dim=-1).gather(
            index=dummy_event_index.unsqueeze(dim=-1), dim=-1
        )
        # [batch_size, 1]
        loss_survival = integral_survival.mean()

        return (
            log_likeli_loss_time_next_without_dummy / the_number_of_events,
            loss_survival,
            marker_loss_time_next_without_dummy / the_number_of_events,
            mae,
            acc,
            macro_f1,
            micro_f1,
            the_number_of_events,
        )

    def loss_function(
        self: Self,
        integral_all_events: torch.Tensor,
        intensity_all_events: torch.Tensor,
        events_next: torch.Tensor,
        mask_next: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """This function computes the NLL loss and mark loss at all true event in events_next.

        Args:
            self (Self): the SAHP model
            integral_all_events (torch.Tensor): intensity integral from t_{i - 1} to t_{i} (t_0 = 0).
            intensity_all_events (torch.Tensor): intensity values at t_i.
            events_next (torch.Tensor): the mark of the events that we need to predict.
            mask_next (torch.Tensor): mask out unneeded loss values.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: the NLL at true events, the cross entropy loss of the mark
        """
        type_mask = F.one_hot(events_next, num_classes=self.num_events)  # [batch_size, seq_len, num_events]

        # MTPP loss function
        selected_intensity = (intensity_all_events * type_mask).sum(dim=-1)  # [batch_size, seq_len]
        log_intensity = torch.log(selected_intensity + self.epsilon)  # [batch_size, seq_len]
        nll = -log_intensity + integral_all_events.sum(dim=-1)  # [batch_size, seq_len]

        mtpp_loss = torch.sum(nll * mask_next)

        # Event loss function. Only for evaluation, do NOT use this loss as a part of the training loss.
        events_prediction_probability = torch.log(intensity_all_events + self.epsilon)
        # [batch_size, seq_len, num_events]
        events_prediction_probability = F.softmax(events_prediction_probability, dim=-1)
        # [batch_size, seq_len, num_events]
        reshaped_events_prediction_probability = rearrange(events_prediction_probability, "b s ne -> b ne s")
        # [batch_size, num_events, seq_len]
        events_loss = F.cross_entropy(
            input=reshaped_events_prediction_probability,
            target=events_next,
            reduction="none",
        )
        # [batch_size, seq_len]
        events_loss = (events_loss * mask_next).sum()

        return mtpp_loss, events_loss

    sample_time = sample_time

    @torch.inference_mode()
    def next_event_prediction_time_mark(
        self: Self,
        time_history: torch.Tensor,
        time_next: torch.Tensor,
        events_history: torch.Tensor,
        mask_history: torch.Tensor,
        mean: float,
        std: float,
        sample_rate: int,
        mae_step: int,
        evaluation: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the next event prediction from the SAHP by MAE and F1.
        This function first predict the time of the next event then the mark of the next event given the TRUE time.
        This can ensure the mark accuracy is only influenced by the model, not by the predicted time.

        Args:
            self (Self): the SAHP model.
            time_history (torch.Tensor): the time of historical events
            time_next (torch.Tensor): the time of the next true events
            events_history (torch.Tensor): the mark of historical events
            events_next (torch.Tensor): the mark of the next true events
            mask_history (torch.Tensor): the mask of historical sequences, 1 meaning a true event, and 0 meaning a fake event.
            mask_next (torch.Tensor): the mask of the next true events
            mean (float): the mean of all time intervals
            std (float): the standard variance of all time intervals
            sample_rate (int): how many samples are needed for one time prediction
            mae_step (int): how many samples for one event are generated in one shot
            evaluation (bool): If true, we are in the evaluation mode, the mark distribution is at the time_next.
                               If false, we are in the prediction mode, the mark distribution is at the pred_time

        Returns:
            tuple[torch.Tensor, torch.Tensor]: predicted time and mark distribution
        """
        pred_time = self.sample_time(
            sampling_approach="its",
            task="tm",
            time_history=time_history,
            events_history=events_history,
            mask_history=mask_history,
            number_of_total_samples=sample_rate,
            step=mae_step,
            mean=mean,
            std=std,
        )  # [sample_rate, batch_size, seq_len]
        pred_time = pred_time.mean(dim=0)  # [batch_size, seq_len]

        if evaluation:
            _, intensity_all_events = self.model(time_history, time_next, events_history, mask_history)
        # [batch_size, seq_len, num_events]
        else:
            _, intensity_all_events = self.model(time_history, pred_time, events_history, mask_history)
            # [batch_size, seq_len, num_events]

        mark_distribution = intensity_all_events / intensity_all_events.sum(dim=-1, keepdim=True)
        # [batch_size, seq_len, num_events]
        return pred_time, mark_distribution

    @torch.inference_mode()
    def next_event_prediction_mark_time(
        self: Self,
        time_history: torch.Tensor,
        events_history: torch.Tensor,
        events_next: torch.Tensor,
        mask_history: torch.Tensor,
        mean: float,
        std: float,
        sample_rate: int,
        mae_e_step: int,
        evaluation: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the next event prediction from the SAHP by MAE and F1.
        This function first predict the time of the next event then the mark of the next event given the TRUE time.
        This can ensure the mark accuracy is only influenced by the model, not by the predicted time.

        Args:
            self (Self): the SAHP model.
            time_history (torch.Tensor): the time of historical events
            time_next (torch.Tensor): the time of the next true events
            events_history (torch.Tensor): the mark of historical events
            events_next (torch.Tensor): the mark of the next true events
            mask_history (torch.Tensor): the mask of historical sequences, 1 meaning a true event, and 0 meaning a fake event.
            mask_next (torch.Tensor): the mask of the next true events
            mean (float): the mean of all time intervals
            std (float): the standard variance of all time intervals
            sample_rate (int): how many samples are needed for one time prediction
            mae_e_step (int): how many samples for one event are generated in one shot
            evaluation (bool): If true, we are in the evaluation mode, the mark distribution is at the time_next.
                               If false, we are in the prediction mode, the mark distribution is at the pred_time

        Returns:
            tuple[torch.Tensor, torch.Tensor]: predicted time and mark distribution
        """
        inf_val, resolution_inf, resolution_between_events = decide_resolution_inf_and_resolution_between_events(
            time_history, memory_ceiling, self.num_events, mean, std
        )
        time_next_inf = torch.ones_like(time_history, device=self.device) * inf_val

        (
            expanded_integral_all_events_to_inf,
            expanded_intensity_all_events_to_inf,
            timestamp,
        ) = self.model.integral_intensity_time_next_2d(
            events_history, time_history, time_next_inf, mask_history, resolution_inf
        )
        # 2 * [batch_size, seq_len, resolution_inf, num_events]
        expanded_integral_sum_over_events_to_inf = expanded_integral_all_events_to_inf.sum(dim=-1, keepdim=True)
        # [batch_size, seq_len, resolution_inf, 1]
        expanded_probability_inf = expanded_intensity_all_events_to_inf * torch.exp(
            -expanded_integral_sum_over_events_to_inf
        )
        # [batch_size, seq_len, resolution_inf, num_events]
        mark_distribution = approximate_integration(expanded_probability_inf, timestamp, dim=-2, only_integral=True)
        # [batch_size, seq_len, num_events]
        tau_sampled_all_event = self.sample_time(
            sampling_approach="its",
            task="mt",
            events_history=events_history,
            time_history=time_history,
            mask_history=mask_history,
            p_m=mark_distribution,
            resolution=resolution_between_events,
            number_of_total_samples=sample_rate,
            step=mae_e_step,
            inf_val=inf_val,
            mean=mean,
            std=std,
        )  # [sample_rate, batch_size, seq_len, num_events]
        tau_pred_all_event = tau_sampled_all_event.mean(dim=0)  # [batch_size, seq_len, num_events]

        if evaluation:
            events_next_mask = torch.nn.functional.one_hot(events_next, num_classes=self.num_events)
            # [batch_size, seq_len, num_events]
            pred_time = (tau_pred_all_event * events_next_mask).sum(dim=-1)  # [batch_size, seq_len]
        else:
            pred_time = tau_pred_all_event  # [batch_size, seq_len, num_events]

        return pred_time, mark_distribution

    @torch.inference_mode()
    def mean_absolute_error_e_and_f1(
        self: Self,
        time_history: torch.Tensor,
        time_next: torch.Tensor,
        events_history: torch.Tensor,
        events_next: torch.Tensor,
        mask_history: torch.Tensor,
        mask_next: torch.Tensor,
        mean: float,
        std: float,
        sample_rate: int,
        mae_e_step: int,
    ) -> tuple[
        float,
        list[float],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
    ]:
        """Evaluate the next event prediction from the SAHP by MAE-E and F1.
        This function first predict the mark of the next event then the time of the next event given the TRUE mark.
        This can ensure the time accuracy is only influenced by the model, not by the predicted mark.

        Args:
            self (Self): the SAHP model.
            time_history (torch.Tensor): the time of historical events
            time_next (torch.Tensor): the time of the next true events
            events_history (torch.Tensor): the mark of historical events
            events_next (torch.Tensor): the mark of the next true events
            mask_history (torch.Tensor): the mask of historical sequences, 1 meaning a true event, and 0 meaning a fake event.
            mask_next (torch.Tensor): the mask of the next true events
            mean (float): the mean of all time intervals
            std (float): the standard variance of all time intervals
            sample_rate (int): how many samples are needed for one time prediction
            mae_e_step (int): how many samples for one event are generated in one shot

        Returns:
            tuple[float, list[float], torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]: f1, top_k_acc, probability_integral_sum, p_m, tau_pred_all_event, (mae_per_event_with_predict_index_avg, mae_per_event_with_event_next_avg), (mae_per_event_with_predict_index, mae_per_event_with_event_next)
        """
        inf_val, resolution_inf, resolution_between_events = decide_resolution_inf_and_resolution_between_events(
            time_next, memory_ceiling, self.num_events, mean, std
        )
        time_next_inf = torch.ones_like(time_history, device=self.device) * inf_val

        (
            expanded_integral_all_events_to_inf,
            expanded_intensity_all_events_to_inf,
            timestamp,
        ) = self.model.integral_intensity_time_next_2d(
            events_history, time_history, time_next_inf, mask_history, resolution_inf
        )
        # 2 * [batch_size, seq_len, resolution_inf, num_events]

        expanded_integral_sum_over_events_to_inf = expanded_integral_all_events_to_inf.sum(dim=-1, keepdim=True)
        # [batch_size, seq_len, resolution_inf, 1]
        expanded_probability_inf = expanded_intensity_all_events_to_inf * torch.exp(
            -expanded_integral_sum_over_events_to_inf
        )
        # [batch_size, seq_len, resolution_inf, num_events]
        probability_integral_to_inf = approximate_integration(
            expanded_probability_inf, timestamp, dim=-2, only_integral=True
        )
        # [batch_size, seq_len, num_events]
        predicted_events = torch.argmax(probability_integral_to_inf, dim=-1)  # [batch_size, seq_len]

        f1, top_k_acc = get_f1_and_top_k_acc_in_mae_e(
            events_next, probability_integral_to_inf, mask_next, self.num_events
        )

        tau_pred_all_event = self.sample_time(
            sampling_approach="its",
            task="mt",
            events_history=events_history,
            time_history=time_history,
            mask_history=mask_history,
            p_m=probability_integral_to_inf,
            resolution=resolution_between_events,
            number_of_total_samples=sample_rate,
            step=mae_e_step,
            inf_val=inf_val,
            mean=mean,
            std=std,
        )  # [sample_rate, batch_size, seq_len, num_events]

        predicted_event_mask = F.one_hot(predicted_events.long(), num_classes=self.num_events)
        # [batch_size, seq_len, num_events]
        event_next_mask = F.one_hot(events_next.long(), num_classes=self.num_events)
        # [batch_size, seq_len, num_events]

        mae_per_event_with_predict_index = torch.abs(
            (tau_pred_all_event * predicted_event_mask.unsqueeze(dim=0)).sum(dim=-1) - time_next
        ) * mask_next.unsqueeze(dim=0)
        # [sample_rate, batch_size, seq_len]
        mae_per_event_with_event_next = torch.abs(
            (tau_pred_all_event * event_next_mask.unsqueeze(dim=0)).sum(dim=-1) - time_next
        ) * mask_next.unsqueeze(dim=0)
        # [sample_rate, batch_size, seq_len]

        mae_per_event_with_predict_index_avg = torch.sum(mae_per_event_with_predict_index, dim=-1) / mask_next.sum(
            dim=-1
        )
        # [sample_rate, batch_size]
        mae_per_event_with_event_next_avg = torch.sum(mae_per_event_with_event_next, dim=-1) / mask_next.sum(dim=-1)
        # [sample_rate, batch_size]
        # Calculate mean
        mae_per_event_with_predict_index = mae_per_event_with_predict_index.mean(dim=0)
        # [batch_size, seq_len]
        mae_per_event_with_event_next = mae_per_event_with_event_next.mean(dim=0)
        # [batch_size, seq_len]
        mae_per_event_with_predict_index_avg = mae_per_event_with_predict_index_avg.mean(dim=0)
        # [batch_size]
        mae_per_event_with_event_next_avg = mae_per_event_with_event_next_avg.mean(dim=0)
        # [batch_size]

        return (
            f1,
            top_k_acc,
            probability_integral_to_inf,
            tau_pred_all_event,
            (mae_per_event_with_predict_index_avg, mae_per_event_with_event_next_avg),
            (mae_per_event_with_predict_index, mae_per_event_with_event_next),
        )

    def extract_plot_data(self: Self, minibatch: list) -> tuple[Any]:
        """This function extracts input_time, input_events, input_intensity, mask, mean, and std from the minibatch.

        Args:
            self (Self): The model
            minibatch (list): The minibatch from the dataloader.

        Returns:
            tuple[Any]: The extracted data from the minibatch.
        """
        input_time, input_events, _, mask, input_intensity = minibatch[0]
        mean, std = minibatch[1]

        return input_time, input_events, input_intensity, mask, mean, std

    @torch.inference_mode()
    def figure_intensity(self: Self, input_data: list, opt: argparse.Namespace) -> None:
        """Function prober, used by evaluator to draw plots of the intensity function.

        You should declare the following arguments in your config file:
        1. ```int``` resolution: The number of interpolated points in a time interval between two adjoint events for integration estimation.
                                 The number of interpolated points counts the start and end point of the interval.
        Args:
            self (Self): the model
            input_data (list): the minibatch from the dataloader.
            opt (argparse.Namespace): the input arguments.
        """
        argument_check(opt, **{"resolution": int})

        input_time, input_events, input_intensity, mask, mean, std = self.extract_plot_data(input_data)

        time_history, time_next = self.divide_history_and_next(input_time)  # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
        # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)  # [batch_size, seq_len]

        expand_integral, expand_intensity, timestamp = self.model.integral_intensity_time_next_2d(
            events_history, time_history, time_next, mask_history, opt.resolution
        )
        # 3 * [batch_size, seq_len, resolution, num_events]
        check_tensor(expand_integral)
        check_tensor(expand_intensity)
        if expand_intensity.shape != expand_integral.shape:
            raise ValueError("Why expand_intensity and expand_integral have different shapes?")

        data = {
            "time_next": time_next,
            "events_next": events_next,
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
          1. ```int``` resolution: The number of interpolated points in a time interval between two adjoint events for integration estimation.
                                   The number of interpolated points counts the start and end point of the interval.

        Args:
            self (Self): the model
            input_data (list): the minibatch from the dataloader.
            opt (argparse.Namespace): the input arguments.
        """
        argument_check(opt, **{"resolution": int})

        input_time, input_events, input_intensity, mask, mean, std = self.extract_plot_data(input_data)

        time_history, time_next = self.divide_history_and_next(input_time)  # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
        # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)  # [batch_size, seq_len]

        expand_integral, expand_intensity, timestamp = self.model.integral_intensity_time_next_2d(
            events_history, time_history, time_next, mask_history, opt.resolution
        )
        # 3 * [batch_size, seq_len, resolution, num_events]
        check_tensor(expand_integral)
        check_tensor(expand_intensity)
        if expand_intensity.shape != expand_integral.shape:
            raise ValueError("Why expand_intensity and expand_integral have different shapes?")

        data = {
            "time_next": time_next,
            "events_next": events_next,
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
          1. ```int``` resolution: The number of interpolated points in a time interval between two adjoint events for integration estimation.
                                   The number of interpolated points counts the start and end point of the interval.


        Args:
            self (Self): the model
            input_data (list): the minibatch from the dataloader.
            opt (argparse.Namespace): the input arguments.
        """
        argument_check(opt, **{"resolution": int})

        input_time, input_events, input_intensity, mask, mean, std = self.extract_plot_data(input_data)

        time_history, time_next = self.divide_history_and_next(input_time)  # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
        # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)  # [batch_size, seq_len]

        expand_integral, expand_intensity, timestamp = self.model.integral_intensity_time_next_2d(
            events_history, time_history, time_next, mask_history, opt.resolution
        )
        # 3 * [batch_size, seq_len, resolution, num_events]
        check_tensor(expand_integral)
        check_tensor(expand_intensity)
        if expand_intensity.shape != expand_integral.shape:
            raise ValueError("Why expand_intensity and expand_integral have different shapes?")
        expand_probability = expand_intensity * torch.exp(-expand_integral.sum(dim=-1, keepdim=True))
        # [batch_size, seq_len, resolution, num_events]
        data = {
            "time_next": time_next,
            "events_next": events_next,
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
          1. ```int``` resolution: The number of interpolated points in a time interval between two adjoint events for integration estimation.
                                   The number of interpolated points counts the start and end point of the interval.
          2. ```int``` sample_rate: The number of interpolated points in a time interval between two adjoint events for integration estimation.
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

        input_time, input_events, input_intensity, mask, mean, std = self.extract_plot_data(input_data)

        time_history, time_next = self.divide_history_and_next(input_time)  # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
        # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)  # [batch_size, seq_len]

        data, timestamp = self.model.model_probe_function(
            events_history,
            time_history,
            time_next,
            mask_history,
            mask_next,
            opt.resolution,
        )

        pred_time, _ = self.next_event_prediction_time_mark(
            time_history,
            time_next,
            events_history,
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
            events_history,
            events_next,
            mask_history,
            mean=mean,
            std=std,
            sample_rate=self.sample_rate,
            mae_e_step=opt.mae_e_step,
            evaluation=False,
        )
        # [batch_size, seq_len, num_events] + [batch_size, seq_len, num_events]
        events_next_mask = torch.nn.functional.one_hot(events_next, num_classes=self.num_events)
        # [batch_size, seq_len, num_events]
        pred_time = (pred_time_all_marks * events_next_mask).sum(dim=-1)  # [batch_size, seq_len, num_events]
        maes_ptm = torch.abs(pred_time - time_next) * mask_next  # [batch_size, seq_len]
        top_k = evaluate_on_one_batch(
            mark_dist,
            events_next,
            mask_next,
            "top_k",
            dim_input=-2
        )
        # [batch_size, num_events]
        probability_sum = mark_dist.sum(dim=-1)  # [batch_size, seq_len]

        # Append additional info into the data dict.
        data.update(
            {
                "events_next": events_next,
                "time_next": time_next,
                "mask_next": mask_next,
                "mae_pt": mae_tm,
                "maes_ptm": maes_ptm,
                "top_k": top_k,
                "tau_pred_all_event": pred_time_all_marks,
                "probability_sum": probability_sum,
                "timestamp": timestamp,
            }
        )

        generate_debug_figure(data, opt)

    # Evaluation over the entire dataset.
    @torch.inference_mode()
    def get_spearman_and_l1(self: Self, input_data: list, opt: argparse.Namespace) -> tuple[list, list]:
        """Used by evaluator to calculate the average gap between the predicted and real distribution using L1 distance and spearman coefficient.

        You should declare the following arguments in your config file:
        1. ```int``` resolution: The number of interpolated points in a time interval between two adjoint events for integration estimation.
                                 The number of interpolated points counts the start and end point of the interval.

        Args:
            self (Self): the model
            input_data (list): the minibatch from the dataloader.
            opt (argparse.Namespace): the input arguments.

        Returns:
            tuple[list, list]: the spearman and l1
        """
        argument_check(opt, **{"resolution": int})

        input_time, input_events, input_intensity, mask, mean, std = self.extract_plot_data(input_data)
        time_history, time_next = self.divide_history_and_next(input_time)  # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
        # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)  # [batch_size, seq_len]

        expand_integral, expand_intensity, timestamp = self.model.integral_intensity_time_next_2d(
            events_history, time_history, time_next, mask_history, opt.resolution
        )
        # 3 * [batch_size, seq_len, resolution, num_events]
        check_tensor(expand_integral)
        check_tensor(expand_intensity)
        if expand_intensity.shape != expand_integral.shape:
            raise ValueError("Why expand_intensity and expand_integral have different shapes?")
        expand_probability = expand_intensity * torch.exp(-expand_integral.sum(dim=-1, keepdim=True))
        # [batch_size, seq_len, resolution, num_events]
        expand_probability = expand_probability.sum(dim=-1)  # [batch_size, seq_len, resolution]
        true_probability = expand_true_probability(time_next, input_intensity, opt)
        # [batch_size, seq_len, resolution] or batch_size * None

        expand_probability, true_probability, timestamp = move_from_tensor_to_ndarray(
            expand_probability, true_probability, timestamp
        )
        spearman = evaluate_on_one_batch(expand_probability, true_probability, mask_next, "spearman", -2, -2, -1)
        l1 = evaluate_on_one_batch(
            expand_probability,
            true_probability,
            mask_next,
            "l1",
            -2,
            -2,
            -1,
            additional_inputs=[
                timestamp,
            ],
        )

        return spearman.tolist(), l1.tolist()

    @torch.inference_mode()
    def get_mae_and_f1(self: Self, input_data: list, opt: argparse.Namespace) -> tuple[Any]:
        """Used by evaluator to evaluate the performance of predicted time from p(t) and mark from p(m|t).

          You should declare the following arguments in your config file:
          1. ```int``` sample_rate: The number of interpolated points in a time interval between two adjoint events for integration estimation.
                                    The number of interpolated points counts the start and end point of the interval.
          2. ```int``` mae_step: This parameter controls how many samples are generated in one shot when sampling from p(t).

        Args:
            self (Self): the model
            input_data (list): the minibatch from the dataloader.
            opt (argparse.Namespace): the input arguments.

        Returns:
            tuple[Any]: the results: mae, acc, macro-f1, micro-f1,
                        distribution of mark at the true time (evaluation = True),
                        the mark of the next event.
        """
        argument_check(opt, **{"sample_rate": int, "mae_step": int})

        input_time, input_events, input_intensity, mask, mean, std = self.extract_plot_data(input_data)
        time_history, time_next = self.divide_history_and_next(input_time)  # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
        # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)  # [batch_size, seq_len]
        pred_time, mark_dist = self.next_event_prediction_time_mark(
            time_history,
            time_next,
            events_history,
            mask_history,
            mean=mean,
            std=std,
            sample_rate=self.sample_rate,
            mae_step=opt.mae_step,
        )
        # [batch_size, seq_len] + [batch_size, seq_len, num_events]
        mae = torch.abs(pred_time - time_next) * mask_next  # [batch_size, seq_len]
        pred_mark = mark_dist.argmax(dim=-1)  # [batch_size, seq_len]
        results = evaluate_on_one_batch(pred_mark, events_next, mask_next, ["acc", "macro-f1", "micro-f1"])
        acc = results["acc"]
        macro_f1 = results["macro-f1"]
        micro_f1 = results["micro-f1"]

        mae, events_next, mark_dist, acc, macro_f1, micro_f1, mask_next = move_from_tensor_to_ndarray(
            mae, events_next, mark_dist, acc, macro_f1, micro_f1, mask_next
        )
        mae, events_next, mark_dist = break_batched_inputs_into_seqs(mask_next, mae, events_next, mark_dist)

        return mae, acc.tolist(), macro_f1.tolist(), micro_f1.tolist(), mark_dist, events_next

    @torch.inference_mode()
    def get_mae_e_and_f1(self: Self, input_data: list, opt: argparse.Namespace) -> tuple[Any]:
        """Used by evaluator to evaluate the performance of predicted time from p(m) and mark from p(t|m).

           You should declare the following arguments in your config file:
           1. ```int``` sample_rate: The number of interpolated points in a time interval between two adjoint events for integration estimation.
                                     The number of interpolated points counts the start and end point of the interval.
           2. ```int``` mae_e_step: This parameter controls how many samples are generated in one shot when sampling from p(t|m).
        Args:
            self (Self): the model
            input_data (list): the minibatch from the dataloader.
            opt (argparse.Namespace): the input arguments.

        Returns:
            tuple[Any]: the results: mae_e, acc, macro-f1, micro-f1,
                        distribution of mark at the true time (evaluation = True),
                        predicted time of all marks, the true time of the next event,
                        the true mark of the next event
        """
        argument_check(opt, **{"sample_rate": int, "mae_e_step": int})

        input_time, input_events, _, mask, mean, std = self.extract_plot_data(input_data)
        time_history, time_next = self.divide_history_and_next(input_time)  # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
        # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)  # [batch_size, seq_len]

        pred_time_all_marks, mark_dist = self.next_event_prediction_mark_time(
            time_history,
            events_history,
            events_next,
            mask_history,
            mean=mean,
            std=std,
            sample_rate=self.sample_rate,
            mae_e_step=opt.mae_e_step,
            evaluation=False,
        )
        # [batch_size, seq_len, num_events] + [batch_size, seq_len, num_events]
        events_next_mask = torch.nn.functional.one_hot(events_next, num_classes=self.num_events)
        # [batch_size, seq_len, num_events]
        pred_time = (pred_time_all_marks * events_next_mask).sum(dim=-1)  # [batch_size, seq_len, num_events]
        mae_e = torch.abs(pred_time - time_next) * mask_next  # [batch_size, seq_len]
        pred_mark = mark_dist.argmax(dim=-1)  # [batch_size, seq_len]
        results = evaluate_on_one_batch(pred_mark, events_next, mask_next, ["acc", "macro-f1", "micro-f1"])
        acc = results["acc"]
        macro_f1 = results["macro-f1"]
        micro_f1 = results["micro-f1"]

        (
            mae_e,
            events_next,
            mark_dist,
            acc,
            macro_f1,
            micro_f1,
            mask_next,
            pred_time_all_marks,
            time_next,
        ) = move_from_tensor_to_ndarray(
            mae_e,
            events_next,
            mark_dist,
            acc,
            macro_f1,
            micro_f1,
            mask_next,
            pred_time_all_marks,
            time_next,
        )
        mae_e, mark_dist, pred_time_all_marks, time_next, events_next = break_batched_inputs_into_seqs(
            mask_next, mae_e, mark_dist, pred_time_all_marks, time_next, events_next
        )

        return (
            mae_e,
            acc.tolist(),
            macro_f1.tolist(),
            micro_f1.tolist(),
            mark_dist,
            pred_time_all_marks,
            time_next,
            events_next,
        )

    @torch.inference_mode()
    def get_which_event_first(self, input_data, opt):
        """
        Used by evaluator to evaluate the performance of predicted time from p(m) and mark from p(t|m).
        Instead of picking the most probable event, we pick the event predicted to happen first.

        You should declare the following arguments in your config file:
        1. ```int``` sample_rate: how many time samples from the time distribution are needed.
        2. ```int``` which_event_first_step: This parameter controls how many samples are generated in one shot when sampling from p(t|m).

        ### Args
            * ```list``` input_data
              shape: ```[[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]```
              data structure: [[input_time, input_events, score, mask], (mean, std)]
              data type: [```torch.tensor```, ```torch.tensor```, ```torch.tensor```, ```torch.tensor```, (```float```, ```float```)]
              The input minibatch.
            * ```namespace``` opt
              plot and model configs

        ### Outputs:
            * ```np.ndarray``` maes
              shape: ```[batch_size, seq_len]```
              The MAE values when we pick predicted times using real marks.
            * ```float``` f1
              The f1 value shows the accuracy of the predicted marks.
        """
        argument_check(opt, **{"sample_rate": int, "which_event_first_step": int})

        input_time, input_events, input_intensity, mask, mean, std = self.extract_plot_data(input_data)
        time_history, time_next = self.divide_history_and_next(input_time)  # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
        # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)  # [batch_size, seq_len]

        inf_val, resolution_inf, resolution_between_events = decide_resolution_inf_and_resolution_between_events(
            time_next, memory_ceiling, self.num_events, mean, std
        )
        time_next_inf = torch.ones_like(time_history, device=self.device) * inf_val
        # [batch_size, seq_len]
        (
            expanded_integral_all_events_to_inf,
            expanded_intensity_all_events_to_inf,
            timestamp,
        ) = self.model.integral_intensity_time_next_2d(
            events_history, time_history, time_next_inf, mask_history, resolution_inf
        )
        # 2 * [batch_size, seq_len, resolution, num_events]
        expanded_probability_inf = (
            torch.exp(-expanded_integral_all_events_to_inf.sum(dim=-1, keepdim=True))
            * expanded_intensity_all_events_to_inf
        )
        # [batch_size, seq_len, resolution, num_events]
        probability_integral_to_inf = approximate_integration(
            expanded_probability_inf, timestamp, dim=-2, only_integral=True
        )
        # [batch_size, seq_len, num_events]
        # step 2: get the time prediction for that kind of event
        tau_pred_all_event = self.sample_time(
            sampling_approach="its",
            task="mt",
            events_history=events_history,
            time_history=time_history,
            mask_history=mask_history,
            p_m=probability_integral_to_inf,
            resolution=resolution_between_events,
            number_of_total_samples=opt.sample_rate,
            step=opt.which_event_first_step,
            inf_val=inf_val,
            mean=mean,
            std=std,
        )  # [sample_rate, batch_size, seq_len, num_events]

        sampled_times_mean = tau_pred_all_event.mean(dim=0)  # [batch_size, seq_len, num_events]
        predicted_time, predicted_mark = sampled_times_mean.min(dim=-1)  # [batch_size, seq_len] + [batch_size, seq_len]
        maes = torch.abs(time_next - predicted_time) * mask_next  # [batch_size, seq_len]

        events_pred_index = predicted_mark[mask_next == 1]
        events_true = events_next[mask_next == 1]
        events_true, events_pred_index = move_from_tensor_to_ndarray(events_true, events_pred_index)
        f1_val = evaluate_func("f1")(y_true=events_true, y_pred=events_pred_index)

        maes = move_from_tensor_to_ndarray(maes)

        return maes, f1_val

    def samples_from_et(self: Self, input_data: list, opt: argparse.Namespace) -> tuple[Any]:
        """This function samples from the distribution p(m, t) by sampling the mark first from p(m) then time from p(t|m).
          All samples can later be used to draw the distribution plot.

          You should declare the following arguments in your config file:
          1. ```int``` sample_rate: how many time samples from the time distribution are needed.
          2. ```int``` sample_substep: This parameter controls how many samples are generated in one shot when sampling from p(t|m).

        Args:
            self (Self): the model
            input_data (list): the minibatch from the dataloader.
            opt (argparse.Namespace): the input arguments.

        Returns:
            tuple[Any]: the results: mae_e, acc, macro-f1, micro-f1,
                        distribution of mark at the true time (evaluation = True),
                        predicted time of all marks, the true time of the next event,
                        the true mark of the next event
        """
        argument_check(opt, **{"sample_rate": int, "sample_substep": int})

        input_time, input_events, input_intensity, mask, mean, std = self.extract_plot_data(input_data)
        time_history, time_next = self.divide_history_and_next(input_time)  # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
        # [batch_size, seq_len]

        input_time, input_events, input_intensity, mask, mean, std = self.extract_plot_data(input_data)
        time_history, time_next = self.divide_history_and_next(input_time)  # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
        # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)  # [batch_size, seq_len]

        inf_val, resolution_inf, resolution_between_events = decide_resolution_inf_and_resolution_between_events(
            time_next, memory_ceiling, self.num_events, mean, std
        )
        time_next_inf = torch.ones_like(time_history, device=self.device) * inf_val
        # [batch_size, seq_len]
        (
            expanded_integral_all_events_to_inf,
            expanded_intensity_all_events_to_inf,
            timestamp,
        ) = self.model.integral_intensity_time_next_2d(
            events_history, time_history, time_next_inf, mask_history, resolution_inf
        )
        # 2 * [batch_size, seq_len, resolution, num_events]
        expanded_probability_inf = (
            torch.exp(-expanded_integral_all_events_to_inf.sum(dim=-1, keepdim=True))
            * expanded_intensity_all_events_to_inf
        )
        # [batch_size, seq_len, resolution, num_events]
        probability_integral_to_inf = approximate_integration(
            expanded_probability_inf, timestamp, dim=-2, only_integral=True
        )
        # [batch_size, seq_len, num_events]
        # step 2: get the time prediction for that kind of event
        tau_pred_all_event = self.sample_time(
            sampling_approach="its",
            task="mt",
            events_history=events_history,
            time_history=time_history,
            mask_history=mask_history,
            p_m=probability_integral_to_inf,
            resolution=resolution_between_events,
            number_of_total_samples=opt.sample_rate,
            step=opt.sample_substep,
            inf_val=inf_val,
            mean=mean,
            std=std,
        )  # [sample_rate, batch_size, seq_len, num_events]

        return tau_pred_all_event, probability_integral_to_inf

    def get_event_embedding(self, input_events):
        return self.model.get_event_embedding(input_events)  # [batch_size, seq_len, d_history]

    def ehd_perplexity(
        self,
        padded_filtered_time,
        padded_filtered_events,
        padded_filtered_event_embeddings,
        padded_filtered_masks,
        seq_len_x,
        mean,
        std,
    ):
        padded_filtered_time_history, padded_filtered_time_next = self.divide_history_and_next(padded_filtered_time)
        # 2 * [batch_size, filtered_seq_len - 1]
        padded_filtered_events_history, padded_filtered_events_next = self.divide_history_and_next(
            padded_filtered_events
        )
        # 2 * [batch_size, filtered_seq_len- 1]
        (
            padded_filtered_events_embeddings_history,
            padded_filtered_events_embeddings_next,
        ) = self.divide_history_and_next(
            padded_filtered_event_embeddings
        )  # 2 * [batch_size, filtered_seq_len- 1, d_history]
        padded_filtered_mask_history, padded_filtered_mask_next = self.divide_history_and_next(padded_filtered_masks)
        # [batch_size, filtered_seq_len - 1]
        the_number_of_events_per_sequence = padded_filtered_mask_next.sum(dim=-1)
        # [batch_size]
        # \\int_{t}^{+\\inf}{p(m, \\tau|\\mathcal{H})d\\tau}
        (
            padded_filtered_intensity_integral_from_t_o_to_t,
            padded_filtered_intensity_at_t,
        ) = self.model(
            padded_filtered_time_history,
            padded_filtered_time_next,
            padded_filtered_events_embeddings_history,
            padded_filtered_mask_history,
            custom_events_history=True,
        )
        # [batch_size, filtered_seq_len - 1, num_events]
        padded_filtered_mask_next_without_dummy = self.remove_dummy_event_from_mask(padded_filtered_mask_next)
        # [batch_size, filtered_seq_len - 1]
        padded_filtered_events_next_without_dummy = (
            padded_filtered_events_next * padded_filtered_mask_next_without_dummy
        )
        # [batch_size, filtered_seq_len - 1]
        event_mask = torch.nn.functional.one_hot(padded_filtered_events_next_without_dummy, num_classes=self.num_events)
        # [batch_size, filtered_seq_len - 1, num_events]
        padded_filtered_intensity_at_t = (padded_filtered_intensity_at_t * event_mask).sum(dim=-1)
        # [batch_size, filtered_seq_len - 1]
        log_probability = torch.log(
            padded_filtered_intensity_at_t + self.epsilon
        ) - padded_filtered_intensity_integral_from_t_o_to_t.sum(dim=-1)
        # [batch_size, filtered_seq_len - 1]
        log_probability_x = pick_log_probability(log_probability, the_number_of_events_per_sequence, seq_len_x)
        # [batch_size, seq_len_x]
        # -\\frac{1}{N} \\log p(\\mathbf{x}_o|\\mathcal{H})
        log_perplexity = -log_probability_x.mean(dim=-1)  # [batch_size]

        return log_perplexity

    def convert_missing_mask_to_gap_mask(self, missing_mask):
        # input shape: [num_samples, seq_len]

        masks = []
        for missing_mask_per_seq in missing_mask:
            current_in_missing = False
            mask_current_seq = []
            for item in missing_mask_per_seq[1:]:
                if item == 1 and not current_in_missing:
                    mask_current_seq.append(1)
                elif item == 1 and current_in_missing:
                    current_in_missing = False
                elif item == 0 and not current_in_missing:
                    mask_current_seq.append(0)
                    current_in_missing = True
                else:
                    continue

            masks.append(mask_current_seq)

        return masks

    def cppod_evaluation(self, input_data, opt):
        """
        Take care. This function only evaluates the omission outlier.
        Interestingly, the original CPPOD code seems only focusing on omission too as only omission scores are recorded in model.detect_outlier().
        """
        (
            forward_complete_data,
            backward_complete_data,
            padded_obs_data,
            padded_backward_obs_event_seq,
            (mean, std),
        ) = input_data

        roc_result = []
        for (
            obs_time_for_one_seq,
            obs_events_for_one_seq,
            obs_mask_for_one_seq,
            missing_mask_for_one_seq,
            _,
        ) in padded_obs_data:
            obs_time_history_for_one_seq, obs_time_next_for_one_seq = self.divide_history_and_next(obs_time_for_one_seq)
            # [batch_size, seq_len] * 2
            obs_events_history_for_one_seq, obs_events_next_for_one_seq = self.divide_history_and_next(
                obs_events_for_one_seq
            )
            # [batch_size, seq_len] * 2
            obs_mask_history_for_one_seq, obs_mask_next_for_one_seq = self.divide_history_and_next(obs_mask_for_one_seq)
            # [batch_size, seq_len]

            missing_mask_for_one_seq = self.convert_missing_mask_to_gap_mask(missing_mask_for_one_seq)
            # [num_samples, ...]
            integral_all_events, intensity_all_events = self.model(
                obs_time_history_for_one_seq.float(),
                obs_time_next_for_one_seq.float(),
                obs_events_history_for_one_seq,
                obs_mask_history_for_one_seq,
            )
            # [num_samples, seq_len, num_events]

            integral_sum = integral_all_events.sum(dim=-1)  # [num_samples, seq_len]
            intensity_sum = intensity_all_events.sum(dim=-1)  # [num_samples, seq_len]

            all_roauc_area = []
            for (
                integral_sum_per_seq_per_sample,
                missing_mask_for_one_seq_per_sample,
            ) in zip(integral_sum, missing_mask_for_one_seq):
                sample_len = len(missing_mask_for_one_seq_per_sample)
                selected_integral_sum_per_seq_per_sample = move_from_tensor_to_ndarray(
                    integral_sum_per_seq_per_sample[:sample_len]
                )

                roauc_area = roc_auc_score(
                    y_true=np.array(missing_mask_for_one_seq_per_sample) ^ 1,
                    y_score=selected_integral_sum_per_seq_per_sample,
                )
                all_roauc_area.append(roauc_area)

            roc_result.append(np.mean(all_roauc_area))

        roc_result = np.array(roc_result)
        return roc_result

    def cppod_commission_evaluation(self, input_data, opt):
        (time_seq, events, commission, mask), (mean, std) = input_data

        time_history, time_next = self.divide_history_and_next(time_seq)  # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(events)  # [batch_size, seq_len]
        _, commission_next = self.divide_history_and_next(commission)  # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)  # [batch_size, seq_len]
        time_history = time_history.float()
        time_next = time_next.float()

        _, intensity_all_events = self.model(time_history, time_next, events_history, mask_history)
        # 2 * [batch_size, seq_len, num_events]

        intensity_sum_from_tl_to_time_next = intensity_all_events.sum(dim=-1)
        # [batch_size, seq_len]
        score = -intensity_sum_from_tl_to_time_next  # [batch_size, seq_len]

        packed_data = zip(score, commission_next, mask_next)
        all_roauc_area = []

        for score_per_seq, commission_next_per_seq, mask_next_per_seq in packed_data:
            available_score = score_per_seq[mask_next_per_seq]
            available_commission_label = commission_next_per_seq[mask_next_per_seq]

            available_score, available_commission_label = move_from_tensor_to_ndarray(
                available_score, available_commission_label
            )
            roauc_area = roc_auc_score(y_true=available_commission_label, y_score=available_score)
            all_roauc_area.append(roauc_area)

        all_roauc_area = np.array(all_roauc_area)
        return all_roauc_area

    def train_step(self, minibatch):
        """
        This function unpacks the minibatch, calls the train_procedure() to calculate the loss, and do the backpropagation.

        ### Args
            * ```torch.nn.Module``` model
              The MTPP model that we train.
            * ```list``` minibatch
              shape: ```[[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]```
              data structure: [[input_time, input_events, score, mask], (mean, std)]
              data type: [```torch.tensor```, ```torch.tensor```, ```torch.tensor```, ```torch.tensor```, (```float```, ```float```)]
              The input minibatch.
            * ```torch.device``` device
              where we train the model.

        ### Outputs:
            * ```float``` time_loss_without_dummy
              The average NLL loss without dummy events, specifically the start and the end event.
            * ```float``` fact
              The average NLL loss with the real distribution. This value only makes sense for synthetic datasets.
            * ```float``` events_loss
              The average cross-entropy loss of the event prediction distribution. The value is only for performance measure porpose.
              The training loss does not and should not include this value.
        """
        self.train()

        time, events, score, mask = minibatch[0]  # 3 * [batch_size, seq_len + 1, 1] & [batch_size, seq_len, 1]
        mean, std = minibatch[1]

        loss, time_loss_without_dummy, events_loss, the_number_of_events = self.forward(
            "train", time, events, mask, mean, std
        )
        loss.backward()

        time_loss_without_dummy = time_loss_without_dummy.item()
        events_loss = events_loss.item()
        fact = score.sum().item() / the_number_of_events

        return time_loss_without_dummy, fact, events_loss

    def evaluation_step(self, minibatch):
        """
        This function unpacks the minibatch, calls the evaluation_procedure() to calculate the metrics.

        ### Args
            * ```torch.nn.Module``` model
              The MTPP model that we train.
            * ```list``` minibatch
              shape: ```[[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]```
              data structure: [[input_time, input_events, score, mask], (mean, std)]
              data type: [```torch.tensor```, ```torch.tensor```, ```torch.tensor```, ```torch.tensor```, (```float```, ```float```)]
              The input minibatch.
            * ```torch.device``` device
              where we train the model.

        ### Outputs:
            * ```float``` time_loss
              The average NLL loss without dummy events, specifically the start and the end event.
            * ```float``` loss_survival
              The average NLL loss of the end event, which is the integral of the intensity function from the last occurred event to the end time.
            * ```float``` fact
              The average NLL loss with the real distribution. This value only makes sense for synthetic datasets.
            * ```float``` events_loss
              The average cross-entropy loss of the event prediction distribution. The value is only for performance measure porpose.
            * ```float``` mae
              The average error between predicted time and real time.
            * ```float``` f1
              The prediction accuracy of predicted marks.
        """
        self.eval()

        time, events, score, mask = minibatch[0]  # 3 * [batch_size, seq_len + 1, 1] & [batch_size, seq_len, 1]
        mean, std = minibatch[1]

        (
            time_loss,
            loss_survival,
            events_loss,
            mae,
            acc,
            macro_f1,
            micro_f1,
            the_number_of_events,
        ) = self.forward("evaluate", time, events, mask, mean, std)

        time_loss = time_loss.item()
        loss_survival = loss_survival.item()
        events_loss = events_loss.item()
        fact = score.sum().item() / the_number_of_events

        return time_loss, loss_survival, fact, events_loss, mae, acc, macro_f1, micro_f1

    def postprocess(self, input_data, procedure):
        """
        This function makes some modifications to the output of training_step() and evaluation_step().

        ### Args
            * ```list``` input_data
              The output of either training_step() or evaluation_step().
            * ```str``` procedure
              This string tells the function which function the input comes from.

        ### Outputs:
            * ```list```
              The postprocessed outputs.
        """

        def train_postprocess(input_data):
            """
            Training process
            [absolute loss, relative loss, events loss]
            """
            return [input_data[0], input_data[0] - input_data[1], input_data[2]]

        def test_postprocess(input_data):
            """
            Evaluation process
            [absolute loss, relative loss, events loss, mae value]
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

        return train_postprocess(input_data) if procedure == "Training" else test_postprocess(input_data)

    def log_print_format(self, input_data, procedure):
        """
        This function packs the procedure input into a dict that can be handled by trainer and evaluator for logging.

        ### Args
            * ```list``` input
              The output of either training_step() or evaluation_step().
            * ```str``` procedure
              This string tells the function which function the input comes from.

        ### Outputs:
            * ```dict``` format_dict
              format: {..., <variable name>: {'data': <value>, 'num_format': <num_format>, 'suffix': <suffix>}, ...}
              example: {..., 'memory': {'data': 12.123456, 'num_format': ':2.4f', 'suffix': 'GiB'}, ...}
              The formated results.
        """

        def train_log_print_format(input_data):
            format_dict = {}
            format_dict["absolute_loss"] = pack_one_value_to_dict(input_data[0])
            format_dict["relative_loss"] = pack_one_value_to_dict(input_data[1])
            format_dict["events_loss"] = pack_one_value_to_dict(input_data[2])
            return format_dict

        def test_log_print_format(input_data):
            format_dict = {}
            format_dict["absolute_NLL_loss"] = pack_one_value_to_dict(input_data[0])
            format_dict["avg_survival_loss"] = pack_one_value_to_dict(input_data[1])
            format_dict["relative_NLL_loss"] = pack_one_value_to_dict(input_data[2])
            format_dict["events_loss"] = pack_one_value_to_dict(input_data[3])
            format_dict["mae"] = pack_one_value_to_dict(input_data[4], "2.8f")
            format_dict["acc"] = pack_one_value_to_dict(input_data[5], "2.8f")
            format_dict["macro-f1"] = pack_one_value_to_dict(input_data[6], "2.8f")
            format_dict["micro-f1"] = pack_one_value_to_dict(input_data[7], "2.8f")
            return format_dict

        return train_log_print_format(input_data) if procedure == "Training" else test_log_print_format(input_data)

    # The maximum length of the format_dict in different procedures.
    format_dict_length = 8

    def choose_metric(self, evaluation_report_format_dict, test_report_format_dict):
        """
        This function helps the trainer to pick the best checkpoint based on several metrics.

        ### Args
            * ```dict``` evaluation_report_format_dict
            * ```dict``` test_report_format_dict
              The formated output of training_step() and evaluation_step().

        ### Outputs:
            * ```list```
              The picked metrics used for model select.
            * ```list```
              The name of these metrics.
        """
        return [
            evaluation_report_format_dict["absolute_NLL_loss"],
            test_report_format_dict["absolute_NLL_loss"],
        ], ["evaluation_absolute_loss", "test_absolute_loss"]

    """
    metric number is the length of the output of choose_metric
    """
    metric_number = 2
