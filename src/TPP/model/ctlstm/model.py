from typing import Self

import torch
import torch.nn.functional as F
from einops import rearrange

from src.toolbox.algorithms import approximate_integration
from src.toolbox.metrics import evaluate_on_one_batch
from src.toolbox.misc import (
    argument_check,
    check_tensor,
    compile_model,
    move_from_tensor_to_ndarray,
    pack_one_value_to_dict,
)
from src.TPP.model.basic_tpp_model import BasicModel, memory_ceiling
from src.TPP.model.ctlstm.plot import (
    generate_debug_figure,
    generate_integral_figure,
    generate_intensity_figure,
    generate_probability_figure,
)
from src.TPP.model.ctlstm.sample import sample_time
from src.TPP.model.ctlstm.submodel import CTLSTM
from src.TPP.model.utils import (
    BalancedSamplingFromDistributionMixin,
    GetWhichEventFirstMixin,
    NextEventPredictionMarkTimeMixin,
    NextEventPredictionTimeMarkMixin,
    SpearmanL1EvaluationMixin,
    decide_resolution_inf_and_resolution_between_events,
)


class CTLSTMWrapper(
    BasicModel,
    BalancedSamplingFromDistributionMixin,
    NextEventPredictionTimeMarkMixin,
    NextEventPredictionMarkTimeMixin,
    SpearmanL1EvaluationMixin,
    GetWhichEventFirstMixin,
):
    """
    continuous-time LSTM, the backbone of the Neural Hawkes Process, proposed by Mei et al. at NeurIPS 2017.
    """

    def __init__(
        self,
        opt,
        device,
        d_input=64,
        history_module_name="LSTM",
        history_encoder_layers=1,
        d_mark_embedding=64,
        d_hidden=256,
        dropout=0.1,
        epsilon=1e-20,
        sample_rate=32,
        mae_step=8,
        mae_e_step=8,
        integration_sample_rate=100,
        survival_loss_during_training=True,
    ):
        """
        This function creates a CTLSTM model.

        ### Args
            * ```int``` d_mark_embedding
              The dimension of the mark embeddings.
            * ```str``` history_module_name
              Which RNN model do we use to encode the history? Default is LSTM. We don't recommend to change it to something else.
            * ```int``` d_hidden
              The dimension of the history representation.
            * ```float``` dropout
              Dropout rate for the history encoder. Only works when history_encoder_layers > 1.
            * ```int``` history_encoder_layers
              How many layer of RNN our model will have?
            * ```int``` d_input
              The dimension of the cumulative hazard function network.
            * ```namespace``` opt
              Model arguments.
            * ```torch.device``` device
              Running models on GPU or CPU?
            * ```float``` epsilon
              Shiftting the calculated intensity function and probability distribution by a little bit so that ```torch.log()``` won't fail.
            * ```int``` sample_rate
              This tells how many time samples from the time distribution are needed for one time prediction.
            * ```int``` mae_step
              This parameter controls how many samples are generated in one shot when sampling from p(t).
            * ```int``` mae_e_step
              This parameter controls how many samples are generated in one shot when sampling from all p(t|m)s at the same time.
              mae_step and mae_e_step are useful when you cannot get sample_rate time samples from time distributions because of insufficient GPU memory.
            * ```int``` integration_sample_rate
              The number of interpolated points in a time interval between two adjoint marks for integration estimation.
              The number of interpolated points counts the start and end point of the interval.
            * ```bool``` survival_loss_during_training
              When true, the training loss includes the integral between the last observed mark to the end time T. Most of time this argument should be true.
        """
        super().__init__()
        self.device = device
        self.num_marks = opt.info_dict["num_marks"]
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

        self.model = CTLSTM(
            device=device,
            num_marks=self.num_marks,
            history_module_name=history_module_name,
            d_mark_embedding=d_mark_embedding,
            d_input=d_input,
            d_hidden=d_hidden,
            history_encoder_layers=history_encoder_layers,
            dropout=dropout,
            integration_sample_rate=integration_sample_rate,
        )

        self.model = compile_model(self.model, opt.compile, opt.compile_backend)

    def divide_history_and_next(self, input_data):
        """
        Extract the history and prediction sequences from the input sequence.

        ### Args
            * ```torch.tensor``` input
              shape: [batch_size, seq_len + 1]
              The input sequence.

        ### Outputs
            * ```torch.tensor``` input_history
              shape: [batch_size, seq_len]
              The history sequence extracted from the original input.
            * ```torch.tensor``` input_next
              shape: [batch_size, seq_len]
              The history sequence extracted from the original input.
        """
        input_history, input_next = input_data[:, :-1].clone(), input_data[:, 1:].clone()
        return input_history, input_next

    def remove_dummy_events_from_mask(self: Self, mask: torch.Tensor) -> torch.Tensor:
        """Remove the dummy events by altering the mask.

        Args:
            self (Self): the SAHP model
            mask (torch.Tensor): the input mask tensor. shape: [batch, seq_len]

        Returns:
            torch.Tensor: The input mask tensor with the dummy events at the end removed. shape: [batch, seq_len]
        """
        dummy_indices = mask.sum(dim=1, dtype=torch.long) - 1  # [batch_size]
        mask_without_dummy = mask.clone()  # [batch_size, seq_len]
        batch_indices = torch.arange(mask.size(0), device=mask.device)  # [batch_size]
        mask_without_dummy[batch_indices, dummy_indices] = 0  # [batch_size, seq_len]

        return mask_without_dummy

    def forward(self, task_name, *args, **kwargs):
        """
        The entrance of the CTLSTM.

        ### Args
            * ```str``` task_name
              The name of the executed task.
        """
        task_mapper = {
            "train": self.train_procedure,
            "evaluate": self.evaluate_procedure,
            "spearman_and_l1": self.get_spearman_and_l1,
            "mae_and_f1": self.get_mae_and_f1,
            "mae_e_and_f1": self.get_mae_e_and_f1,
            "which_mark_occurs_first": self.get_which_event_first,
            "balanced_sampling_from_distribution": self.balanced_sampling_from_distribution,
            # figure drawing funtions
            "intensity": self.figure_intensity,
            "integral": self.figure_integral,
            "probability": self.figure_probability,
            "debug": self.figure_debug,
            # For CPPOD, should be used with the od_generic dataloader.
            # "cppod_evaluation": self.cppod_evaluation,
            # "cppod_commission_evaluation": self.cppod_commission_evaluation,
        }

        return task_mapper[task_name](*args, **kwargs)

    def train_procedure(self, time, marks, mask, mean, std):
        """
        CTLSTM's forwardpropagation function for training.

        ### Args
            * ```torch.tensor``` time
              shape: ```[batch_size, seq_len + 1]```
              Time sequence for training.
            * ```torch.tensor``` marks
              shape: ```[batch_size, seq_len + 1]```
              Event sequence for training.
            * ```torch.tensor``` mask
              shape: ```[batch_size,, seq_len + 1]```
              Mask sequence. Events whose corresponding mask is 0 are dummy marks.
            * ```float``` mean
            * ```float``` std
              Used for input time scaling.

        ### Outputs
            * ```torch.tensor``` loss
              shape: ```[1]```
              The sum of NLL loss L = -log \\frac{\\partial \\Lambda^*(m, t)}{\\partial t} + \\Lambda^*(m, t) at each happened mark (the dummy mark at end time T included).
            * ```torch.tensor``` log_likeli_loss_without_dummy
              shape: ```[1]```
              The sum of NLL loss L = -log \\frac{\\partial \\Lambda^*(m, t)}{\\partial t} + \\Lambda^*(m, t) at each happened mark (the dummy mark at end time T excluded).
            * ```torch.tensor``` marker_loss_without_dummy
              shape: ```[1]```
              The sum of the mark loss: L = -log \\frac{\\lambda^*(m, t)}{\\sum_{n \\in M}{\\lambda^*(n, t)}} where m is the mark of the real mark.
            * ```int``` the_number_of_marks
              The number of legit marks.
        """
        time_history, time_next = self.divide_history_and_next(time)  # [batch_size, seq_len] * 2
        marks_history, marks_next = self.divide_history_and_next(marks)  # [batch_size, seq_len] * 2
        mask_history, mask_next = self.divide_history_and_next(mask)  # [batch_size, seq_len]

        integral_all_marks, intensity_all_marks = self.model(time_history, time_next, marks_history)
        # 2 * [batch_size, seq_len, num_marks]

        mask_next_without_dummy = self.remove_dummy_events_from_mask(mask_next)  # [batch_size, seq_len]
        mark_next_without_dummy = (mask_next_without_dummy * marks_next).long()
        # [batch_size, seq_len]
        the_number_of_marks = mask_next_without_dummy.sum().item()

        # L = \\sum_{i}{\\lambda^_k*(t_i)} + \\int_{t_0}^{t_n}{\\sum_{k}{\\lambda^*_k(\\tau)}d\\tau}
        log_likeli_loss_without_dummy, mark_loss_without_dummy = self.loss_function(
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

        return (
            loss / the_number_of_marks,
            log_likeli_loss_without_dummy / the_number_of_marks,
            mark_loss_without_dummy / the_number_of_marks,
            the_number_of_marks,
        )

    @torch.inference_mode()
    def evaluate_procedure(self, time, marks, mask, mean, std):
        """
        CTLSTM's forwardpropagation function for evaluation.

        ### Args
            * ```torch.tensor``` time
              shape: ```[batch_size, seq_len + 1]```
              Time sequencalculatesce for training.
            * ```torch.tensor``` marks
              shape: ```[batch_size, seq_len + 1]```
              Event sequence for training.
            * ```torch.tensor``` mask
              shape: ```[batch_size,, seq_len + 1]```
              Mask sequence. Events whose corresponding mask is 0 are dummy marks.
            * ```float``` mean
            * ```float``` std
              Used for input time scaling.

        ### Outputs
            * ```torch.tensor``` log_likeli_loss_time_next_without_dummy
              shape: ```[1]```
              The sum of NLL loss L = -log \\frac{\\partial \\Lambda^*(m, t)}{\\partial t} + \\Lambda^*(m, t) at each happened mark.
            * ```torch.tensor``` loss_survival
              shape: ```[1]```
              The sum of the integration \\Lambda^*(m, t) from the last observed mark to the end time T.
            * ```torch.tensor``` marker_loss_time_next_without_dummy
              shape: ```[1]```
              The sum of the mark loss: L = -log \\frac{\\lambda^*(m, t)}{\\sum_{n \\in M}{\\lambda^*(n, t)}} where m is the mark of the real mark.
            * ```float``` mae
              The average error between predicted time and real time.
            * ```float``` f1
              The prediction accuracy of predicted marks.
            * ```int``` the_number_of_marks
              The number of legit marks.
        """
        time_history, time_next = self.divide_history_and_next(time)  # [batch_size, seq_len] * 2
        marks_history, marks_next = self.divide_history_and_next(marks)  # [batch_size, seq_len] * 2
        mask_history, mask_next = self.divide_history_and_next(mask)  # [batch_size, seq_len]

        mask_next_without_dummy = self.remove_dummy_events_from_mask(mask_next)  # [batch_size, seq_len]
        mark_next_without_dummy = (mask_next_without_dummy * marks_next).long()
        # [batch_size, seq_len]
        the_number_of_marks = mask_next_without_dummy.sum().item()

        pred_time, mark_dist = self.next_event_prediction_time_mark(
            time_history,
            time_next,
            marks_history,
            mean=mean,
            std=std,
            sample_rate=self.sample_rate,
            mae_step=self.mae_step,
        )

        mae = torch.abs(pred_time - time_next) * mask_next  # [batch_size, seq_len]
        mae = mae.sum().item() / the_number_of_marks

        pred_mark = mark_dist.argmax(dim=-1)  # [batch_size, seq_len]
        results = evaluate_on_one_batch(pred_mark, marks_next, mask_next, ["acc", "macro-f1", "micro-f1"])
        acc = results["acc"].mean()
        macro_f1 = results["macro-f1"].mean()
        micro_f1 = results["micro-f1"].mean()

        integral_all_marks_time_next, intensity_all_marks_time_next = self.model(
            time_history, time_next, marks_history
        )  # 2 * [batch_size, seq_len, num_marks]

        # loss_kl = self.likelihood_loss(marks_history, time_history, mark_next_without_dummy, time_next, mask_next_without_dummy, mean, std)

        # NLL loss and mark loss at time_next
        # L = \\sum_{i}{\\lambda^_k*(t_i)} + \\int_{t_0}^{t_n}{\\sum_{k}{\\lambda^*_k(\\tau)}d\\tau}
        log_likeli_loss_time_next_without_dummy, mark_loss_time_next_without_dummy = self.loss_function(
            integral_all_marks=integral_all_marks_time_next,
            intensity_all_marks=intensity_all_marks_time_next,
            marks_next=mark_next_without_dummy,
            mask_next=mask_next_without_dummy,
        )
        # Survival probability: \\int_{t_N}^{T}{\\sum_{k}\\lambda_k^(\\tau)d\\tau}
        dummy_mark_index = mask_next.sum(dim=-1) - 1  # [batch_size]
        integral_survival = integral_all_marks_time_next.sum(dim=-1).gather(
            index=dummy_mark_index.unsqueeze(dim=-1), dim=-1
        )
        # [batch_size, 1]
        loss_survival = integral_survival.mean()

        return (
            log_likeli_loss_time_next_without_dummy / the_number_of_marks,
            loss_survival,
            mark_loss_time_next_without_dummy / the_number_of_marks,
            mae,
            acc,
            macro_f1,
            micro_f1,
            the_number_of_marks,
        )

    def loss_function(self, integral_all_marks, intensity_all_marks, marks_next, mask_next):
        """
        This function computes the NLL loss at each legit mark in marks_next.

        ### Args
            * ```torch.tensor``` intensity_all_marks
              shape: ```[batch_size, seq_len, num_marks]```
              intensity values at t_i.
            * ```torch.tensor``` integral_all_marks
              shape: ```[batch_size, seq_len, num_marks]```
              intensity integral from t_{i - 1} to t_{i} (t_0 = 0).
            * ```torch.tensor``` marks_next
              shape: ```[batch_size, seq_len]```
              The mark of the marks that we need to predict.
            * ```torch.tensor``` mask_next
              shape: ```[batch_size, seq_len]```
              Needed mask to mask out unneeded loss values.

        ### Outputs
            * ```torch.tensor``` mtpp_loss,
              shape: ```[1]```
              The sum of NLL loss on all mark.
            * ```torch.tensor``` marks_loss
              shape: ```[1]```
              The sum of the mark loss: L = -log \\frac{\\lambda^*(m, t)}{\\sum_{n \\in M}{\\lambda^*(n, t)}} where m is the mark of the real mark.
        """
        type_mask = F.one_hot(marks_next, num_classes=self.num_marks)  # [batch_size, seq_len, num_marks]

        # MTPP loss function
        selected_intensity = (intensity_all_marks * type_mask).sum(dim=-1)  # [batch_size, seq_len]
        log_intensity = torch.log(selected_intensity + self.epsilon)  # [batch_size, seq_len]
        nll = -log_intensity + integral_all_marks.sum(dim=-1)  # [batch_size, seq_len]

        mtpp_loss = torch.sum(nll * mask_next)

        # Event loss function. Only for evaluation, do not use this loss as a part of the training loss.
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
        mean: float,
        std: float,
        sample_rate: int,
        mae_step: int,
        mask_history: torch.Tensor = None,
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
            mae_step (int): how many samples for one mark are generated in one shot
            evaluation (bool): If true, we are in the evaluation mode, the mark distribution is at the time_next.
                               If false, we are in the prediction mode, the mark distribution is at the pred_time

        Returns:
            tuple[torch.Tensor, torch.Tensor]: predicted time and mark distribution
        """
        pred_time = self.sample_time(
            sampling_approach="its",
            task="tm",
            time_history=time_history,
            marks_history=marks_history,
            number_of_total_samples=sample_rate,
            step=mae_step,
            mean=mean,
            std=std,
        )  # [sample_rate, batch_size, seq_len]
        if not get_time_sample:
            pred_time = pred_time.mean(dim=0)  # [batch_size, seq_len]

        if evaluation:
            _, intensity_all_marks = self.model(time_history, time_next, marks_history)
        # [batch_size, seq_len, num_marks]
        else:
            _, intensity_all_marks = self.model(time_history, pred_time, marks_history)
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
        mean: float,
        std: float,
        sample_rate: int,
        mae_e_step: int,
        mask_history: torch.Tensor = None,
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
        inf_val, resolution_inf, resolution_between_marks = decide_resolution_inf_and_resolution_between_events(
            time_history, memory_ceiling, self.num_marks, mean, std
        )

        mark_distribution = self.get_pm_next_event(time_history, marks_history, inf_val, resolution_inf, mean, std)
        # [batch_size, seq_len, num_marks]

        tau_sampled_all_mark = self.sample_time(
            sampling_approach="its",
            task="mt",
            marks_history=marks_history,
            time_history=time_history,
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
        inf_val: int,
        resolution_inf: int,
        mean,
        std,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the next mark prediction from the SAHP by MAE and F1.
        This function first predict the time of the next mark then the mark of the next mark given the TRUE time.
        This can ensure the mark accuracy is only influenced by the model, not by the predicted time.

        Args:
            self (Self): the SAHP model.
            time_history (torch.Tensor): the time of historical marks.
            marks_history (torch.Tensor): the mark of historical marks.
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
        ) = self.model.integral_intensity_time_next_2d(time_history, time_next_inf, marks_history, resolution_inf)
        # 2 * [batch_size, seq_len, resolution, num_marks]
        expanded_probability_inf = (
            torch.exp(-expanded_integral_all_marks_to_inf.sum(dim=-1, keepdim=True))
            * expanded_intensity_all_marks_to_inf
        )
        # [batch_size, seq_len, resolution, num_marks]
        return approximate_integration(expanded_probability_inf, timestamp, dim=-2, only_integral=True)
        # [batch_size, seq_len, num_marks]

    @torch.inference_mode()
    def probability_time_next_2d(
        self, time_history, time_next, marks_history, mask_history, integration_sample_rate, mean, std
    ):
        expand_integral, expand_intensity, timestamp = self.model.integral_intensity_time_next_2d(
            time_history,
            time_next,
            marks_history,
            integration_sample_rate,
        )
        return expand_intensity * torch.exp(-expand_integral.sum(dim=-1, keepdim=True)), timestamp

    def extract_plot_data(self, minibatch):
        """
        This function extracts input_time, input_marks, input_intensity, mask, mean, and std from the minibatch.

        ### Args
            * ```list``` minibatch
              shape: ```[[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]```
              data structure: [[input_time, input_marks, score, mask], (mean, std)]
              data type: [```torch.tensor```, ```torch.tensor```, ```torch.tensor```, ```torch.tensor```, (```float```, ```float```)]
              The input minibatch.

        ### Outputs
            * ```torch.tensor``` input_time
              shape: ```[batch_size, seq_len + 1]```
              Raw mark timestamp sequence.
            * ```torch.tensor``` input_marks
              shape: ```[batch_size, seq_len + 1]```
              Raw mark marks sequence.
            * ```torch.tensor``` mask
              shape: ```[batch_size, seq_len + 1]```
              Raw mask sequence.
            * ```int``` mean
              The mean of all $ t_i - t_{i - 1} $ in the entire dataset. Dataloader is responsible to provide this value if needed.
            * ```int``` std
              The mean of all $ t_i - t_{i - 1} $ in the entire dataset. Dataloader is responsible to provide this value if needed.
        """
        input_time, input_marks, _, mask, input_intensity = minibatch[0]
        mean, std = minibatch[1]

        return input_time, input_marks, input_intensity, mask, mean, std

    @torch.inference_mode()
    def figure_intensity(self, input_data, opt):
        """
        Function prober, used by evaluator to draw plots of the intensity function.

        You should declare the following arguments in your config file:
        1. ```int``` resolution: The number of interpolated points in a time interval between two adjoint marks for integration estimation.
                                 The number of interpolated points counts the start and end point of the interval.

        ### Args
            * ```list``` input_data
              shape: ```[[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]```
              data structure: [[input_time, input_marks, score, mask], (mean, std)]
              data type: [```torch.tensor```, ```torch.tensor```, ```torch.tensor```, ```torch.tensor```, (```float```, ```float```)]
              The input minibatch.
            * ```namespace``` opt
              plot and model configs
        """
        argument_check(opt, **{"resolution": int})

        input_time, input_marks, input_intensity, mask, mean, std = self.extract_plot_data(input_data)

        time_history, time_next = self.divide_history_and_next(input_time)  # [batch_size, seq_len]
        marks_history, marks_next = self.divide_history_and_next(input_marks)
        # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)  # [batch_size, seq_len]

        expand_integral, expand_intensity, timestamp = self.model.integral_intensity_time_next_2d(
            time_history, time_next, marks_history, opt.resolution
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
    def figure_integral(self, input_data, opt):
        """
        Function prober, used by evaluator to draw plots of the integral of the intensity function.

        You should declare the following arguments in your config file:
        1. ```int``` resolution: The number of interpolated points in a time interval between two adjoint marks for integration estimation.
                                 The number of interpolated points counts the start and end point of the interval.

        ### Args
            * ```list``` input_data
              shape: ```[[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]```
              data structure: [[input_time, input_marks, score, mask], (mean, std)]
              data type: [```torch.tensor```, ```torch.tensor```, ```torch.tensor```, ```torch.tensor```, (```float```, ```float```)]
              The input minibatch.
            * ```namespace``` opt
              plot and model configs
        """
        argument_check(opt, **{"resolution": int})

        input_time, input_marks, input_intensity, mask, mean, std = self.extract_plot_data(input_data)

        time_history, time_next = self.divide_history_and_next(input_time)  # [batch_size, seq_len]
        marks_history, marks_next = self.divide_history_and_next(input_marks)
        # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)  # [batch_size, seq_len]

        expand_integral, expand_intensity, timestamp = self.model.integral_intensity_time_next_2d(
            time_history, time_next, marks_history, opt.resolution
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
    def figure_probability(self, input_data, opt):
        """
        Function prober, used by evaluator to draw plots of the probability distribution.

        You should declare the following arguments in your config file:
        1. ```int``` resolution: The number of interpolated points in a time interval between two adjoint marks for integration estimation.
                                 The number of interpolated points counts the start and end point of the interval.

        ### Args
            * ```list``` input_data
              shape: ```[[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]```
              data structure: [[input_time, input_marks, score, mask], (mean, std)]
              data type: [```torch.tensor```, ```torch.tensor```, ```torch.tensor```, ```torch.tensor```, (```float```, ```float```)]
              The input minibatch.
            * ```namespace``` opt
              plot and model configs
        """
        argument_check(opt, **{"resolution": int})

        input_time, input_marks, input_intensity, mask, mean, std = self.extract_plot_data(input_data)

        time_history, time_next = self.divide_history_and_next(input_time)  # [batch_size, seq_len]
        marks_history, marks_next = self.divide_history_and_next(input_marks)
        # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)  # [batch_size, seq_len]

        expand_integral, expand_intensity, timestamp = self.model.integral_intensity_time_next_2d(
            time_history, time_next, marks_history, opt.resolution
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
    def figure_debug(self, input_data, opt):
        """
        Function prober, used by evaluator to draw plots for deeper insight of intensity functions and other metrics.

        You should declare the following arguments in your config file:
        1. ```int``` resolution: The number of interpolated points in a time interval between two adjoint marks for integration estimation.
                                 The number of interpolated points counts the start and end point of the interval.
        2. ```int``` sample_rate: how many time samples from the time distribution are needed.
        3. ```int``` mae_step: This parameter controls how many samples are generated in one shot when sampling from p(t).
        4. ```int``` mae_e_step: This parameter controls how many samples are generated in one shot when sampling from all p(t|m)s at the same time.

        ### Args
            * ```list``` input_data
              shape: ```[[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]```
              data structure: [[input_time, input_marks, score, mask], (mean, std)]
              data type: [```torch.tensor```, ```torch.tensor```, ```torch.tensor```, ```torch.tensor```, (```float```, ```float```)]
              The input minibatch.
            * ```namespace``` opt
              plot and model configs
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
            time_history, time_next, marks_history, mask_next, opt.resolution
        )

        pred_time, _ = self.next_event_prediction_time_mark(
            time_history,
            time_next,
            marks_history,
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
        top_k = evaluate_on_one_batch(mark_dist, marks_next, mask_next, "top_k", dim_input=-2)
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
        Paired with the od_genetic dataloader.
        """
        forward_complete_data, backward_complete_data, padded_obs_data, padded_backward_obs_mark_seq, (mean, std) = (
            input_data
        )

        roc_result = []
        for (
            obs_time_for_one_seq,
            obs_marks_for_one_seq,
            obs_mask_for_one_seq,
            missing_mask_for_one_seq,
            _,
        ) in padded_obs_data:
            obs_time_history_for_one_seq, obs_time_next_for_one_seq = self.divide_history_and_next(obs_time_for_one_seq)
            # [batch_size, seq_len] * 2
            obs_marks_history_for_one_seq, obs_marks_next_for_one_seq = self.divide_history_and_next(
                obs_marks_for_one_seq
            )
            # [batch_size, seq_len] * 2
            obs_mask_history_for_one_seq, obs_mask_next_for_one_seq = self.divide_history_and_next(obs_mask_for_one_seq)
            # [batch_size, seq_len]

            missing_mask_for_one_seq = self.convert_missing_mask_to_gap_mask(missing_mask_for_one_seq)
            # [num_samples, ...]
            integral_all_marks, intensity_all_marks = self.model(
                obs_time_history_for_one_seq.float(), obs_time_next_for_one_seq.float(), obs_marks_history_for_one_seq
            )
            # [num_samples, seq_len, num_marks]

            integral_sum = integral_all_marks.sum(dim=-1)  # [num_samples, seq_len]
            intensity_sum = intensity_all_marks.sum(dim=-1)  # [num_samples, seq_len]

            all_roauc_area = []
            for integral_sum_per_seq_per_sample, missing_mask_for_one_seq_per_sample in zip(
                integral_sum, missing_mask_for_one_seq
            ):
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
        (time_seq, marks, commission, mask), (mean, std) = input_data

        time_history, time_next = self.divide_history_and_next(time_seq)  # [batch_size, seq_len]
        marks_history, marks_next = self.divide_history_and_next(marks)  # [batch_size, seq_len]
        _, commission_next = self.divide_history_and_next(commission)  # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)  # [batch_size, seq_len]
        time_history = time_history.float()
        time_next = time_next.float()

        _, intensity_all_marks = self.model(time_history, time_next, marks_history)
        # 2 * [batch_size, seq_len, num_marks]

        intensity_sum_from_tl_to_time_next = intensity_all_marks.sum(dim=-1)
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
              data structure: [[input_time, input_marks, score, mask], (mean, std)]
              data type: [```torch.tensor```, ```torch.tensor```, ```torch.tensor```, ```torch.tensor```, (```float```, ```float```)]
              The input minibatch.
            * ```torch.device``` device
              where we train the model.

        ### Outputs:
            * ```float``` time_loss_without_dummy
              The average NLL loss without dummy marks, specifically the start and the end mark.
            * ```float``` fact
              The average NLL loss with the real distribution. This value only makes sense for synthetic datasets.
            * ```float``` marks_loss
              The average cross-entropy loss of the mark prediction distribution. The value is only for performance measure porpose.
              The training loss does not and should not include this value.
        """
        self.train()

        # Maybe need another function to extract data from minibatches.
        # For now, we don't acquire any prediction loss to assist the model training.
        (time, marks, score, mask), (mean, std) = (
            minibatch  # 3 * [batch_size, seq_len + 1, 1] & [batch_size, seq_len, 1]
        )
        loss, time_loss_without_dummy, marks_loss, the_number_of_marks = self.forward(
            "train", time, marks, mask, mean=mean, std=std
        )

        loss.backward()

        time_loss_without_dummy = time_loss_without_dummy.item()
        marks_loss = marks_loss.item()
        fact = score.sum().item() / the_number_of_marks

        return time_loss_without_dummy, fact, marks_loss

    def evaluation_step(self, minibatch):
        """
        This function unpacks the minibatch, calls the evaluation_procedure() to calculate the metrics.

        ### Args
            * ```torch.nn.Module``` model
              The MTPP model that we train.
            * ```list``` minibatch
              shape: ```[[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]```
              data structure: [[input_time, input_marks, score, mask], (mean, std)]
              data type: [```torch.tensor```, ```torch.tensor```, ```torch.tensor```, ```torch.tensor```, (```float```, ```float```)]
              The input minibatch.
            * ```torch.device``` device
              where we train the model.

        ### Outputs:
            * ```float``` time_loss
              The average NLL loss without dummy marks, specifically the start and the end mark.
            * ```float``` loss_survival
              The average NLL loss of the end mark, which is the integral of the intensity function from the last occurred mark to the end time.
            * ```float``` fact
              The average NLL loss with the real distribution. This value only makes sense for synthetic datasets.
            * ```float``` marks_loss
              The average cross-entropy loss of the mark prediction distribution. The value is only for performance measure porpose.
            * ```float``` mae
              The average error between predicted time and real time.
            * ```float``` f1
              The prediction accuracy of predicted marks.
        """
        self.eval()

        (time, marks, score, mask), (mean, std) = (
            minibatch  # 3 * [batch_size, seq_len + 1, 1] & [batch_size, seq_len, 1]
        )
        time_loss, loss_survival, marks_loss, mae, acc, macro_f1, micro_f1, the_number_of_marks = self.forward(
            "evaluate", time, marks, mask, mean=mean, std=std
        )

        time_loss = time_loss.item()
        loss_survival = loss_survival.item()
        marks_loss = marks_loss.item()
        fact = score.sum().item() / the_number_of_marks

        return time_loss, loss_survival, fact, marks_loss, mae, acc, macro_f1, micro_f1

    def postprocess(self, input_data, procedure):
        """
        This function makes some modifications to the output of training_step() and evaluation_step().

        ### Args
            * ```list``` input
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
