from typing import Self

import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from src.toolbox.metrics import evaluate_on_one_batch
from src.toolbox.misc import (
    argument_check,
    compile_model,
    move_from_tensor_to_ndarray,
    pack_one_value_to_dict,
)
from src.tpp.tpp_models.model.basic_tpp_model import BasicModel
from src.tpp.tpp_models.model.utils import (
    BalancedSamplingFromDistributionMixin,
    GetWhichEventFirstMixin,
    NextEventPredictionMarkTimeMixin,
    NextEventPredictionTimeMarkMixin,
    SpearmanL1EvaluationMixin,
)

from .plot import generate_debug_figure, generate_probability_figure
from .sample import sample_mark_time, sample_time, sample_time_mark
from .submodel import IFN


class IFNModel(
    BasicModel,
    BalancedSamplingFromDistributionMixin,
    GetWhichEventFirstMixin,
    NextEventPredictionMarkTimeMixin,
    NextEventPredictionTimeMarkMixin,
    SpearmanL1EvaluationMixin,
):
    """
    IFN (Integration-free Neural Marked Temporal Point Process)
    """

    def __init__(
        self,
        training,
        d_history,
        d_intensity,
        dropout,
        history_module_layers,
        mlp_layers,
        opt,
        device,
        removes_tail,
        tanh_parameter,
        history_module="LSTM",
        survival_loss_during_training=True,
        epsilon=0.0,
        sample_rate=32,
        mae_step=32,
        mae_e_step=32,
    ):
        """
        This function creates a IFN model.

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
            * ```bool``` survival_loss_during_training
              When true, the training loss includes the integral between the last observed mark to the end time T. Most of time this argument should be true.
            * ```bool``` removes_tail
              In some cases, the calculated \\Gamma(m, t) failed to converge to a small number instead of 0 when t -> +\\infty.
              This trick somehow mitigates this issue by slightly offsetting the value of \\Gamma(m, t) so its value is 0 when t -> +\\infty.
            * ```float``` tanh_parameter
              Hyperparameter of scaled_tanh(). Please check scaled_tanh for detailed information.
        """
        super().__init__()
        self.device = device
        self.num_marks = opt.info_dict["num_marks"]
        self.start_time = opt.info_dict["t_0"]
        self.end_time = opt.info_dict["T"]
        self.epsilon = epsilon
        self.survival_loss_during_training = survival_loss_during_training
        self.sample_rate = sample_rate
        self.mae_step = mae_step
        self.mae_e_step = mae_e_step
        self.bisect_early_stop_threshold = 1e-4
        self.max_step = 50

        self.model = IFN(
            d_history=d_history,
            d_intensity=d_intensity,
            num_marks=self.num_marks,
            dropout=dropout,
            history_module=history_module,
            history_module_layers=history_module_layers,
            mlp_layers=mlp_layers,
            removes_tail=removes_tail,
            tanh_parameter=tanh_parameter,
            epsilon=epsilon,
            device=device,
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
        """Remove the dummy event by altering the mask.

        Args:
            self (Self): the SAHP model
            mask (torch.Tensor): the input mask tensor. shape: [batch, seq_len]

        Returns:
            torch.Tensor: The input mask tensor with the dummy mark at the end removed. shape: [batch, seq_len]
        """
        dummy_indices = mask.sum(dim=1, dtype=torch.long) - 1  # [batch_size]
        mask_without_dummy = mask.clone()  # [batch_size, seq_len]
        batch_indices = torch.arange(mask.size(0), device=mask.device)  # [batch_size]
        mask_without_dummy[batch_indices, dummy_indices] = 0  # [batch_size, seq_len]

        return mask_without_dummy

    def forward(self, task_name, *args, **kwargs):
        """
        The entrance of the IFN.

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
            "generate_hypro_dataset": self.generate_hypro_dataset,
            # Figure Drawing.
            "probability": self.figure_probability,
            "debug": self.figure_debug,
            # For CPPOD, should be used with the od_generic dataloader.
            # "cppod_evaluation": self.cppod_evaluation,
        }

        return task_mapper[task_name](*args, **kwargs)

    def train_procedure(self, input_time, input_marks, mask, mean, std):
        """
        IFN's forwardpropagation function for training.

        ### Args
            * ```torch.tensor``` input_time
              shape: ```[batch_size, seq_len + 1]```
              Time sequence for training.
            * ```torch.tensor``` input_marks
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
            * ```torch.tensor``` time_loss_without_dummy
              shape: ```[1]```
              The sum of NLL loss L = -log \\frac{\\partial \\Lambda^*(m, t)}{\\partial t} + \\Lambda^*(m, t) at each happened mark (the dummy mark at end time T excluded).
            * ```torch.tensor``` marks_loss
              shape: ```[1]```
              The sum of the mark loss: L = -log \\frac{\\lambda^*(m, t)}{\\sum_{n \\in M}{\\lambda^*(n, t)}} where m is the mark of the real mark.
            * ```int``` the_number_of_marks
              The number of legit marks.
        """
        time_history, time_next = self.divide_history_and_next(input_time)  # 2 * [batch_size, seq_len]
        marks_history, marks_next = self.divide_history_and_next(input_marks)
        # 2 * [batch_size, seq_len]
        _, mask_next = self.divide_history_and_next(mask)  # [batch_size, seq_len]

        # \\int_{t}^{+\\inf}{p(m, \\tau|\\mathcal{H})d\\tau}
        probability_integral_from_t_to_infinite, probability_for_each_mark = self.model(
            time_history, time_next, marks_history, mean=mean, std=std, training=True
        )
        # [batch_size, seq_len, num_marks] * 2

        # Remove the probability of the dummy mark by mask.
        mask_next_without_dummy = self.remove_dummy_events_from_mask(mask_next)  # [batch_size, seq_len]
        marks_next_without_dummy = marks_next * mask_next_without_dummy  # [batch_size, seq_len]
        the_number_of_marks = mask_next_without_dummy.sum().item()

        mtpp_loss_without_dummy, marks_loss_without_dummy = self.loss_function(
            probability_for_each_mark, marks_next_without_dummy, mask_next_without_dummy
        )

        mtpp_loss_survival = 0
        if self.survival_loss_during_training:
            # Survival probability: \\int_{t_N}^{T}{\\sum_{k}\\lambda_k^(\\tau)d\\tau} = -\\log(1 - P(t)) = -log(IFIB-C(t)).
            dummy_mark_index = mask_next.sum(dim=-1) - 1  # [batch_size]
            probability_survival = probability_integral_from_t_to_infinite.sum(dim=-1).gather(
                index=dummy_mark_index.unsqueeze(dim=-1), dim=-1
            )
            # [batch_size, 1]
            # The experiment result shows that the existence of probability_survival could significantly damage the performance on the synthetic dataset.
            # Given other models are not affected, it is highly possible that I calculate the wrong survival loss.
            # However, I have no idea why I am wrong and what the correct one should be.
            mtpp_loss_survival = -torch.log(probability_survival).sum()

        loss = mtpp_loss_without_dummy + mtpp_loss_survival

        # we need time_loss_without_dummy to compare our distribution against the ground truth.
        return (
            loss / the_number_of_marks,
            mtpp_loss_without_dummy / the_number_of_marks,
            marks_loss_without_dummy / the_number_of_marks,
            the_number_of_marks,
        )

    def evaluate_procedure(self, input_time, input_marks, mask, mean, std):
        """
        IFN's forwardpropagation function for evaluation.

        ### Args
            * ```torch.tensor``` input_time
              shape: ```[batch_size, seq_len + 1]```
              Time sequencalculatesce for training.
            * ```torch.tensor``` input_marks
              shape: ```[batch_size, seq_len + 1]```
              Event sequence for training.
            * ```torch.tensor``` mask
              shape: ```[batch_size,, seq_len + 1]```
              Mask sequence. Events whose corresponding mask is 0 are dummy marks.
            * ```float``` mean
            * ```float``` std
              Used for input time scaling.

        ### Outputs
            * ```torch.tensor``` time_loss_without_dummy
              shape: ```[1]```
              The sum of NLL loss L = -log \\frac{\\partial \\Lambda^*(m, t)}{\\partial t} + \\Lambda^*(m, t) at each happened mark.
            * ```torch.tensor``` time_loss_survival
              shape: ```[1]```
              The sum of the integration \\Lambda^*(m, t) from the last observed mark to the end time T.
            * ```torch.tensor``` marks_loss
              shape: ```[1]```
              The sum of the mark loss: L = -log \\frac{\\lambda^*(m, t)}{\\sum_{n \\in M}{\\lambda^*(n, t)}} where m is the mark of the real mark.
            * ```float``` mae
              The average error between predicted time and real time.
            * ```float``` f1
              The prediction accuracy of predicted marks.
            * ```int``` the_number_of_marks
              The number of legit marks.
        """
        time_history, time_next = self.divide_history_and_next(input_time)  # 2 * [batch_size, seq_len]
        marks_history, marks_next = self.divide_history_and_next(input_marks)
        # 2 * [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)  # 2 * [batch_size, seq_len]

        # Remove the probability of the dummy mark by mask.
        mask_next_without_dummy = self.remove_dummy_events_from_mask(mask_next)  # [batch_size, seq_len]
        marks_next_without_dummy = marks_next * mask_next_without_dummy  # [batch_size, seq_len]
        the_number_of_marks = mask_next_without_dummy.sum().item()

        pred_time, mark_distribution = self.next_event_prediction_time_mark(
            time_history=time_history,
            time_next=time_next,
            marks_history=marks_history,
            mean=mean,
            std=std,
            sample_rate=self.sample_rate,
            mae_step=self.mae_step,
        )  # 2 * [batch_size, seq_len]
        mae = torch.abs(pred_time - time_next) * the_number_of_marks  # [batch_size, seq_len]
        mae = mae.sum().item() / the_number_of_marks

        pred_mark = mark_distribution.argmax(dim=-1)  # [batch_size, seq_len]
        results = evaluate_on_one_batch(
            pred_mark,
            marks_next,
            mask_next_without_dummy,
            ["acc", "macro-f1", "micro-f1"],
            multiprocessing=True,
            num_workers=4,
        )
        acc = results["acc"].mean()
        macro_f1 = results["macro-f1"].mean()
        micro_f1 = results["micro-f1"].mean()

        probability_integral_from_time_next_to_infinite, probability_of_all_marks = self.model(
            time_history, time_next, marks_history, mean=mean, std=std
        )
        # [batch_size, seq_len, num_marks] * 2

        # Time loss: -log p(t) = \\sum_{i = 1}^{N}{\\lambda_{k}(t_i)} + \\int_{t_0}^{t_N}{\\sum_{k}\\lambda_k^(\\tau)d\\tau}
        mtpp_loss_without_dummy, mark_loss_without_dummy = self.loss_function(
            probability_all_marks=probability_of_all_marks,
            marks_next=marks_next_without_dummy,
            mask_next=mask_next_without_dummy,
        )
        # Survival probability: \\int_{t_N}^{T}{\\sum_{k}\\lambda_k^(\\tau)d\\tau} = -\\log(1 - P(t)) = -log(\\sum_{m}{IFIB-C(m, t)}).
        dummy_mark_index = mask_next.sum(dim=-1) - 1  # [batch_size]
        probability_survival = probability_integral_from_time_next_to_infinite.sum(dim=-1).gather(
            index=dummy_mark_index.unsqueeze(dim=-1), dim=-1
        )
        # [batch_size, 1]
        time_loss_survival = -torch.log(probability_survival).mean()

        return (
            mtpp_loss_without_dummy / the_number_of_marks,
            time_loss_survival,
            mark_loss_without_dummy / the_number_of_marks,
            mae,
            acc,
            macro_f1,
            micro_f1,
            the_number_of_marks,
        )

    def loss_function(
        self: Self,
        probability_all_marks: torch.Tensor,
        marks_next: torch.Tensor,
        mask_next: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """This function computes the NLL loss and mark loss at all true mark in marks_next.

        Args:
            self (Self): the IFN model
            integral_all_marks (torch.Tensor): intensity integral from t_{i - 1} to t_{i} (t_0 = 0).
            intensity_all_marks (torch.Tensor): intensity values at t_i.
            marks_next (torch.Tensor): the mark of the marks that we need to predict.
            mask_next (torch.Tensor): mask out unneeded loss values.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: the NLL at true marks, the cross entropy loss of the mark
        """
        type_mask = F.one_hot(marks_next, num_classes=self.num_marks)  # [batch_size, seq_len, num_marks]

        # MTPP loss function
        selected_probability = (probability_all_marks * type_mask).sum(dim=-1)  # [batch_size, seq_len]
        nll = -torch.log(selected_probability + self.epsilon)  # [batch_size, seq_len]

        mtpp_loss = torch.sum(nll * mask_next)

        # Event loss function. Only for evaluation, do NOT use this loss as a part of the training loss.
        marks_prediction_probability = torch.log(probability_all_marks + self.epsilon)
        # [batch_size, seq_len, num_marks]
        marks_prediction_probability = F.softmax(marks_prediction_probability, dim=-1)
        # [batch_size, seq_len, num_marks]
        reshaped_marks_prediction_probability = rearrange(marks_prediction_probability, "b s ne -> b ne s")
        # [batch_size, num_marks, seq_len]
        marks_loss = F.cross_entropy(
            input=reshaped_marks_prediction_probability,
            target=marks_next,
            reduction="none",
        )
        # [batch_size, seq_len]
        marks_loss = (marks_loss * mask_next).sum()

        return mtpp_loss, marks_loss

    sample_time = sample_time
    sample_time_mark = sample_time_mark
    sample_mark_time = sample_mark_time

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
            _, probablity_all_marks = self.model(time_history, time_next, marks_history, mean, std)
        # [batch_size, seq_len, num_marks]
        else:
            _, probablity_all_marks = self.model(time_history, pred_time, marks_history, mean, std)
            # [batch_size, seq_len, num_marks]

        mark_distribution = probablity_all_marks / (probablity_all_marks.sum(dim=-1, keepdim=True) + self.epsilon)
        # [batch_size, seq_len, num_marks]
        return pred_time, mark_distribution

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
        evaluation: bool = True,
        get_time_sample: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the next event prediction from the SAHP by MAE and F1.
        This function first predict the time of the next event then the mark of the next event given the TRUE time.
        This can ensure the mark accuracy is only influenced by the model, not by the predicted time.

        Args:
            self (Self): the SAHP model.
            time_history (torch.Tensor): the time of historical marks
            time_next (torch.Tensor): the time of the next true marks
            marks_history (torch.Tensor): the mark of historical marks
            marks_next (torch.Tensor): the mark of the next true marks
            mask_history (torch.Tensor): the mask of historical sequences, 1 meaning a true event, and 0 meaning a fake event.
            mask_next (torch.Tensor): the mask of the next true marks
            mean (float): the mean of all time intervals
            std (float): the standard variance of all time intervals
            sample_rate (int): how many samples are needed for one time prediction
            mae_e_step (int): how many samples for one event are generated in one shot
            evaluation (bool): If true, we are in the evaluation mode, the mark distribution is at the time_next.
                               If false, we are in the prediction mode, the mark distribution is at the pred_time

        Returns:
            tuple[torch.Tensor, torch.Tensor]: predicted time and mark distribution
        """
        mark_distribution = self.get_pm_next_event(time_history, marks_history, 0, 0, mean, std)
        # [batch_size, seq_len, num_marks]

        tau_sampled_all_mark = self.sample_time(
            sampling_approach="its",
            task="mt",
            time_history=time_history,
            marks_history=marks_history,
            p_m=mark_distribution,
            number_of_total_samples=sample_rate,
            step=mae_e_step,
            inf_val=1e6,
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

    def get_pm_next_event(
        self: Self,
        time_history: torch.Tensor,
        marks_history: torch.Tensor,
        inf_val: int,
        resolution_inf: int,
        mean,
        std,
        mask_history: torch.Tensor = None,
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
        time_zero = torch.zeros_like(time_history)  # [batch_size, seq_len]
        mark_distribution, _ = self.model(
            time_history, time_zero, marks_history, mean=mean, std=std
        )  # [batch_size, seq_len, num_marks]

        return mark_distribution

    def probability_time_next_2d(
        self, time_history, time_next, marks_history, mask_history, integration_sample_rate, mean, std
    ):
        return self.model.probability_time_next_2d(
            time_history, time_next, marks_history, integration_sample_rate, mean, std
        )

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
        _, mask_next = self.divide_history_and_next(mask)  # [batch_size, seq_len]

        expand_probability, timestamp = self.model.probability_time_next_2d(
            time_history, time_next, marks_history, opt.resolution, mean, std
        )
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

    def figure_debug(self, input_data, opt):
        """
        Function prober, used by evaluator to draw plots for deeper insight of intensity functions and other metrics.

        You should declare the following arguments in your config file:
        1. ```int``` resolution: The number of interpolated points in a time interval between two adjoint marks for integration estimation.
                                 The number of interpolated points counts the start and end point of the interval.
        2. ```int``` sample_rate: The number of interpolated points in a time interval between two adjoint marks for integration estimation.
                                  The number of interpolated points counts the start and end point of the interval.
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
        argument_check(opt, **{"resolution": int, "sample_rate": int, "mae_step": int, "mae_e_step": int})

        input_time, input_marks, input_intensity, mask, mean, std = self.extract_plot_data(input_data)

        time_history, time_next = self.divide_history_and_next(input_time)  # [batch_size, seq_len]
        marks_history, marks_next = self.divide_history_and_next(input_marks)
        # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)  # [batch_size, seq_len]

        # probed data.
        data, timestamp = self.model.model_probe_function(
            time_history, time_next, marks_history, mask_next, opt.resolution, mean, std
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

        # We show how porobability distribution goes on two sampled sequences, one following the mark-time routine, and
        # the other following the time-mark routine.
        time_history_for_sampling_mark_time, marks_history_for_sampling_mark_time, sampled_mask_mark_time = (
            self.sample_mark_time(
                None,
                None,
                mean,
                std,
                end_sampling_requirement="time_and_event_num",
                number_of_sampled_sequences=1,
                end_time=self.end_time - self.start_time,
                max_seq_len=250,
            )
        )
        # 3 * [number_of_sampled_sequences, length_of_sampled_sequences]

        sampled_time_history_mark_time, sampled_time_next_mark_time = self.divide_history_and_next(
            time_history_for_sampling_mark_time
        )
        # 2 * [batch_size, seq_len]
        sampled_marks_history_mark_time, sampled_marks_next_mark_time = self.divide_history_and_next(
            marks_history_for_sampling_mark_time
        )
        # 2 * [batch_size, seq_len]
        _, sampled_mask_next_mark_time = self.divide_history_and_next(sampled_mask_mark_time)
        # 2 * [batch_size, seq_len]

        sampled_data_mark_time, sampled_timestamp_mark_time = self.model.model_probe_function(
            sampled_time_history_mark_time,
            sampled_time_next_mark_time,
            sampled_marks_history_mark_time,
            sampled_mask_next_mark_time,
            opt.resolution,
            mean,
            std,
        )

        time_history_for_sampling_time_mark, marks_history_for_sampling_time_mark, sampled_mask_time_mark = (
            self.sample_time_mark(
                None,
                None,
                mean,
                std,
                end_sampling_requirement="time_and_event_num",
                number_of_sampled_sequences=1,
                end_time=self.end_time - self.start_time,
                max_seq_len=250,
            )
        )
        # 3 * [number_of_sampled_sequences, length_of_sampled_sequences]

        sampled_time_history_time_mark, sampled_time_next_time_mark = self.divide_history_and_next(
            time_history_for_sampling_time_mark
        )
        # 2 * [batch_size, seq_len]
        sampled_marks_history_time_mark, sampled_marks_next_time_mark = self.divide_history_and_next(
            marks_history_for_sampling_time_mark
        )
        # 2 * [batch_size, seq_len]
        _, sampled_mask_next_time_mark = self.divide_history_and_next(sampled_mask_time_mark)
        # 2 * [batch_size, seq_len]

        sampled_data_time_mark, sampled_timestamp_time_mark = self.model.model_probe_function(
            sampled_time_history_time_mark,
            sampled_time_next_time_mark,
            sampled_marks_history_time_mark,
            sampled_mask_next_time_mark,
            opt.resolution,
            mean,
            std,
        )

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
                # Show the mark sequence sampled from p(t) and p(m|t)
                "sampled_marks_next_mark_time": sampled_marks_next_mark_time,
                "sampled_time_next_mark_time": sampled_time_next_mark_time,
                "sampled_mask_next_mark_time": sampled_mask_next_mark_time,
                "sampled_timestamp_mark_time": sampled_timestamp_mark_time,
                "sampled_subprobability_mark_time": sampled_data_mark_time["expand_probability_for_each_mark"],
                # Show the mark sequence sampled from p(m) and p(t|m)
                "sampled_marks_next_time_mark": sampled_marks_next_time_mark,
                "sampled_time_next_time_mark": sampled_time_next_time_mark,
                "sampled_mask_next_time_mark": sampled_mask_next_time_mark,
                "sampled_timestamp_time_mark": sampled_timestamp_time_mark,
                "sampled_subprobability_time_mark": sampled_data_time_mark["expand_probability_for_each_mark"],
            }
        )

        generate_debug_figure(data, opt)

    def generate_hypro_dataset(self, input_data, opt):
        # CAUTION: Only works when batch_size = 1.

        input_time, input_marks, input_intensity, mask, mean, std = self.extract_plot_data(input_data)

        if mask.sum(dim=-1) <= opt.number_of_marks_hypro:
            """
            Sequence too short to perform HYPRO. Considering to make the number_of_marks_hypro lower to avoid this.
            """
            return None

        time_history_for_sampling = repeat(
            input_time[..., : -opt.number_of_marks_hypro], "() ... -> nns ...", nns=opt.number_of_negative_samples
        )
        # [number_of_negative_samples, seq_len - opt.number_of_marks_hypro]
        mark_history_for_sampling = repeat(
            input_marks[..., : -opt.number_of_marks_hypro], "() ... -> nns ...", nns=opt.number_of_negative_samples
        )
        # [number_of_negative_samples, seq_len - opt.number_of_marks_hypro]

        (
            tau_sampled,
            marks_sampled,
            _,
        ) = self.sample_mark_time(
            time_history_for_sampling,
            mark_history_for_sampling,
            mean,
            std,
            end_sampling_requirement="mark_num",
            max_seq_len=mask.sum(dim=-1),
        )
        # [number_of_negative_samples, seq_len]

        input_time, input_marks, tau_sampled, marks_sampled = move_from_tensor_to_ndarray(
            input_time, input_marks, tau_sampled, marks_sampled
        )

        return input_time, input_marks, tau_sampled, marks_sampled

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
              The average NLL loss without dummy marks, specifically the start and the end event.
            * ```float``` fact
              The average NLL loss with the real distribution. This value only makes sense for synthetic datasets.
            * ```float``` marks_loss
              The average cross-entropy loss of the event prediction distribution. The value is only for performance measure porpose.
              The training loss does not and should not include this value.
        """
        self.train()

        [time_seq, mark_seq, score, mask], (mean, std) = minibatch
        loss, time_loss_without_dummy, marks_loss, the_number_of_marks = self.forward(
            task_name="train", input_time=time_seq, input_marks=mark_seq, mask=mask, mean=mean, std=std
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
              The average NLL loss without dummy marks, specifically the start and the end event.
            * ```float``` loss_survival
              The average NLL loss of the end event, which is the integral of the intensity function from the last occurred event to the end time.
            * ```float``` fact
              The average NLL loss with the real distribution. This value only makes sense for synthetic datasets.
            * ```float``` marks_loss
              The average cross-entropy loss of the event prediction distribution. The value is only for performance measure porpose.
            * ```float``` mae
              The average error between predicted time and real time.
            * ```float``` f1
              The prediction accuracy of predicted marks.
        """
        self.eval()

        [time_seq, mark_seq, score, mask], (mean, std) = minibatch
        time_loss, loss_survival, marks_loss, mae, acc, macro_f1, micro_f1, the_number_of_marks = self.forward(
            task_name="evaluate", input_time=time_seq, input_marks=mark_seq, mask=mask, mean=mean, std=std
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
