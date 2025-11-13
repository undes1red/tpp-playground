import copy
from typing import Self

import torch
import torch.nn.functional as F
from einops import rearrange

from src.toolbox.algorithms import approximate_integration
from src.toolbox.metrics import evaluate_on_one_batch
from src.toolbox.misc import (
    argument_check,
    break_batched_inputs_into_seqs,
    check_tensor,
    compile_model,
    move_from_tensor_to_ndarray,
    pack_one_value_to_dict,
)
from src.TPP.model.basic_tpp_model import BasicModel, memory_ceiling
from src.TPP.model.thp.plot import (
    generate_debug_figure,
    generate_integral_figure,
    generate_intensity_figure,
    generate_probability_figure,
)
from src.TPP.model.thp.sample import sample_time
from src.TPP.model.thp.submodel import THP
from src.TPP.model.utils import (
    decide_resolution_inf_and_resolution_between_mark,
)
from src.TPP.resources import expand_true_probability


class THPWrapper(BasicModel):
    """
    Transformer Hawkes Process (THP), proposed by Zuo et al. at ICML 2020.
    """

    def __init__(
        self,
        opt,
        device,
        d_input=64,
        d_rnn=64,
        d_hidden=256,
        n_layers=3,
        n_head=3,
        d_qk=64,
        d_v=64,
        dropout=0.1,
        sample_rate=32,
        integration_sample_rate=100,
        epsilon=1e-20,
        history_time_offset=1.0,
        mae_step=32,
        mae_e_step=32,
        survival_loss_during_training=True,
    ):
        """
        This function creates a THP model.

        ### Args
            * ```int``` d_input
            The dimension of the Transformer input tensor.
            * ```int``` d_hidden
              The dimension of the FFN module in the Transformer.
            * ```int``` n_layers
              The number of self attention + FFN layers in the Transformer.
            * ```int``` n_head
              The number of head in self attention.
            * ```int``` d_qk
              The dimension of matrices Q and K.
            * ```int``` d_v
              The dimension of metrix V.
            * ```float``` dropout
              Dropout rate for the history encoder.
            * ```int``` d_rnn
              The dimension of RNN's hidden state.
            * ```float``` history_time_offset
              THP scales the input time by dividing it with the time interval from start to the latest event in history.
              This can cause issues when there is no event in history-the input time will be divided by 0.
              So, we add this offset to the time interval to avoid this divided-by-0 case.
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
              The number of interpolated points in a time interval between two adjoint mark for integration estimation.
              The number of interpolated points counts the start and end point of the interval.
            * ```bool``` survival_loss_during_training
              When true, the training loss includes the integral between the last observed event to the end time T. Most of time this argument should be true.
        """
        super().__init__()
        self.device = device
        self.compile_or_not = opt.compile
        self.num_mark = opt.info_dict["num_mark"]
        self.start_time = opt.info_dict["t_0"]
        self.end_time = opt.info_dict["T"]
        self.epsilon = epsilon
        self.history_time_offset = history_time_offset
        self.survival_loss_during_training = survival_loss_during_training
        self.integration_sample_rate = integration_sample_rate
        self.sample_rate = sample_rate
        self.mae_step = mae_step
        self.mae_e_step = mae_e_step
        self.bisect_early_stop_threshold = 1e-8
        self.max_step = 50

        self.model = THP(
            num_mark=self.num_mark,
            d_input=d_input,
            d_rnn=d_rnn,
            d_hidden=d_hidden,
            n_layers=n_layers,
            n_head=n_head,
            d_qk=d_qk,
            d_v=d_v,
            dropout=dropout,
            integration_sample_rate=integration_sample_rate,
            device=device,
            history_time_offset=history_time_offset,
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

    def remove_dummy_event_from_mask(self, mask):
        """
        Remove the dummy event by altering the mask.

        ### Args
            * ```torch.tensor``` mask
              shape: [batch_size, seq_len]
              The input mask tensor.

        ### Outputs
            * ```torch.tensor``` mask_without_dummy
              shape: [batch_size, seq_len]
              The output mask tensor with the last unmask event in each sequence removed.
        """
        mask_without_dummy = torch.zeros_like(mask)  # [batch_size, seq_len - 1]
        for idx, mask_per_seq in enumerate(mask):
            dummy_index = mask_per_seq.sum() - 1
            mask_without_dummy_per_seq = copy.deepcopy(mask_per_seq.detach())
            mask_without_dummy_per_seq[dummy_index] = 0
            mask_without_dummy[idx] = mask_without_dummy_per_seq

        return mask_without_dummy

    def forward(self, task_name, *args, **kwargs):
        """
        The entrance of the THP.

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
            "which_event_occurs_first": self.get_which_event_first,
            "balanced_sampling_from_distribution": self.balanced_sampling_from_distribution,
            # Figure Drawing.
            "intensity": self.figure_intensity,
            "integral": self.figure_integral,
            "probability": self.figure_probability,
            "debug": self.figure_debug,
            # For CPPOD, should be used with the od_generic dataloader.
            # "cppod_evaluation": self.cppod_evaluation,
            # "cppod_commission_evaluation": self.cppod_commission_evaluation,
        }

        return task_mapper[task_name](*args, **kwargs)

    def train_procedure(self, time, mark, mask, mean, std):
        """
        THP's forwardpropagation function for training.

        ### Args
            * ```torch.tensor``` time
              shape: ```[batch_size, seq_len + 1]```
              Time sequence for training.
            * ```torch.tensor``` mark
              shape: ```[batch_size, seq_len + 1]```
              Event sequence for training.
            * ```torch.tensor``` mask
              shape: ```[batch_size,, seq_len + 1]```
              Mask sequence. Events whose corresponding mask is 0 are dummy events.
            * ```float``` mean
            * ```float``` std
              Used for input time scaling.

        ### Outputs
            * ```torch.tensor``` loss
              shape: ```[1]```
              The sum of NLL loss L = -log \\frac{\\partial \\Lambda^*(m, t)}{\\partial t} + \\Lambda^*(m, t) at each happened event (the dummy event at end time T included).
            * ```torch.tensor``` neg_log_likeli_loss_without_dummy
              shape: ```[1]```
              The sum of NLL loss L = -log \\frac{\\partial \\Lambda^*(m, t)}{\\partial t} + \\Lambda^*(m, t) at each happened event (the dummy event at end time T excluded).
            * ```torch.tensor``` marker_loss_without_dummy
              shape: ```[1]```
              The sum of the event loss: L = -log \\frac{\\lambda^*(m, t)}{\\sum_{n \\in M}{\\lambda^*(n, t)}} where m is the mark of the real event.
            * ```int``` the_number_of_mark
              The number of legit mark.
        """
        time_history, time_next = self.divide_history_and_next(time)  # [batch_size, seq_len] * 2
        mark_history, mark_next = self.divide_history_and_next(mark)  # [batch_size, seq_len] * 2
        mask_history, mask_next = self.divide_history_and_next(mask)  # [batch_size, seq_len] * 2

        mask_next_without_dummy = self.remove_dummy_event_from_mask(mask_next)  # [batch_size, seq_len]
        mark_next_without_dummy = mark_next * mask_next_without_dummy  # [batch_size, seq_len]
        the_number_of_mark = mask_next_without_dummy.sum().item()

        integral_all_mark, intensity_all_mark = self.model(time_history, time_next, mark_history, mask_history)
        # [batch_size, seq_len, num_mark]
        # L = \\sum_{i}{\\lambda^_k*(t_i)} + \\int_{t_0}^{t_n}{\\sum_{k}{\\lambda^*_k(\\tau)}d\\tau}
        neg_log_likeli_loss_without_dummy, marker_loss_without_dummy = self.loss_function(
            intensity_all_mark=intensity_all_mark,
            integral_all_mark=integral_all_mark,
            mark_next=mark_next_without_dummy,
            mask_next=mask_next_without_dummy,
        )

        loss_survival = 0
        if self.survival_loss_during_training:
            # survival_loss = \\int_{t_n}^{T}{\\sum_{k}{\\lambda^*_k(\\tau)}d\\tau}
            dummy_event_index = mask_next.sum(dim=-1) - 1  # [batch_size]
            integral_survival = integral_all_mark.sum(dim=-1).gather(
                index=dummy_event_index.unsqueeze(dim=-1), dim=-1
            )
            # [batch_size, 1]
            loss_survival = integral_survival.sum()

        loss = neg_log_likeli_loss_without_dummy + loss_survival

        return (
            loss / the_number_of_mark,
            neg_log_likeli_loss_without_dummy / the_number_of_mark,
            marker_loss_without_dummy / the_number_of_mark,
            the_number_of_mark,
        )

    @torch.inference_mode()
    def evaluate_procedure(self, time, mark, mask, mean, std):
        """
        THP's forwardpropagation function for evaluation.

        ### Args
            * ```torch.tensor``` time
              shape: ```[batch_size, seq_len + 1]```
              Time sequencalculatesce for training.
            * ```torch.tensor``` mark
              shape: ```[batch_size, seq_len + 1]```
              Event sequence for training.
            * ```torch.tensor``` mask
              shape: ```[batch_size,, seq_len + 1]```
              Mask sequence. Events whose corresponding mask is 0 are dummy events.
            * ```float``` mean
            * ```float``` std
              Used for input time scaling.

        ### Outputs
            * ```torch.tensor``` log_likeli_loss_time_next_without_dummy
              shape: ```[1]```
              The sum of NLL loss L = -log \\frac{\\partial \\Lambda^*(m, t)}{\\partial t} + \\Lambda^*(m, t) at each happened event.
            * ```torch.tensor``` loss_survival
              shape: ```[1]```
              The sum of the integration \\Lambda^*(m, t) from the last observed event to the end time T.
            * ```torch.tensor``` marker_loss_time_next_without_dummy
              shape: ```[1]```
              The sum of the event loss: L = -log \\frac{\\lambda^*(m, t)}{\\sum_{n \\in M}{\\lambda^*(n, t)}} where m is the mark of the real event.
            * ```float``` mae
              The average error between predicted time and real time.
            * ```float``` f1
              The prediction accuracy of predicted marks.
            * ```int``` the_number_of_mark
              The number of legit mark.
        """
        time_history, time_next = self.divide_history_and_next(time)  # [batch_size, seq_len] * 2
        mark_history, mark_next = self.divide_history_and_next(mark)  # [batch_size, seq_len] * 2
        mask_history, mask_next = self.divide_history_and_next(mask)  # [batch_size, seq_len]

        mask_next_without_dummy = self.remove_dummy_event_from_mask(mask_next)  # [batch_size, seq_len]
        mark_next_without_dummy = mark_next * mask_next_without_dummy  # [batch_size, seq_len]
        the_number_of_mark = mask_next_without_dummy.sum().item()

        pred_time, mark_dist = self.next_event_prediction_time_mark(
            time_history,
            time_next,
            mark_history,
            mask_history,
            mean=mean,
            std=std,
            sample_rate=self.sample_rate,
            mae_step=self.mae_step,
        )

        mae = torch.abs(pred_time - time_next) * mask_next  # [batch_size, seq_len]
        mae = mae.sum().item() / the_number_of_mark

        pred_mark = mark_dist.argmax(dim=-1)  # [batch_size, seq_len]
        results = evaluate_on_one_batch(pred_mark, mark_next, mask_next, ["acc", "macro-f1", "micro-f1"])
        acc = results["acc"].mean()
        macro_f1 = results["macro-f1"].mean()
        micro_f1 = results["micro-f1"].mean()

        integral_all_mark_time_next, intensity_all_mark_time_next = self.model(
            time_history, time_next, mark_history, mask_history
        )
        # 2 * [batch_size, seq_len, num_mark]

        # L = \\sum_{i}{\\lambda^_k*(t_i)} + \\int_{t_0}^{t_n}{\\sum_{k}{\\lambda^*_k(\\tau)}d\\tau}
        log_likeli_loss_time_next_without_dummy, marker_loss_time_next_without_dummy = self.loss_function(
            intensity_all_mark=intensity_all_mark_time_next,
            integral_all_mark=integral_all_mark_time_next,
            mark_next=mark_next_without_dummy,
            mask_next=mask_next_without_dummy,
        )

        # survival_loss = \\int_{t_n}^{T}{\\sum_{k}{\\lambda^*_k(\\tau)}d\\tau}
        dummy_event_index = mask_next.sum(dim=-1) - 1  # [batch_size]
        integral_survival = integral_all_mark_time_next.sum(dim=-1).gather(
            index=dummy_event_index.unsqueeze(dim=-1), dim=-1
        )
        # [batch_size, 1]
        loss_survival = integral_survival.sum()

        return (
            log_likeli_loss_time_next_without_dummy / the_number_of_mark,
            loss_survival,
            marker_loss_time_next_without_dummy / the_number_of_mark,
            mae,
            acc,
            macro_f1,
            micro_f1,
            the_number_of_mark,
        )

    def loss_function(self, intensity_all_mark, integral_all_mark, mark_next, mask_next):
        """
        This function computes the NLL loss at each legit event in mark_next.

        ### Args
            * ```torch.tensor``` intensity_all_mark
              shape: ```[batch_size, seq_len, num_mark]```
              intensity values at t_i.
            * ```torch.tensor``` integral_all_mark
              shape: ```[batch_size, seq_len, num_mark]```
              intensity integral from t_{i - 1} to t_{i} (t_0 = 0).
            * ```torch.tensor``` mark_next
              shape: ```[batch_size, seq_len]```
              The mark of the mark that we need to predict.
            * ```torch.tensor``` mask_next
              shape: ```[batch_size, seq_len]```
              Needed mask to mask out unneeded loss values.

        ### Outputs
            * ```torch.tensor``` mtpp_nll_loss,
              shape: ```[1]```
              The sum of NLL loss on all event.
            * ```torch.tensor``` mark_loss
              shape: ```[1]```
              The sum of the event loss: L = -log \\frac{\\lambda^*(m, t)}{\\sum_{n \\in M}{\\lambda^*(n, t)}} where m is the mark of the real event.
        """
        type_mask = F.one_hot(mark_next.long(), num_classes=self.num_mark)

        # MTPP loss function
        intensity = torch.sum(intensity_all_mark * type_mask, dim=-1)  # [batch_size, seq_len]
        log_intensity = torch.log(intensity + self.epsilon) * mask_next  # [batch_size, seq_len]
        intensity_integral = integral_all_mark.sum(dim=-1)  # [batch_size, seq_len]
        ll = -log_intensity + intensity_integral  # [batch_size, seq_len]
        mtpp_nll_loss = torch.sum(ll)

        # Event loss function. Only for evaluation, do not use this loss as a part of the training loss.
        mark_prediction_probability = torch.log(intensity_all_mark + self.epsilon)
        # [batch_size, seq_len, num_mark]
        mark_prediction_probability = F.softmax(mark_prediction_probability, dim=-1)
        # [batch_size, seq_len, num_mark]
        mark_prediction_probability = rearrange(mark_prediction_probability, "b s ne -> b ne s")
        # [batch_size, num_mark, seq_len]
        mark_loss = F.cross_entropy(input=mark_prediction_probability, target=mark_next.long(), reduction="none")
        # [batch_size, seq_len]
        mark_loss = (mark_loss * mask_next).sum()

        return mtpp_nll_loss, mark_loss

    sample_time = sample_time

    @torch.inference_mode()
    def next_event_prediction_time_mark(
        self: Self,
        time_history: torch.Tensor,
        time_next: torch.Tensor,
        mark_history: torch.Tensor,
        mask_history: torch.Tensor,
        mean: float,
        std: float,
        sample_rate: int,
        mae_step: int,
        evaluation: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the next event prediction from the THP by MAE and F1.
        This function first predict the time of the next event then the mark of the next event given the TRUE time.
        This can ensure the mark accuracy is only influenced by the model, not by the predicted time.

        Args:
            self (Self): the THP model.
            time_history (torch.Tensor): the time of historical mark
            time_next (torch.Tensor): the time of the next true mark
            mark_history (torch.Tensor): the mark of historical mark
            mark_next (torch.Tensor): the mark of the next true mark
            mask_history (torch.Tensor): the mask of historical sequences, 1 meaning a true event, and 0 meaning a fake event.
            mask_next (torch.Tensor): the mask of the next true mark
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
            mark_history=mark_history,
            mask_history=mask_history,
            number_of_total_samples=sample_rate,
            step=mae_step,
            mean=mean,
            std=std,
        )  # [sample_rate, batch_size, seq_len]
        pred_time = pred_time.mean(dim=0)  # [batch_size, seq_len]

        if evaluation:
            _, intensity_all_mark = self.model(time_history, time_next, mark_history, mask_history)
        # [batch_size, seq_len, num_mark]
        else:
            _, intensity_all_mark = self.model(time_history, pred_time, mark_history, mask_history)
            # [batch_size, seq_len, num_mark]

        mark_distribution = intensity_all_mark / intensity_all_mark.sum(dim=-1, keepdim=True)
        # [batch_size, seq_len, num_mark]
        return pred_time, mark_distribution

    @torch.inference_mode()
    def next_event_prediction_mark_time(
        self: Self,
        time_history: torch.Tensor,
        mark_history: torch.Tensor,
        mark_next: torch.Tensor,
        mask_history: torch.Tensor,
        mean: float,
        std: float,
        sample_rate: int,
        mae_e_step: int,
        evaluation: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the next event prediction from the THP by MAE and F1.
        This function first predict the time of the next event then the mark of the next event given the TRUE time.
        This can ensure the mark accuracy is only influenced by the model, not by the predicted time.

        Args:
            self (Self): the THP model.
            time_history (torch.Tensor): the time of historical mark
            time_next (torch.Tensor): the time of the next true mark
            mark_history (torch.Tensor): the mark of historical mark
            mark_next (torch.Tensor): the mark of the next true mark
            mask_history (torch.Tensor): the mask of historical sequences, 1 meaning a true event, and 0 meaning a fake event.
            mask_next (torch.Tensor): the mask of the next true mark
            mean (float): the mean of all time intervals
            std (float): the standard variance of all time intervals
            sample_rate (int): how many samples are needed for one time prediction
            mae_e_step (int): how many samples for one event are generated in one shot
            evaluation (bool): If true, we are in the evaluation mode, the mark distribution is at the time_next.
                               If false, we are in the prediction mode, the mark distribution is at the pred_time

        Returns:
            tuple[torch.Tensor, torch.Tensor]: predicted time and mark distribution
        """
        inf_val, resolution_inf, resolution_between_mark = decide_resolution_inf_and_resolution_between_mark(
            time_history, memory_ceiling, self.num_mark, mean, std
        )
        time_next_inf = torch.ones_like(time_history, device=self.device) * inf_val

        (
            expanded_integral_all_mark_to_inf,
            expanded_intensity_all_mark_to_inf,
            timestamp,
        ) = self.model.integral_intensity_time_next_2d(
            mark_history, time_history, time_next_inf, mask_history, resolution_inf
        )
        # 2 * [batch_size, seq_len, resolution_inf, num_mark]
        expanded_integral_sum_over_mark_to_inf = expanded_integral_all_mark_to_inf.sum(dim=-1, keepdim=True)
        # [batch_size, seq_len, resolution_inf, 1]
        expanded_probability_inf = expanded_intensity_all_mark_to_inf * torch.exp(
            -expanded_integral_sum_over_mark_to_inf
        )
        # [batch_size, seq_len, resolution_inf, num_mark]
        mark_distribution = approximate_integration(expanded_probability_inf, timestamp, dim=-2, only_integral=True)
        # [batch_size, seq_len, num_mark]
        tau_sampled_all_event = self.sample_time(
            sampling_approach="its",
            task="mt",
            mark_history=mark_history,
            time_history=time_history,
            mask_history=mask_history,
            p_m=mark_distribution,
            resolution=resolution_between_mark,
            number_of_total_samples=sample_rate,
            step=mae_e_step,
            inf_val=inf_val,
            mean=mean,
            std=std,
        )  # [sample_rate, batch_size, seq_len, num_mark]
        tau_pred_all_event = tau_sampled_all_event.mean(dim=0)  # [batch_size, seq_len, num_mark]

        if evaluation:
            mark_next_mask = torch.nn.functional.one_hot(mark_next, num_classes=self.num_mark)
            # [batch_size, seq_len, num_mark]
            pred_time = (tau_pred_all_event * mark_next_mask).sum(dim=-1)  # [batch_size, seq_len]
        else:
            pred_time = tau_pred_all_event  # [batch_size, seq_len, num_mark]

        return pred_time, mark_distribution

    def extract_plot_data(self, minibatch):
        """
        This function extracts input_time, input_mark, input_intensity, mask, mean, and std from the minibatch.

        ### Args
            * ```list``` minibatch
              shape: ```[[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]```
              data structure: [[input_time, input_mark, score, mask], (mean, std)]
              data type: [```torch.tensor```, ```torch.tensor```, ```torch.tensor```, ```torch.tensor```, (```float```, ```float```)]
              The input minibatch.

        ### Outputs
            * ```torch.tensor``` input_time
              shape: ```[batch_size, seq_len + 1]```
              Raw event timestamp sequence.
            * ```torch.tensor``` input_mark
              shape: ```[batch_size, seq_len + 1]```
              Raw event marks sequence.
            * ```torch.tensor``` mask
              shape: ```[batch_size, seq_len + 1]```
              Raw mask sequence.
            * ```int``` mean
              The mean of all $ t_i - t_{i - 1} $ in the entire dataset. Dataloader is responsible to provide this value if needed.
            * ```int``` std
              The mean of all $ t_i - t_{i - 1} $ in the entire dataset. Dataloader is responsible to provide this value if needed.
        """
        input_time, input_mark, _, mask, input_intensity = minibatch[0]
        mean, std = minibatch[1]

        return input_time, input_mark, input_intensity, mask, mean, std

    @torch.inference_mode()
    def figure_intensity(self, input_data, opt):
        """
        Function prober, used by evaluator to draw plots of the intensity function.

        You should declare the following arguments in your config file:
        1. ```int``` resolution: The number of interpolated points in a time interval between two adjoint mark for integration estimation.
                                 The number of interpolated points counts the start and end point of the interval.

        ### Args
            * ```list``` input_data
              shape: ```[[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]```
              data structure: [[input_time, input_mark, score, mask], (mean, std)]
              data type: [```torch.tensor```, ```torch.tensor```, ```torch.tensor```, ```torch.tensor```, (```float```, ```float```)]
              The input minibatch.
            * ```namespace``` opt
              plot and model configs
        """
        argument_check(opt, **{"resolution": int})

        input_time, input_mark, input_intensity, mask, mean, std = self.extract_plot_data(input_data)

        time_history, time_next = self.divide_history_and_next(input_time)  # [batch_size, seq_len]
        mark_history, mark_next = self.divide_history_and_next(input_mark)
        # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)  # [batch_size, seq_len]

        expand_integral, expand_intensity, timestamp = self.model.integral_intensity_time_next_2d(
            mark_history, time_history, time_next, mask_history, opt.resolution
        )
        # 3 * [batch_size, seq_len, resolution, num_mark]
        check_tensor(expand_integral)
        check_tensor(expand_intensity)
        if expand_intensity.shape != expand_integral.shape:
            raise ValueError("Why expand_intensity and expand_integral have different shapes?")

        data = {
            "time_next": time_next,
            "mark_next": mark_next,
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
        1. ```int``` resolution: The number of interpolated points in a time interval between two adjoint mark for integration estimation.
                                 The number of interpolated points counts the start and end point of the interval.

        ### Args
            * ```list``` input_data
              shape: ```[[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]```
              data structure: [[input_time, input_mark, score, mask], (mean, std)]
              data type: [```torch.tensor```, ```torch.tensor```, ```torch.tensor```, ```torch.tensor```, (```float```, ```float```)]
              The input minibatch.
            * ```namespace``` opt
              plot and model configs
        """
        argument_check(opt, **{"resolution": int})

        input_time, input_mark, input_intensity, mask, mean, std = self.extract_plot_data(input_data)

        time_history, time_next = self.divide_history_and_next(input_time)  # [batch_size, seq_len]
        mark_history, mark_next = self.divide_history_and_next(input_mark)
        # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)  # [batch_size, seq_len]

        expand_integral, expand_intensity, timestamp = self.model.integral_intensity_time_next_2d(
            mark_history, time_history, time_next, mask_history, opt.resolution
        )
        # 3 * [batch_size, seq_len, resolution, num_mark]
        check_tensor(expand_integral)
        check_tensor(expand_intensity)
        if expand_intensity.shape != expand_integral.shape:
            raise ValueError("Why expand_intensity and expand_integral have different shapes?")

        data = {
            "time_next": time_next,
            "mark_next": mark_next,
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
        1. ```int``` resolution: The number of interpolated points in a time interval between two adjoint mark for integration estimation.
                                 The number of interpolated points counts the start and end point of the interval.

        ### Args
            * ```list``` input_data
              shape: ```[[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]```
              data structure: [[input_time, input_mark, score, mask], (mean, std)]
              data type: [```torch.tensor```, ```torch.tensor```, ```torch.tensor```, ```torch.tensor```, (```float```, ```float```)]
              The input minibatch.
            * ```namespace``` opt
              plot and model configs
        """
        argument_check(opt, **{"resolution": int})

        input_time, input_mark, input_intensity, mask, mean, std = self.extract_plot_data(input_data)

        time_history, time_next = self.divide_history_and_next(input_time)  # [batch_size, seq_len]
        mark_history, mark_next = self.divide_history_and_next(input_mark)
        # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)  # [batch_size, seq_len]

        expand_integral, expand_intensity, timestamp = self.model.integral_intensity_time_next_2d(
            mark_history, time_history, time_next, mask_history, opt.resolution
        )
        # 3 * [batch_size, seq_len, resolution, num_mark]
        check_tensor(expand_integral)
        check_tensor(expand_intensity)
        if expand_intensity.shape != expand_integral.shape:
            raise ValueError("Why expand_intensity and expand_integral have different shapes?")
        expand_probability = expand_intensity * torch.exp(-expand_integral.sum(dim=-1, keepdim=True))
        # [batch_size, seq_len, resolution, num_mark]
        data = {
            "time_next": time_next,
            "mark_next": mark_next,
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
        1. ```int``` resolution: The number of interpolated points in a time interval between two adjoint mark for integration estimation.
                                 The number of interpolated points counts the start and end point of the interval.
        2. ```int``` sample_rate: The number of interpolated points in a time interval between two adjoint mark for integration estimation.
                                  The number of interpolated points counts the start and end point of the interval.
        3. ```int``` mae_step: This parameter controls how many samples are generated in one shot when sampling from p(t).
        4. ```int``` mae_e_step: This parameter controls how many samples are generated in one shot when sampling from all p(t|m)s at the same time.

        ### Args
            * ```list``` input_data
              shape: ```[[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]```
              data structure: [[input_time, input_mark, score, mask], (mean, std)]
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

        input_time, input_mark, input_intensity, mask, mean, std = self.extract_plot_data(input_data)

        time_history, time_next = self.divide_history_and_next(input_time)  # [batch_size, seq_len]
        mark_history, mark_next = self.divide_history_and_next(input_mark)
        # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)  # [batch_size, seq_len]

        data, timestamp = self.model.model_probe_function(
            mark_history,
            time_history,
            time_next,
            mask_history,
            mask_next,
            opt.resolution,
        )

        pred_time, _ = self.next_event_prediction_time_mark(
            time_history,
            time_next,
            mark_history,
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
            mark_history,
            mark_next,
            mask_history,
            mean=mean,
            std=std,
            sample_rate=self.sample_rate,
            mae_e_step=opt.mae_e_step,
            evaluation=False,
        )
        # [batch_size, seq_len, num_mark] + [batch_size, seq_len, num_mark]
        mark_next_mask = torch.nn.functional.one_hot(mark_next, num_classes=self.num_mark)
        # [batch_size, seq_len, num_mark]
        pred_time = (pred_time_all_marks * mark_next_mask).sum(dim=-1)  # [batch_size, seq_len, num_mark]
        maes_ptm = torch.abs(pred_time - time_next) * mask_next  # [batch_size, seq_len]
        top_k = evaluate_on_one_batch(mark_dist, mark_next, mask_next, "top_k", dim_input=-2)
        # [batch_size, num_mark]
        probability_sum = mark_dist.sum(dim=-1)  # [batch_size, seq_len]

        # Append additional info into the data dict.
        data.update(
            {
                "mark_next": mark_next,
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
    def get_spearman_and_l1(self, input_data, opt):
        """
        Used by evaluator to calculate the average gap between the predicted and real distribution using L1 distance and spearman coefficient.

        You should declare the following arguments in your config file:
        1. ```int``` resolution: The number of interpolated points in a time interval between two adjoint mark for integration estimation.
                                 The number of interpolated points counts the start and end point of the interval.

        ### Args
            * ```list``` input_data
              shape: ```[[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]```
              data structure: [[input_time, input_mark, score, mask], (mean, std)]
              data type: [```torch.tensor```, ```torch.tensor```, ```torch.tensor```, ```torch.tensor```, (```float```, ```float```)]
              The input minibatch.
            * ```namespace``` opt
              plot and model configs

        ### Outputs:
            * ```float``` spearman
              The spearman coefficient between the predicted and real distribution.
            * ```float``` l1
              The l1 distance between the predicted and real distribution.
        """
        argument_check(opt, **{"resolution": int})

        input_time, input_mark, input_intensity, mask, mean, std = self.extract_plot_data(input_data)
        time_history, time_next = self.divide_history_and_next(input_time)  # [batch_size, seq_len]
        mark_history, mark_next = self.divide_history_and_next(input_mark)
        # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)  # [batch_size, seq_len]

        expand_integral, expand_intensity, timestamp = self.model.integral_intensity_time_next_2d(
            mark_history, time_history, time_next, mask_history, opt.resolution
        )
        # 3 * [batch_size, seq_len, resolution, num_mark]
        check_tensor(expand_integral)
        check_tensor(expand_intensity)
        if expand_intensity.shape != expand_integral.shape:
            raise ValueError("Why expand_intensity and expand_integral have different shapes?")
        expand_probability = expand_intensity * torch.exp(-expand_integral.sum(dim=-1, keepdim=True))
        # [batch_size, seq_len, resolution, num_mark]
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
    def get_mae_and_f1(self, input_data, opt):
        """
        Used by evaluator to evaluate the performance of predicted time from p(t) and mark from p(m|t).

        You should declare the following arguments in your config file:
        1. ```int``` sample_rate: The number of interpolated points in a time interval between two adjoint mark for integration estimation.
                                  The number of interpolated points counts the start and end point of the interval.
        2. ```int``` mae_step: This parameter controls how many samples are generated in one shot when sampling from p(t).

        ### Args
            * ```list``` input_data
              shape: ```[[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]```
              data structure: [[input_time, input_mark, score, mask], (mean, std)]
              data type: [```torch.tensor```, ```torch.tensor```, ```torch.tensor```, ```torch.tensor```, (```float```, ```float```)]
              The input minibatch.
            * ```namespace``` opt
              plot and model configs

        ### Outputs:
            * ```np.ndarray``` mae
              shape: ```[batch_size, seq_len]```
              The MAE value, which is the time gap between each predicted and real event.
            * ```float``` f1_1
              The f1 value shows the accuracy of the predicted marks.
            * ```np.ndarray``` dist
              shape: ```[batch_size, seq_len]```
              Predicted mark distribution at when an event is observed.
            * ```np.ndarray``` mark_next
              shape: ```[batch_size, seq_len]```
              Real marks of observed mark.
        """
        argument_check(opt, **{"sample_rate": int, "mae_step": int})

        input_time, input_mark, input_intensity, mask, mean, std = self.extract_plot_data(input_data)
        time_history, time_next = self.divide_history_and_next(input_time)  # [batch_size, seq_len]
        mark_history, mark_next = self.divide_history_and_next(input_mark)
        # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)  # [batch_size, seq_len]
        pred_time, mark_dist = self.next_event_prediction_time_mark(
            time_history,
            time_next,
            mark_history,
            mask_history,
            mean=mean,
            std=std,
            sample_rate=self.sample_rate,
            mae_step=opt.mae_step,
        )
        # [batch_size, seq_len] + [batch_size, seq_len, num_mark]
        mae = torch.abs(pred_time - time_next) * mask_next  # [batch_size, seq_len]
        mae = mae.sum(dim=-1) / mask_next.sum(dim=-1)
        pred_mark = mark_dist.argmax(dim=-1)  # [batch_size, seq_len]
        results = evaluate_on_one_batch(pred_mark, mark_next, mask_next, ["acc", "macro-f1", "micro-f1"])
        acc = results["acc"]
        macro_f1 = results["macro-f1"]
        micro_f1 = results["micro-f1"]

        mae, mark_next, mark_dist, acc, macro_f1, micro_f1, mask_next = move_from_tensor_to_ndarray(
            mae, mark_next, mark_dist, acc, macro_f1, micro_f1, mask_next
        )
        pred_time, mark_next, mark_dist = break_batched_inputs_into_seqs(mask_next, pred_time, mark_next, mark_dist)
        return mae.tolist(), acc.tolist(), macro_f1.tolist(), micro_f1.tolist(), pred_time, mark_dist, mark_next

    @torch.inference_mode()
    def get_mae_e_and_f1(self, input_data, opt):
        """
        Used by evaluator to evaluate the performance of predicted time from p(m) and mark from p(t|m).

        You should declare the following arguments in your config file:
        1. ```int``` sample_rate: The number of interpolated points in a time interval between two adjoint mark for integration estimation.
                                  The number of interpolated points counts the start and end point of the interval.
        2. ```int``` mae_e_step: This parameter controls how many samples are generated in one shot when sampling from p(t|m).

        ### Args
            * ```list``` input_data
              shape: ```[[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]```
              data structure: [[input_time, input_mark, score, mask], (mean, std)]
              data type: [```torch.tensor```, ```torch.tensor```, ```torch.tensor```, ```torch.tensor```, (```float```, ```float```)]
              The input minibatch.
            * ```namespace``` opt
              plot and model configs

        ### Outputs:
            * ```np.ndarray``` maes
              shape: ```[batch_size, seq_len]```
              The MAE-E values when we pick predicted times using real marks.
            * ```float``` f1_2
              The f1 value shows the accuracy of the predicted marks.
            * ```np.ndarray``` probability_sum
              shape: ```[batch_size, seq_len]```
              The sum of calculated p(m) over all marks.
            * ```np.adarray``` p_m
              shape: ```[batch_size, seq_len, num_mark]```
              The value of calculated p(m).
            * ```np.ndarray``` tau_pred_all_event
              shape: ```[batch_size, seq_len, num_mark]```
              The predicted time for each mark using p(t|m).
            * ```np.ndarray``` time_next
              shape: ```[batch_size, seq_len]```
              Real time of observed mark.
            * ```np.ndarray``` mark_next
              shape: ```[batch_size, seq_len]```
              Real marks of observed mark.
        """
        argument_check(opt, **{"sample_rate": int, "mae_e_step": int})

        input_time, input_mark, _, mask, mean, std = self.extract_plot_data(input_data)
        time_history, time_next = self.divide_history_and_next(input_time)  # [batch_size, seq_len]
        mark_history, mark_next = self.divide_history_and_next(input_mark)
        # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)  # [batch_size, seq_len]

        pred_time_all_marks, mark_dist = self.next_event_prediction_mark_time(
            time_history,
            mark_history,
            mark_next,
            mask_history,
            mean=mean,
            std=std,
            sample_rate=self.sample_rate,
            mae_e_step=opt.mae_e_step,
            evaluation=False,
        )
        # [batch_size, seq_len, num_mark] + [batch_size, seq_len, num_mark]
        mark_next_mask = torch.nn.functional.one_hot(mark_next, num_classes=self.num_mark)
        # [batch_size, seq_len, num_mark]
        pred_time = (pred_time_all_marks * mark_next_mask).sum(dim=-1)  # [batch_size, seq_len, num_mark]
        mae_e = torch.abs(pred_time - time_next) * mask_next  # [batch_size, seq_len]
        mae_e = mae_e.sum(dim=-1) / mask_next.sum(dim=-1)
        pred_mark = mark_dist.argmax(dim=-1)  # [batch_size, seq_len]
        results = evaluate_on_one_batch(pred_mark, mark_next, mask_next, ["acc", "macro-f1", "micro-f1"])
        acc = results["acc"]
        macro_f1 = results["macro-f1"]
        micro_f1 = results["micro-f1"]

        (
            mae_e,
            mark_next,
            mark_dist,
            acc,
            macro_f1,
            micro_f1,
            mask_next,
            pred_time_all_marks,
            time_next,
        ) = move_from_tensor_to_ndarray(
            mae_e,
            mark_next,
            mark_dist,
            acc,
            macro_f1,
            micro_f1,
            mask_next,
            pred_time_all_marks,
            time_next,
        )
        mark_dist, pred_time_all_marks, time_next, mark_next = break_batched_inputs_into_seqs(
            mask_next, mark_dist, pred_time_all_marks, time_next, mark_next
        )

        return (
            mae_e.tolist(),
            acc.tolist(),
            macro_f1.tolist(),
            micro_f1.tolist(),
            mark_dist,
            pred_time_all_marks,
            time_next,
            mark_next,
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
              data structure: [[input_time, input_mark, score, mask], (mean, std)]
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
        argument_check(opt, **{"sample_rate": int, "sample_substep": int})

        input_time, input_mark, input_intensity, mask, mean, std = self.extract_plot_data(input_data)
        time_history, time_next = self.divide_history_and_next(input_time)  # [batch_size, seq_len]
        mark_history, mark_next = self.divide_history_and_next(input_mark)
        # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)  # [batch_size, seq_len]

        inf_val, resolution_inf, resolution_between_mark = decide_resolution_inf_and_resolution_between_mark(
            time_next, memory_ceiling, self.num_mark, mean, std
        )
        time_next_inf = torch.ones_like(time_history, device=self.device) * inf_val
        # [batch_size, seq_len]
        (
            expanded_integral_all_mark_to_inf,
            expanded_intensity_all_mark_to_inf,
            timestamp,
        ) = self.model.integral_intensity_time_next_2d(
            mark_history, time_history, time_next_inf, mask_history, resolution_inf
        )
        # 2 * [batch_size, seq_len, resolution, num_mark]
        expanded_probability_inf = (
            torch.exp(-expanded_integral_all_mark_to_inf.sum(dim=-1, keepdim=True))
            * expanded_intensity_all_mark_to_inf
        )
        # [batch_size, seq_len, resolution, num_mark]
        probability_integral_to_inf = approximate_integration(
            expanded_probability_inf, timestamp, dim=-2, only_integral=True
        )
        # [batch_size, seq_len, num_mark]
        # step 2: get the time prediction for that kind of event
        tau_pred_all_event = self.sample_time(
            sampling_approach="its",
            task="mt",
            mark_history=mark_history,
            time_history=time_history,
            mask_history=mask_history,
            p_m=probability_integral_to_inf,
            resolution=resolution_between_mark,
            number_of_total_samples=opt.sample_rate,
            step=opt.sample_substep,
            inf_val=inf_val,
            mean=mean,
            std=std,
        )  # [sample_rate, batch_size, seq_len, num_mark]

        sampled_times_mean = tau_pred_all_event.mean(dim=0)  # [batch_size, seq_len, num_mark]
        predicted_time, predicted_mark = sampled_times_mean.min(dim=-1)  # [batch_size, seq_len] + [batch_size, seq_len]
        maes = torch.abs(time_next - predicted_time) * mask_next  # [batch_size, seq_len]
        maes = maes.sum(dim=-1) / mask_next.sum(dim=-1)  # [batch_size]

        results = evaluate_on_one_batch(predicted_mark, mark_next, mask_next, ["acc", "macro-f1", "micro-f1"])
        acc = results["acc"]
        macro_f1 = results["macro-f1"]
        micro_f1 = results["micro-f1"]
        maes = move_from_tensor_to_ndarray(maes)

        return maes.tolist(), acc.tolist(), macro_f1.tolist(), micro_f1.tolist()

    def balanced_sampling_from_distribution(self, input_data, opt):
        """
        This function samples from the distribution p(m, t) by sampling the mark first from p(m) then time from p(t|m).
        All samples can later be used to draw the distribution plot.

        You should declare the following arguments in your config file:
        1. ```int``` sample_rate: how many time samples from the time distribution are needed.
        2. ```int``` sample_substep: This parameter controls how many samples are generated in one shot when sampling from p(t|m).

        ### Args
            * ```list``` input_data
              shape: ```[[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]```
              data structure: [[input_time, input_mark, score, mask], (mean, std)]
              data type: [```torch.tensor```, ```torch.tensor```, ```torch.tensor```, ```torch.tensor```, (```float```, ```float```)]
              The input minibatch.
            * ```namespace``` opt
              plot and model configs

        ### Outputs:
            * ```np.ndarray``` tau_pred_all_event
              shape: ```[batch_size, seq_len, num_mark]```
              Predicted time for all marks using p(t|m)
            * ```np.ndarray``` probability_integral_to_inf
              shape: ```[batch_size, seq_len, num_mark]```
              The value of p(m).
        """
        argument_check(opt, **{"sample_rate": int, "sample_substep": int})

        input_time, input_mark, input_intensity, mask, mean, std = self.extract_plot_data(input_data)
        time_history, time_next = self.divide_history_and_next(input_time)  # [batch_size, seq_len]
        mark_history, mark_next = self.divide_history_and_next(input_mark)
        # [batch_size, seq_len]

        input_time, input_mark, input_intensity, mask, mean, std = self.extract_plot_data(input_data)
        time_history, time_next = self.divide_history_and_next(input_time)  # [batch_size, seq_len]
        mark_history, mark_next = self.divide_history_and_next(input_mark)
        # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)  # [batch_size, seq_len]

        inf_val, resolution_inf, resolution_between_mark = decide_resolution_inf_and_resolution_between_mark(
            time_next, memory_ceiling, self.num_mark, mean, std
        )
        time_next_inf = torch.ones_like(time_history, device=self.device) * inf_val
        # [batch_size, seq_len]
        (
            expanded_integral_all_mark_to_inf,
            expanded_intensity_all_mark_to_inf,
            timestamp,
        ) = self.model.integral_intensity_time_next_2d(
            mark_history, time_history, time_next_inf, mask_history, resolution_inf
        )
        # 2 * [batch_size, seq_len, resolution, num_mark]
        expanded_probability_inf = (
            torch.exp(-expanded_integral_all_mark_to_inf.sum(dim=-1, keepdim=True))
            * expanded_intensity_all_mark_to_inf
        )
        # [batch_size, seq_len, resolution, num_mark]
        probability_integral_to_inf = approximate_integration(
            expanded_probability_inf, timestamp, dim=-2, only_integral=True
        )
        # [batch_size, seq_len, num_mark]
        # step 2: get the time prediction for that kind of event
        tau_pred_all_event = self.sample_time(
            sampling_approach="its",
            task="mt",
            mark_history=mark_history,
            time_history=time_history,
            mask_history=mask_history,
            p_m=probability_integral_to_inf,
            resolution=resolution_between_mark,
            number_of_total_samples=opt.sample_rate,
            step=opt.sample_substep,
            inf_val=inf_val,
            mean=mean,
            std=std,
        )  # [sample_rate, batch_size, seq_len, num_mark]

        probability_integral_to_inf, tau_pred_all_event, mask_next = move_from_tensor_to_ndarray(
            probability_integral_to_inf, tau_pred_all_event, mask_next
        )

        probability_integral_to_inf, tau_pred_all_event = break_batched_inputs_into_seqs(
            mask_next, probability_integral_to_inf, rearrange(tau_pred_all_event, "sr b sl ne -> b sl ne sr")
        )  # batch_size * [seq_len, num_mark] + batch_size * [seq_len, num_mark, sample_rate]

        tau_pred_all_event = [rearrange(item, "sl ne sr -> sr sl ne") for item in tau_pred_all_event]

        return tau_pred_all_event, probability_integral_to_inf

    '''
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
        forward_complete_data, backward_complete_data, padded_obs_data, padded_backward_obs_event_seq, (mean, std) = (
            input_data
        )

        roc_result = []
        for (
            obs_time_for_one_seq,
            obs_mark_for_one_seq,
            obs_mask_for_one_seq,
            missing_mask_for_one_seq,
            _,
        ) in padded_obs_data:
            obs_time_history_for_one_seq, obs_time_next_for_one_seq = self.divide_history_and_next(obs_time_for_one_seq)
            # [batch_size, seq_len] * 2
            obs_mark_history_for_one_seq, obs_mark_next_for_one_seq = self.divide_history_and_next(
                obs_mark_for_one_seq
            )
            # [batch_size, seq_len] * 2
            obs_mask_history_for_one_seq, obs_mask_next_for_one_seq = self.divide_history_and_next(obs_mask_for_one_seq)
            # [batch_size, seq_len]

            missing_mask_for_one_seq = self.convert_missing_mask_to_gap_mask(missing_mask_for_one_seq)
            # [num_samples, ...]
            integral_all_mark, intensity_all_mark = self.model(
                obs_time_history_for_one_seq.float(),
                obs_time_next_for_one_seq.float(),
                obs_mark_history_for_one_seq,
                obs_mask_history_for_one_seq,
            )
            # [num_samples, seq_len, num_mark]

            integral_sum = integral_all_mark.sum(dim=-1)  # [num_samples, seq_len]
            intensity_sum = intensity_all_mark.sum(dim=-1)  # [num_samples, seq_len]

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
        (time_seq, mark, commission, mask), (mean, std) = input_data

        time_history, time_next = self.divide_history_and_next(time_seq)  # [batch_size, seq_len]
        mark_history, mark_next = self.divide_history_and_next(mark)  # [batch_size, seq_len]
        _, commission_next = self.divide_history_and_next(commission)  # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)  # [batch_size, seq_len]
        time_history = time_history.float()
        time_next = time_next.float()

        _, intensity_all_mark = self.model(time_history, time_next, mark_history, mask_history)
        # 2 * [batch_size, seq_len, num_mark]

        intensity_sum_from_tl_to_time_next = intensity_all_mark.sum(dim=-1)
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
    '''

    def train_step(self, minibatch):
        """
        This function unpacks the minibatch, calls the train_procedure() to calculate the loss, and do the backpropagation.

        ### Args
            * ```torch.nn.Module``` model
              The MTPP model that we train.
            * ```list``` minibatch
              shape: ```[[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]```
              data structure: [[input_time, input_mark, score, mask], (mean, std)]
              data type: [```torch.tensor```, ```torch.tensor```, ```torch.tensor```, ```torch.tensor```, (```float```, ```float```)]
              The input minibatch.
            * ```torch.device``` device
              where we train the model.

        ### Outputs:
            * ```float``` tpp_loss
              The average NLL loss without dummy events, specifically the start and the end event.
            * ```float``` fact
              The average NLL loss with the real distribution. This value only makes sense for synthetic datasets.
            * ```float``` mark_loss
              The average cross-entropy loss of the event prediction distribution. The value is only for performance measure porpose.
              The training loss does not and should not include this value.
        """
        self.model.train()

        (time, mark, fact, mask), (mean, std) = (
            minibatch  # 3 * [batch_size, seq_len + 1, 1] & [batch_size, seq_len, 1] & 2 numbers
        )
        loss, neg_log_likeli_loss, marker_loss, the_number_of_mark = self.forward(
            "train", time, mark, mask, mean, std
        )
        loss.backward()

        tpp_loss, mark_loss = neg_log_likeli_loss.item(), marker_loss.item()
        fact = fact.sum() / the_number_of_mark

        return tpp_loss, fact, mark_loss

    def evaluation_step(self, minibatch):
        """
        This function unpacks the minibatch, calls the evaluation_procedure() to compute the metrics.

        ### Args
            * ```torch.nn.Module``` model
              The MTPP model that we train.
            * ```list``` minibatch
              shape: ```[[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]```
              data structure: [[input_time, input_mark, score, mask], (mean, std)]
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
            * ```float``` mark_loss
              The average cross-entropy loss of the event prediction distribution. The value is only for performance measure porpose.
            * ```float``` mae
              The average error between predicted time and real time.
            * ```float``` f1
              The prediction accuracy of predicted marks.
        """
        self.model.eval()

        (time, mark, fact, mask), (mean, std) = (
            minibatch  # 3 * [batch_size, seq_len + 1, 1] & [batch_size, seq_len, 1] & 2 numbers
        )
        (
            time_loss,
            loss_survival,
            mark_loss,
            mae,
            acc,
            macro_f1,
            micro_f1,
            the_number_of_mark,
        ) = self.forward("evaluate", time, mark, mask, mean, std)

        time_loss = time_loss.item()
        loss_survival = loss_survival.item()
        mark_loss = mark_loss.item()
        fact = fact.sum().item() / the_number_of_mark

        return time_loss, loss_survival, fact, mark_loss, mae, acc, macro_f1, micro_f1

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
            [absolute loss, relative loss, mark loss]
            """
            return [input_data[0], input_data[0] - input_data[1], input_data[2]]

        def test_postprocess(input_data):
            """
            Evaluation process
            [absolute loss, relative loss, mark loss, mae value]
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
            format_dict["mark_loss"] = pack_one_value_to_dict(input_data[2])
            return format_dict

        def test_log_print_format(input_data):
            format_dict = {}
            format_dict["absolute_NLL_loss"] = pack_one_value_to_dict(input_data[0])
            format_dict["avg_survival_loss"] = pack_one_value_to_dict(input_data[1])
            format_dict["relative_NLL_loss"] = pack_one_value_to_dict(input_data[2])
            format_dict["mark_loss"] = pack_one_value_to_dict(input_data[3])
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
        return [evaluation_report_format_dict["absolute_NLL_loss"], test_report_format_dict["absolute_NLL_loss"]], [
            "evaluation_absolute_loss",
            "test_absolute_loss",
        ]

    """
    metric number is the length of the output of choose_metric
    """
    metric_number = 2  # metric number is the length of the output of choose_metric
