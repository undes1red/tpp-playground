import copy
import math

import numpy as np
import torch
import torch.nn.functional as F
from einops import pack, rearrange, reduce, repeat

from src.ehd.model.basic_ehd_model import BasicModel
from src.ehd.model.cff.submodel import EHDBackend
from src.ehd.model.cff.utils import filter, generate_masks
from src.ehd.utils import suffix
from src.toolbox.misc import (
    check_tensor,
    compile_model,
    easy_model_load,
    get_logger,
    move_from_tensor_to_ndarray,
    pack_one_value_to_dict,
)

logger = get_logger(__name__)


class CFF(BasicModel):
    """
    The EHD module.
    This module takes data and trained MTPP model, such as FullyNN, FENN, IFIB-C, etc.
    """

    def __init__(
        self,
        training,
        opt,
        d_input,
        d_rnn,
        d_hidden,
        n_layers_encoder,
        n_layers_decoder,
        n_head,
        d_qk,
        d_v,
        dropout,
        epsilon_l,
        epsilon_d,
        alpha,
        beta,
        epsilon,
        device,
        loaded_mtpp_model_args,
        samples_for_l_e=16,
    ):
        super().__init__()
        self.device = device
        self.opt = opt
        # The probability gap.
        self.epsilon_l = math.log(epsilon_l)
        self.epsilon_d = math.log(epsilon_d)
        self.param_number_useful_events = alpha
        self.param_how_useful_events_are = beta
        self.samples_for_l_e = samples_for_l_e

        """
        Load the trained TPP model checkpoint.
        Training: Model parameters will be loaded.
        not Training: Model parameters will not be loaded.
        """
        self.mtpp_model = easy_model_load(
            "TPP",
            training=False,
            root_path=opt.root_path,
            replace_id=loaded_mtpp_model_args["replace_id"],
            dataset_name=opt.info_dict["dataset_name"],
            dataset_name_in_model_config=opt.info_dict["dataset_name"],
            device=device,
            compile=False,
            evaluation=True,
            only_model_structure=not training,
            model_name=loaded_mtpp_model_args["model_name"],
            lr=loaded_mtpp_model_args["lr"],
            used_batch_size=loaded_mtpp_model_args["training_batch_size"],
            n_training_steps=loaded_mtpp_model_args["n_training_steps"],
            used_procedure_config=loaded_mtpp_model_args["procedure_config"],
            used_dataloader_config=loaded_mtpp_model_args["dataloader_config"],
            model_config=loaded_mtpp_model_args["model_config"],
        )

        """
        Preparing the EHD model-agnostic part.
        """
        self.epsilon = epsilon
        self.num_marks = opt.info_dict["num_marks"]
        self.start_time = opt.info_dict["t_0"]
        self.end_time = opt.info_dict["T"]
        # This length does not include the dummy event.
        self.seq_len_x = opt.info_dict["length_of_x"]
        self.seq_len_h = opt.info_dict["length_of_h"]

        self.model = EHDBackend(
            num_marks=self.num_marks,
            seq_len_x=self.seq_len_x,
            seq_len_h=self.seq_len_h,
            d_input=d_input,
            d_rnn=d_rnn,
            d_hidden=d_hidden,
            n_layers_encoder=n_layers_encoder,
            n_layers_decoder=n_layers_decoder,
            n_head=n_head,
            d_qk=d_qk,
            d_v=d_v,
            dropout=dropout,
            device=device,
        )

        self.model = compile_model(self.model, opt.compile, opt.compile_backend)

    def divide_history_and_future(self, input_time, input_marks, input_mask):
        """
        TODO: This function needs an overhaul to handle real-world datasets.
        I don't want to generate too much data from one sequence for memory and training speed concerns.
        Maybe at most around 50 generated data sequence from one original data sequence.
        """
        gen_time_history, gen_time_next = input_time[:, : self.seq_len_h + 1], input_time[:, self.seq_len_h + 1 :]
        gen_marks_history, gen_marks_next = input_marks[:, : self.seq_len_h + 1], input_marks[:, self.seq_len_h + 1 :]
        gen_mask_history, gen_mask_next = input_mask[:, : self.seq_len_h + 1], input_mask[:, self.seq_len_h + 1 :]

        return (
            (gen_time_history, gen_time_next),
            (gen_marks_history, gen_marks_next),
            (gen_mask_history, gen_mask_next),
        )

    def forward(self, task_name, *args, **kwargs):
        """
        The entrance of the CFF.

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
            # for evaluation.
            "get_explanation": self.get_explanation,
            "lsp_and_lrp": self.get_unhinged_perplexity_gap,
            "lsp_and_lrp_compared_with_gs_and_rd": self.lsp_and_lrp_compared_with_gs_and_rd,
            "lsp_and_lrp_trend": self.lsp_and_lrp_trend,
            "removed_marks": self.removed_marks,
        }

        return task_mapper[task_name](*args, **kwargs)

    def train_procedure(self, input_time, input_marks, input_mask, mean, std):
        if not input_mask.all():
            raise Exception("All values in the input_mask should be true since we have nothing to mask.")

        (time_history, time_future), (marks_history, marks_future), (mask_history, mask_future) = (
            self.divide_history_and_future(input_time, input_marks, input_mask)
        )
        # ([batch_size, seq_len_h + 1], [batch_size, seq_len_x + 1]) * 3

        # Here, mask = 1: important. Removing them would cause counterfactual results.
        #       mask = 0: noises or unrelated marks. Keeping them makes no benefit for modeling the future.
        input_probability = self.model(
            marks_history, marks_future, time_history, time_future, mask_history, mask_future
        )
        # [batch_size, length_of_h + 1, 2]
        check_tensor(input_probability)

        # Tell the average gap between p(y = 1|x, H) and p(y = 0|x, H). Bigger probability gap means the model is more certain about the result.
        gap_between_p_1_and_p_0 = input_probability[:, :, 1] - input_probability[:, :, 0]
        # [batch_size, length_of_h + 1]
        gap_between_p_1_and_p_0 = torch.abs(gap_between_p_1_and_p_0.detach() * mask_history).sum()
        avg_gap_between_p_1_and_p_0 = gap_between_p_1_and_p_0 / mask_history.sum().item()

        repeated_input_probability = repeat(input_probability, "... -> n ...", n=self.samples_for_l_e)
        # [samples_for_l_e, batch_size, length_of_h + 1, 2]
        # Generate the history mask and corresponding filter mask.
        history_mask, mask_on_whole_seq = generate_masks(repeated_input_probability, self.seq_len_x + 1)
        # [samples_for_l_e, batch_size, length_of_h + 1, 2] + [samples_for_l_e, batch_size, length_of_h + length_oF_x + 2, 2]

        # No.1 loss: L_n, optimize the length of essential marks.
        mask_on_history_without_dummy = history_mask[..., 1:, 1]
        # [samples_for_l_e, batch_size, length_of_h]
        loss_number_of_useful_events = (
            torch.linalg.norm(mask_on_history_without_dummy.float(), ord=1, dim=-1) / self.seq_len_h
        )
        # [num_of_samples_mask, batch_size]
        loss_number_of_useful_events = loss_number_of_useful_events.mean()

        # No.2 loss: L_e, optimize the quality of distilled marks.
        perplexity_p_h_f_x_os, perplexity_p_h_d_x_os, perplexity_p_h_l_x_os = (
            self.compute_perplexity_on_full_and_seg_sequences(
                input_marks, input_time, input_mask, mask_on_whole_seq, mean, std, training=True
            )
        )
        loss_how_useful_events_are = (
            F.relu(perplexity_p_h_f_x_os - perplexity_p_h_l_x_os + self.epsilon_l).mean()
            + F.relu(perplexity_p_h_d_x_os - perplexity_p_h_f_x_os + self.epsilon_d).mean()
        )

        loss = (
            self.param_number_useful_events * loss_number_of_useful_events
            + self.param_how_useful_events_are * loss_how_useful_events_are
        )

        return loss, loss_number_of_useful_events, loss_how_useful_events_are, avg_gap_between_p_1_and_p_0

    def evaluate_procedure(self, input_time, input_marks, input_mask, mean, std):
        """
        Since we removed all sequence shorter than seq_len_x + seq_len_h.
        We do not need to worry about the input_mask anymore.
        the input sequence has two dummy events at the start and end of the sequence.
        so the sequence length is in fact seq_len_x + seq_len_h + 2.
        """
        if not input_mask.all():
            raise Exception("All values in the input_mask should be true since we have nothing to mask.")
        (time_history, time_future), (marks_history, marks_future), (mask_history, mask_future) = (
            self.divide_history_and_future(input_time, input_marks, input_mask)
        )
        # ([batch_size, seq_len_h + 1], [batch_size, seq_len_x + 1]) * 3

        # Here, mask = 1: important. Removing them would cause counterfactual results.
        #       mask = 0: noises or unrelated marks. Keeping them makes no benefit for modeling the future.
        input_probability = self.model(
            marks_history, marks_future, time_history, time_future, mask_history, mask_future
        )
        # [batch_size, length_of_h + 1, 2]
        check_tensor(input_probability)

        # Tell the average gap between p(y = 1|x, H) and p(y = 0|x, H). Bigger probability gap means the model is more certain about the result.
        gap_between_p_1_and_p_0 = input_probability[:, :, 1] - input_probability[:, :, 0]
        # [batch_size, length_of_h + 1]
        gap_sum = torch.abs(gap_between_p_1_and_p_0.detach() * mask_history).sum()
        avg_gap_between_p_1_and_p_0 = gap_sum / mask_history.sum().item()

        repeated_input_probability = rearrange(input_probability, "... -> () ...")
        # [samples_for_l_e, batch_size, length_of_h + 1, 2]

        # Generate the history_mask filter_mask.
        # history_mask: show the event in history should go H_d or H_l.
        # filter_mask: do the same thing as history_mask but applied on the entire sequence.
        mask_on_history, mask_on_whole_seq = generate_masks(
            repeated_input_probability, self.seq_len_x + 1, evaluate=True
        )
        # [samples_for_l_e, batch_size, length_of_h + 1, 2] + [samples_for_l_e, batch_size, length_of_h + length_oF_x + 2, 2]

        # No.1 loss: L_n, optimize the length of essential marks.
        mask_on_history_without_dummy = mask_on_history[..., 1:, 1]
        # [samples_for_l_e, batch_size, length_of_h]
        loss_number_of_useful_events = (
            torch.linalg.norm(mask_on_history_without_dummy.float(), ord=1, dim=-1) / self.seq_len_h
        )
        # [num_of_samples_mask, batch_size]
        loss_number_of_useful_events = loss_number_of_useful_events.mean()

        # No.2 loss: L_e, optimize the quality of distilled marks.
        perplexity_p_h_f_x_os, perplexity_p_h_d_x_os, perplexity_p_h_l_x_os = (
            self.compute_perplexity_on_full_and_seg_sequences(
                input_marks, input_time, input_mask, mask_on_whole_seq, mean, std
            )
        )
        loss_how_useful_events_are = (
            F.relu(perplexity_p_h_f_x_os - perplexity_p_h_l_x_os + self.epsilon_l).mean()
            + F.relu(perplexity_p_h_d_x_os - perplexity_p_h_f_x_os + self.epsilon_d).mean()
        )

        loss = (
            self.param_number_useful_events * loss_number_of_useful_events
            + self.param_how_useful_events_are * loss_how_useful_events_are
        )

        """
        Evaluation part.
        """
        # How many marks are left in \history_l ?
        the_number_of_left_events = mask_on_history[..., 1:, 0].detach().sum(dim=-1).float().mean()

        return (
            loss,
            loss_number_of_useful_events,
            loss_how_useful_events_are,
            avg_gap_between_p_1_and_p_0,
            the_number_of_left_events,
        )

    def compute_perplexity_on_full_and_seg_sequences(
        self, input_marks, input_time, input_mask, mask_on_whole_seq, mean, std, training=False
    ):
        """
        Calculate L_rp and L_sp based on the given filter_mask.
        """
        batch_size = input_marks.shape[0]
        cum_input_time = input_time.cumsum(dim=-1)  # [batch_size, seq_len]

        marks_embeddings = self.mtpp_model("ehd_mark_emb", input_marks)  # [batch_size, seq_len, d_history]

        (
            (padded_distilled_marks, padded_distilled_masks),
            (padded_distilled_times, padded_distilled_mark_embeddings),
            (padded_left_marks, padded_left_masks),
            (padded_left_times, padded_left_mark_embeddings),
        ) = filter(
            (input_marks, input_mask), (cum_input_time, marks_embeddings), mask_on_whole_seq, evaluate=not training
        )

        # Loss 3 for asking the model to find the most important marks.
        # rebuild the original history for H_{o,t_l} - H_{s,o,t_l} based on history_mask.
        # You should be really careful to implement this part for not accidentally dropping any gradients.
        perplexity_p_h_f_x_o = self.mtpp_model(
            "ehd_perplexity",
            input_time,
            input_marks,
            marks_embeddings,
            input_mask,
            self.seq_len_x + 1,
            mean,
            std,
            training=training,
        )
        # [batch_size]
        perplexity_p_h_f_x_o = perplexity_p_h_f_x_o.unsqueeze(dim=0)
        # [num_of_samples_mask, batch_size]

        packed_data = zip(
            padded_distilled_marks,
            padded_distilled_masks,
            padded_distilled_times,
            padded_distilled_mark_embeddings,
            padded_left_marks,
            padded_left_masks,
            padded_left_times,
            padded_left_mark_embeddings,
        )

        start_val = torch.zeros(batch_size, 1, device=self.device)
        perplexity_p_h_ds_x_o = []
        perplexity_p_h_ls_x_o = []
        for (
            padded_distilled_marks_one_batch,
            padded_distilled_masks_one_batch,
            padded_distilled_times_one_batch,
            padded_distilled_mark_embeddings_one_batch,
            padded_left_marks_one_batch,
            padded_left_masks_one_batch,
            padded_left_times_one_batch,
            padded_left_mark_embeddings_one_batch,
        ) in packed_data:
            padded_distilled_times_one_batch = torch.diff(padded_distilled_times_one_batch, dim=-1, prepend=start_val)
            # [batch_size, seq_distilled_len]
            padded_left_times_one_batch = torch.diff(padded_left_times_one_batch, dim=-1, prepend=start_val)
            # [batch_size, seq_left_len]
            # Loss for asking the model to find the most important marks.
            # rebuild the original history for H_{o,t_l} - H_{s,o,t_l} based on history_mask.
            # You should be really careful to implement this part for not accidentally dropping any gradients.
            perplexity_p_h_ds_x_o.append(
                self.mtpp_model(
                    "ehd_perplexity",
                    padded_distilled_times_one_batch,
                    padded_distilled_marks_one_batch,
                    padded_distilled_mark_embeddings_one_batch,
                    padded_distilled_masks_one_batch,
                    self.seq_len_x + 1,
                    mean,
                    std,
                    training=training,
                )
            )  # [batch_size]
            perplexity_p_h_ls_x_o.append(
                self.mtpp_model(
                    "ehd_perplexity",
                    padded_left_times_one_batch,
                    padded_left_marks_one_batch,
                    padded_left_mark_embeddings_one_batch,
                    padded_left_masks_one_batch,
                    self.seq_len_x + 1,
                    mean,
                    std,
                    training=training,
                )
            )  # [batch_size]

        perplexity_p_h_ds_x_o = torch.stack(perplexity_p_h_ds_x_o, dim=0)  # [num_of_samples_mask, batch_size]
        perplexity_p_h_ls_x_o = torch.stack(perplexity_p_h_ls_x_o, dim=0)  # [num_of_samples_mask, batch_size]

        return perplexity_p_h_f_x_o, perplexity_p_h_ds_x_o, perplexity_p_h_ls_x_o

    def extract_minibatch(self, minibatch):
        """
        This function extracts input_time, input_marks, input_intensity, mask, mean, and std from the minibatch.
        Caution: dataloader won't add the end dummy mark during evaluation!

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
        input_time, input_marks, mask = minibatch[0]
        mean, std = minibatch[1]

        return input_time, input_marks, mask, mean, std

    def removed_marks(self, input_data, opt):
        """
        Since we removed all sequence shorter than seq_len_x + seq_len_h.
        We do not need to worry about the input_mask anymore.
        """

        """
        Extract data from the input minibatch.
        """
        input_time, input_marks, input_mask, mean, std = self.extract_minibatch(input_data)

        if not input_mask.all():
            raise Exception("All values in the input_mask should be true since we have nothing to mask.")
        (time_history, time_future), (marks_history, marks_future), (mask_history, mask_future) = (
            self.divide_history_and_future(input_time, input_marks, input_mask)
        )
        # ([batch_size, seq_len_h + 1], [batch_size, seq_len_x + 1]) * 3

        # Here, mask = 1: important. Removing them would cause counterfactual results.
        #       mask = 0: noises or unrelated marks. Keeping them makes no benefit for modeling the future.
        generated_mask_probability = self.model(
            time_history, time_future, marks_history, marks_future, mask_history, mask_future, mean, std
        )
        # [batch_size, seq_len_h + 1, 2]
        check_tensor(generated_mask_probability)

        # Since we don't need gradient during evaluation, we simply use argmax() here to generate history_mask.u
        history_mask = F.one_hot(torch.argmax(generated_mask_probability, dim=-1), num_classes=2)
        # [batch_size, seq_len_h + 1, 2]
        history_mask[:, 0] = 1
        check_tensor(history_mask)

        future_mask = torch.ones(*mask_future.shape, 2, device=self.device)  # [batch_size, seq_len_x + 1, 2]

        filter_mask, _ = pack((history_mask, future_mask), "b * m")  # [batch_size, seq_len_h + seq_len_x + 2, 2]
        filter_mask = repeat(filter_mask, "b l m -> n b l m", n=1)  # [1, batch_size, seq_len_h + seq_len_x + 2, 2]

        marks_embeddings = self.mtpp_model("ehd_mark_emb", input_marks)  # [batch_size, seq_len, d_history]
        padded_filtered_time, padded_filtered_marks, padded_filtered_mark_embeddings, padded_filtered_masks = (
            self.filter(
                input_time=input_time,
                input_marks=input_marks,
                marks_embeddings=marks_embeddings,
                input_mask=input_mask,
                filter_mask=filter_mask,
                evaluate=True,
            )
        )
        # [1, batch_size, seq_len_h + seq_len_x + 2] * 2 + [samples_for_l_e, batch_size, seq_len_h + seq_len_x + 2, d_history] + [samples_for_l_e, batch_size, seq_len_h + seq_len_x + 2]

        (
            padded_filtered_removed_time,
            padded_filtered_removed_marks,
            padded_filtered_mark_removed_embeddings,
            padded_filtered_removed_masks,
        ) = self.filter(
            input_time=input_time,
            input_marks=input_marks,
            marks_embeddings=marks_embeddings,
            input_mask=input_mask,
            filter_mask=filter_mask,
            evaluate=True,
            output_removed_marks=True,
        )
        # [1, batch_size, seq_len_h + seq_len_x + 2] * 2 + [samples_for_l_e, batch_size, seq_len_h + seq_len_x + 2, d_history] + [samples_for_l_e, batch_size, seq_len_h + seq_len_x + 2]

        # Loss 3 for asking the model to find the most important marks.
        # rebuild the original history for H_{o,t_l} - H_{s,o,t_l} based on history_mask.
        # You should be really careful to implement this part for not accidentally dropping any gradients.
        log_p_h_o_t_l_x_o_mean = self.mtpp_model(
            "ehd_perplexity", input_time, input_marks, marks_embeddings, input_mask, self.seq_len_x, mean, std
        )
        # [batch_size]

        log_p_h_r_o_t_l_x_o_mean = []
        for (
            padded_filtered_time_per_sample,
            padded_filtered_marks_per_sample,
            padded_filtered_mark_embeddings_per_sample,
            padded_filtered_masks_per_sample,
        ) in zip(padded_filtered_time, padded_filtered_marks, padded_filtered_mark_embeddings, padded_filtered_masks):
            log_p_h_r_o_t_l_x_o_mean.append(
                self.mtpp_model(
                    "ehd_perplexity",
                    padded_filtered_time_per_sample,
                    padded_filtered_marks_per_sample,
                    padded_filtered_mark_embeddings_per_sample,
                    padded_filtered_masks_per_sample,
                    self.seq_len_x,
                    mean,
                    std,
                )
            )
            # [batch_size]
        log_p_h_r_o_t_l_x_o_mean = torch.stack(log_p_h_r_o_t_l_x_o_mean, dim=0)
        # [1, batch_size]

        L_rp = (log_p_h_o_t_l_x_o_mean.unsqueeze(dim=0) - log_p_h_r_o_t_l_x_o_mean).mean().item()

        """
        Evaluation part.
        """
        # part 1: How many percents of marks are left?
        discrete_remained_mask = history_mask[..., 0].detach().int()
        the_number_of_remained_marks = discrete_remained_mask.sum(dim=-1)
        the_number_of_total_marks = mask_history.sum(dim=-1)
        percentage_remained_marks = (the_number_of_remained_marks / the_number_of_total_marks).mean().item()

        # part 2: What is the value of log_p_h_s_o_t_l_x_o_mean?
        log_p_h_s_o_t_l_x_o_mean = []
        for (
            padded_filtered_removed_time_per_sample,
            padded_filtered_removed_marks_per_sample,
            padded_filtered_removed_mark_embeddings_per_sample,
            padded_filtered_removed_masks_per_sample,
        ) in zip(
            padded_filtered_removed_time,
            padded_filtered_removed_marks,
            padded_filtered_mark_removed_embeddings,
            padded_filtered_removed_masks,
        ):
            log_p_h_s_o_t_l_x_o_mean.append(
                self.mtpp_model(
                    "ehd_perplexity",
                    padded_filtered_removed_time_per_sample,
                    padded_filtered_removed_marks_per_sample,
                    padded_filtered_removed_mark_embeddings_per_sample,
                    padded_filtered_removed_masks_per_sample,
                    self.seq_len_x,
                    mean,
                    std,
                )
            )
            # [batch_size]
        log_p_h_s_o_t_l_x_o_mean = torch.stack(log_p_h_s_o_t_l_x_o_mean, dim=0)
        # [1, batch_size]

        L_sp = (log_p_h_o_t_l_x_o_mean.unsqueeze(dim=0) - log_p_h_s_o_t_l_x_o_mean).mean().item()

        # Comparison with random removal.
        # we i.i.d. sample the mask multiple times to eliminate serendipity.
        number_of_sampled_sequence = 16
        input_time_random = repeat(input_time, "b ... -> (nss b) ...", nss=number_of_sampled_sequence)
        #
        input_marks_random = repeat(input_marks, "b ... -> (nss b) ...", nss=number_of_sampled_sequence)
        marks_embeddings_random = repeat(marks_embeddings, "b ... -> (nss b) ...", nss=number_of_sampled_sequence)
        input_mask_random = repeat(input_mask, "b ... -> (nss b) ...", nss=number_of_sampled_sequence)

        for the_number_of_remained_marks_per_seq in the_number_of_remained_marks:
            rand_mat = torch.rand(number_of_sampled_sequence, self.seq_len_h, device=self.device)
            # [number_of_sampled_sequence, seq_len_h]
            k_th_quant = torch.topk(rand_mat, the_number_of_remained_marks_per_seq - 1, largest=False)[0][:, -1:]
            # [number_of_sampled_sequence, 1]
            if the_number_of_remained_marks_per_seq == 1:
                mask = torch.ones_like(rand_mat, device=self.device).long()  # [number_of_sampled_sequence, seq_len_h]
            else:
                mask = (rand_mat > k_th_quant).long()  # [number_of_sampled_sequence, seq_len_h]
            generated_mask_probability_random = F.one_hot(mask, num_classes=2)
            # [number_of_sampled_sequence, seq_len_h, 2]
            check_tensor(generated_mask_probability_random)

            # Since we don't need gradient during evaluation, we simply use argmax() here to generate history_mask.
            history_mask_random = F.one_hot(torch.argmax(generated_mask_probability_random, dim=-1), num_classes=2)
            # [number_of_sampled_sequence, seq_len_h, 2]
            history_mask_random, _ = pack(
                (torch.ones(number_of_sampled_sequence, 1, 2, device=self.device), history_mask_random), "nss * m"
            )
            # [number_of_sampled_sequence, seq_len_h, 2]
            check_tensor(history_mask_random)

            future_mask_random = torch.ones(number_of_sampled_sequence, self.seq_len_x + 1, 2, device=self.device)
            # [number_of_sampled_sequence, seq_len_x + 1, 2]

            filter_mask_random, _ = pack((history_mask_random, future_mask_random), "nss * m")
            # [number_of_sampled_sequence, seq_len_h + seq_len_x + 2, 2]
            filter_mask_random = repeat(filter_mask_random, "b l m -> n b l m", n=1)
            # [1, number_of_sampled_sequence, seq_len_h + seq_len_x + 2, 2]

            marks_embeddings = self.mtpp_model(
                "ehd_mark_emb", input_marks
            )  # [number_of_sampled_sequence, seq_len, d_history]
            (
                padded_filtered_time_random,
                padded_filtered_marks_random,
                padded_filtered_mark_embeddings_random,
                padded_filtered_masks_random,
            ) = self.filter(
                input_time=input_time_random,
                input_marks=input_marks_random,
                marks_embeddings=marks_embeddings_random,
                input_mask=input_mask_random,
                filter_mask=filter_mask_random,
                evaluate=True,
            )
            # [1, number_of_sampled_sequence, seq_len_h + seq_len_x + 2] * 2 + [samples_for_l_e, number_of_sampled_sequence, seq_len_h + seq_len_x + 2, d_history] + [samples_for_l_e, number_of_sampled_sequence, seq_len_h + seq_len_x + 2]

            (
                padded_filtered_removed_time_random,
                padded_filtered_removed_marks_random,
                padded_filtered_mark_removed_embeddings_random,
                padded_filtered_removed_masks_random,
            ) = self.filter(
                input_time=input_time_random,
                input_marks=input_marks_random,
                marks_embeddings=marks_embeddings_random,
                input_mask=input_mask_random,
                filter_mask=filter_mask_random,
                evaluate=True,
                output_removed_marks=True,
            )
            # [1, number_of_sampled_sequence, seq_len_h + seq_len_x + 2] * 2 + [samples_for_l_e, number_of_sampled_sequence, seq_len_h + seq_len_x + 2, d_history] + [samples_for_l_e, number_of_sampled_sequence, seq_len_h + seq_len_x + 2]

            # Loss 3 for asking the model to find the most important marks.
            # rebuild the original history for H_{o,t_l} - H_{s,o,t_l} based on history_mask.
            # You should be really careful to implement this part for not accidentally dropping any gradients.
            log_p_h_o_t_l_x_o_mean = self.mtpp_model(
                "ehd_perplexity", input_time, input_marks, marks_embeddings, input_mask, self.seq_len_x, mean, std
            )
            # [number_of_sampled_sequence]

            log_p_h_r_o_t_l_x_o_mean_random = []
            for (
                padded_filtered_time_per_sample,
                padded_filtered_marks_per_sample,
                padded_filtered_mark_embeddings_per_sample,
                padded_filtered_masks_per_sample,
            ) in zip(
                padded_filtered_time_random,
                padded_filtered_marks_random,
                padded_filtered_mark_embeddings_random,
                padded_filtered_masks_random,
            ):
                log_p_h_r_o_t_l_x_o_mean_random.append(
                    self.mtpp_model(
                        "ehd_perplexity",
                        padded_filtered_time_per_sample,
                        padded_filtered_marks_per_sample,
                        padded_filtered_mark_embeddings_per_sample,
                        padded_filtered_masks_per_sample,
                        self.seq_len_x,
                        mean,
                        std,
                    )
                )
                # [number_of_sampled_sequence]
            log_p_h_r_o_t_l_x_o_mean_random = torch.stack(log_p_h_r_o_t_l_x_o_mean_random, dim=0)
            # [1, number_of_sampled_sequence]

            L_rp_r = (log_p_h_o_t_l_x_o_mean.unsqueeze(dim=0) - log_p_h_r_o_t_l_x_o_mean_random).mean().item()

            """
            Evaluation part of EHD_random
            """
            # part 1: What is the value of log_p_h_s_o_t_l_x_o_mean?
            log_p_h_s_o_t_l_x_o_mean_random = []
            for (
                padded_filtered_removed_time_per_sample,
                padded_filtered_removed_marks_per_sample,
                padded_filtered_removed_mark_embeddings_per_sample,
                padded_filtered_removed_masks_per_sample,
            ) in zip(
                padded_filtered_removed_time_random,
                padded_filtered_removed_marks_random,
                padded_filtered_mark_removed_embeddings_random,
                padded_filtered_removed_masks_random,
            ):
                log_p_h_s_o_t_l_x_o_mean_random.append(
                    self.mtpp_model(
                        "ehd_perplexity",
                        padded_filtered_removed_time_per_sample,
                        padded_filtered_removed_marks_per_sample,
                        padded_filtered_removed_mark_embeddings_per_sample,
                        padded_filtered_removed_masks_per_sample,
                        self.seq_len_x,
                        mean,
                        std,
                    )
                )
                # [number_of_sampled_sequence]
            log_p_h_s_o_t_l_x_o_mean_random = torch.stack(log_p_h_s_o_t_l_x_o_mean_random, dim=0)
            # [1, number_of_sampled_sequence]

            L_sp_r = (log_p_h_o_t_l_x_o_mean.unsqueeze(dim=0) - log_p_h_s_o_t_l_x_o_mean_random).mean().item()

        data = {
            "percentage_remained_marks": percentage_remained_marks,
            "generated_mask_probability": generated_mask_probability.detach().cpu(),
            "L_sp": L_sp,
            "L_sp_r": L_sp_r,
            "L_rp": L_rp,
            "L_rp_r": L_rp_r,
            "marks_history": marks_history.detach().cpu(),
            "marks_future": marks_future.detach().cpu(),
            "time_history": time_history.detach().cpu(),
            "time_future": time_future.detach().cpu(),
            "filter_mask": filter_mask.squeeze(dim=0).detach().cpu(),
        }

        plots = plot_removed_marks(data, opt)

        return plots

    def get_explanation(self, input_data, opt):
        input_time, input_marks, input_mask, mean, std = self.extract_minibatch(input_data)
        batch_size = input_time.shape[0]

        (time_history, time_future), (marks_history, marks_future), (mask_history, mask_future) = (
            self.divide_history_and_future(input_time, input_marks, input_mask)
        )
        # ([batch_size, seq_len_h + 1], [batch_size, seq_len_x + 1]) * 3

        # Here, mask = 1: important. Removing them would cause counterfactual results.
        #       mask = 0: noises or unrelated marks. Keeping them makes no benefit for modeling the future.
        input_probability = self.model(
            marks_history, marks_future, time_history, time_future, mask_history, mask_future
        )
        # [batch_size, length_of_h + 1, 2]
        check_tensor(input_probability)

        probability_of_explanation = input_probability[..., 1:, 1]
        sorted_index = torch.argsort(probability_of_explanation, descending=True)  # [batch_size, seq_len_h]

        masked_events = range(1, self.seq_len_h + 1)  # [batch_size, seq_len_h]
        seq_masks = []

        # Initial state
        mask = torch.zeros(batch_size, self.seq_len_h + self.seq_len_x, device=self.device)
        # [batch_size, seq_len_h + self.seq_len_x]
        generated_mask_probability = F.one_hot(mask.to(torch.int64), num_classes=2).float()
        # [batch_size, seq_len_h + self.seq_len_x, 2]
        generated_mask_probability[..., -self.seq_len_x :, :] = 1
        seq_mask, _ = pack(
            (
                torch.ones((batch_size, 1, 2), device=self.device),
                generated_mask_probability,
                torch.ones((batch_size, 1, 2), device=self.device),
            ),
            "bs * m",
        )
        # [batch_size, seq_len_h + seq_len_x + 2, 2]
        seq_masks.append(seq_mask)

        for the_number_of_masked_events in masked_events:
            selected_index = sorted_index[..., the_number_of_masked_events - 1]  # [batch_size]
            generated_mask_probability[torch.arange(batch_size, device=self.device), selected_index] = torch.tensor(
                [[0.0, 1.0] * batch_size], device=self.device
            )
            # [batch_size, seq_len_h + seq_len_x, 2]
            seq_mask, _ = pack(
                (
                    torch.ones((batch_size, 1, 2), device=self.device),
                    generated_mask_probability,
                    torch.ones((batch_size, 1, 2), device=self.device),
                ),
                "bs * m",
            )
            seq_masks.append(seq_mask)

        seq_masks = torch.stack(seq_masks, dim=0)  # [seq_len_h, batch_size, seq_len_h + seq_len_x + 2, 2]

        perplexity_p_h_f_x_o, perplexity_p_h_ds_x_o, perplexity_p_h_ls_x_o = (
            self.compute_perplexity_on_full_and_seg_sequences(input_marks, input_time, input_mask, seq_masks, mean, std)
        )
        # [seq_len_h, batch_size]

        perplexity_gap_between_left_and_full = perplexity_p_h_f_x_o - perplexity_p_h_ls_x_o  # [seq_len_h, batch_size]
        perplexity_gap_between_distilled_and_full = (
            perplexity_p_h_f_x_o - perplexity_p_h_ds_x_o
        )  # [seq_len_h, batch_size]

        perplexity_gap_between_left_and_full, perplexity_gap_between_distilled_and_full = move_from_tensor_to_ndarray(
            perplexity_gap_between_left_and_full, perplexity_gap_between_distilled_and_full
        )

        index_fits_left_constraint = np.where(perplexity_gap_between_left_and_full < self.epsilon_l)[0]
        index_fits_distill_constraint = np.where(perplexity_gap_between_distilled_and_full > self.epsilon_d)[0]

        picked_min_left = self.seq_len_h if len(index_fits_left_constraint) == 0 else index_fits_left_constraint.min()
        picked_min_distill = (
            self.seq_len_h if len(index_fits_distill_constraint) == 0 else index_fits_distill_constraint.min()
        )
        picked_index = max(picked_min_left, picked_min_distill)

        picked_seq_mask = seq_masks[picked_index]
        picked_perplexity_p_h_d_x_o = perplexity_p_h_ds_x_o[picked_index]
        picked_perplexity_p_h_l_x_o = perplexity_p_h_ls_x_o[picked_index]
        picked_perplexity_gap_between_left_and_full = perplexity_gap_between_left_and_full[picked_index]
        picked_perplexity_gap_between_distilled_and_full = perplexity_gap_between_distilled_and_full[picked_index]
        the_number_of_remained_events = picked_seq_mask[..., 1].sum() - 2 - self.seq_len_x

        return (
            the_number_of_remained_events.item(),
            picked_perplexity_p_h_d_x_o.item(),
            picked_perplexity_gap_between_distilled_and_full.item(),
            picked_perplexity_p_h_l_x_o.item(),
            picked_perplexity_gap_between_left_and_full.item()
        )

    def get_unhinged_perplexity_gap(self, input_data, opt):
        input_time, input_marks, input_mask, mean, std = self.extract_minibatch(input_data)

        if not input_mask.all():
            raise Exception("All values in the input_mask should be true since we have nothing to mask.")
        (time_history, time_future), (marks_history, marks_future), (mask_history, mask_future) = (
            self.divide_history_and_future(input_time, input_marks, input_mask)
        )
        # ([batch_size, seq_len_h + 1], [batch_size, seq_len_x + 1]) * 3

        number_of_marks = mask_history.sum().item()
        marks_embeddings = self.mtpp_model("ehd_mark_emb", input_marks)  # [batch_size, seq_len, d_history]
        cum_input_time = input_time.cumsum(dim=-1)  # [batch_size, seq_len]
        batch_size = input_time.shape[0]

        # Here, mask = 1: important. Removing them would cause counterfactual results.
        #       mask = 0: noises or unrelated marks. Keeping them makes no benefit for modeling the future.
        input_probability = self.model(
            marks_history, marks_future, time_history, time_future, mask_history, mask_future
        )
        # [batch_size, length_of_h + 1, 2]
        check_tensor(input_probability)

        repeated_input_probability = rearrange(input_probability, "... -> () ...")
        # [samples_for_l_e, batch_size, length_of_h + 1, 2]
        """
        Generate the history mask and corresponding filter mask.
        """
        history_mask, filter_mask = generate_masks(repeated_input_probability, self.seq_len_x, evaluate=True)
        # [samples_for_l_e, batch_size, length_of_h + 1, 2] + [samples_for_l_e, batch_size, length_of_h + length_oF_x + 2, 2]
        """
        L_n, optimize the length of essential marks.
        """
        selected_mask = history_mask[..., 1:, 1]
        L_n = torch.linalg.norm(selected_mask.float(), ord=1, dim=-1) / (self.seq_len_h - 1)
        # [num_of_samples_mask, batch_size]
        L_n = L_n.mean()

        """
        L_e, optimize the quality of distilled marks.
        """
        (
            (padded_distilled_marks, padded_distilled_masks),
            (padded_distilled_times, padded_distilled_mark_embeddings),
            (padded_left_marks, padded_left_masks),
            (padded_left_times, padded_left_mark_embeddings),
        ) = filter((input_marks, input_mask), (cum_input_time, marks_embeddings), filter_mask, evaluate=True)
        # num_of_samples_mask * [batch_size, seq_len, ...]
        log_p_h_f_x_o = self.mtpp_model(
            "ehd_perplexity", input_time, input_marks, marks_embeddings, input_mask, self.seq_len_x, mean, std
        )  # [batch_size]
        packed_data = zip(
            padded_distilled_marks,
            padded_distilled_masks,
            padded_distilled_times,
            padded_distilled_mark_embeddings,
            padded_left_marks,
            padded_left_masks,
            padded_left_times,
            padded_left_mark_embeddings,
        )
        L_l_without_hinge, L_d_without_hinge = 0, 0
        for (
            padded_distilled_marks_one_batch,
            padded_distilled_masks_one_batch,
            padded_distilled_times_one_batch,
            padded_distilled_mark_embeddings_one_batch,
            padded_left_marks_one_batch,
            padded_left_masks_one_batch,
            padded_left_times_one_batch,
            padded_left_mark_embeddings_one_batch,
        ) in packed_data:
            padded_distilled_times_one_batch = torch.diff(
                padded_distilled_times_one_batch, dim=-1, prepend=torch.zeros(batch_size, 1, device=self.device)
            )
            # [batch_size, seq_distilled_len]
            padded_left_times_one_batch = torch.diff(
                padded_left_times_one_batch, dim=-1, prepend=torch.zeros(batch_size, 1, device=self.device)
            )
            # [batch_size, seq_left_len]
            # Loss for asking the model to find the most important marks.
            # rebuild the original history for H_{o,t_l} - H_{s,o,t_l} based on history_mask.
            # You should be really careful to implement this part for not accidentally dropping any gradients.
            log_p_h_d_x_o = self.mtpp_model(
                "ehd_perplexity",
                padded_distilled_times_one_batch,
                padded_distilled_marks_one_batch,
                padded_distilled_mark_embeddings_one_batch,
                padded_distilled_masks_one_batch,
                self.seq_len_x,
                mean,
                std,
            )  # [batch_size]
            log_p_h_l_x_o = self.mtpp_model(
                "ehd_perplexity",
                padded_left_times_one_batch,
                padded_left_marks_one_batch,
                padded_left_mark_embeddings_one_batch,
                padded_left_masks_one_batch,
                self.seq_len_x,
                mean,
                std,
            )  # [batch_size]

            L_l_without_hinge = (log_p_h_f_x_o - log_p_h_l_x_o).mean()
            L_d_without_hinge = (log_p_h_f_x_o - log_p_h_d_x_o).mean()

        the_number_of_left_events = history_mask[..., 1:, 0].detach().sum(dim=-1).float().mean()

        return the_number_of_left_events, L_l_without_hinge, L_d_without_hinge

    def lsp_and_lrp_compared_with_gs_and_rd(self, input_data, opt):
        """
        Since we removed all sequence shorter than seq_len_x + seq_len_h.
        We do not need to worry about the input_mask anymore.

        Update: we merge ehd_random here because ehd_random should remove the same or close amount of
        mark from the original sequence. Setting up a dedicated ehd_random module won't work because the
        number of removed mark stdies with the sequence length.
        """
        import time

        """
        Extract data from the input minibatch.
        """
        input_time, input_marks, input_mask, mean, std = self.extract_minibatch(input_data)

        assert (input_mask == 1).all()
        (time_history, time_future), (marks_history, marks_future), (mask_history, mask_future) = (
            self.divide_history_and_future(input_time, input_marks, input_mask)
        )
        # ([batch_size, seq_len_h + 1], [batch_size, seq_len_x + 1]) * 3
        """
        Evaluation part.
        """
        # part 1: How many percents of marks are left?
        start = time.time()
        percentage_remained_marks, L_sp, L_rp = self.evaluate_procedure(
            input_time, input_marks, input_mask, mean, std, percentage=False
        )[-3:]
        end = time.time()
        time_ehd_mtpp = end - start
        the_number_of_remained_marks = percentage_remained_marks

        percentage_remained_marks = percentage_remained_marks.float().mean().item()
        L_sp = L_sp.item()
        L_rp = L_rp.item()
        the_number_of_total_marks = mask_history.sum(dim=-1)

        # Comparison with random removal.
        # we i.i.d. sample the mask multiple times to eliminate serendipity.
        start = time.time()
        number_of_sampled_sequence = 16
        for the_number_of_remained_marks_per_seq, the_number_of_historical_marks_per_seq in zip(
            the_number_of_remained_marks, mask_history.sum(dim=-1)
        ):
            # baseline 1, random removal.
            rand_mat = torch.rand(number_of_sampled_sequence, self.seq_len_h, device=self.device)
            # [number_of_sampled_sequence, seq_len_h]
            k_th_quant = torch.topk(rand_mat, the_number_of_remained_marks_per_seq - 1, largest=False)[0][:, -1:]
            # [number_of_sampled_sequence, 1]
            if the_number_of_remained_marks_per_seq == 1:
                mask = torch.ones_like(rand_mat, device=self.device).long()  # [number_of_sampled_sequence, seq_len_h]
            else:
                mask = (rand_mat > k_th_quant).long()  # [number_of_sampled_sequence, seq_len_h]
            generated_mask_probability_random = F.one_hot(mask, num_classes=2)
            # [number_of_sampled_sequence, seq_len_h, 2]
            check_tensor(generated_mask_probability_random)

            # Since we don't need gradient during evaluation, we simply use argmax() here to generate history_mask.
            history_mask_random = F.one_hot(torch.argmax(generated_mask_probability_random, dim=-1), num_classes=2)
            # [number_of_sampled_sequence, seq_len_h, 2]
            history_mask_random, _ = pack(
                (torch.ones(number_of_sampled_sequence, 1, 2, device=self.device), history_mask_random), "nss * m"
            )
            # [number_of_sampled_sequence, seq_len_h, 2]
            check_tensor(history_mask_random)

            future_mask_random = torch.ones(number_of_sampled_sequence, self.seq_len_x + 1, 2, device=self.device)
            # [number_of_sampled_sequence, seq_len_x + 1, 2]

            filter_mask_random, _ = pack((history_mask_random, future_mask_random), "nss * m")
            # [number_of_sampled_sequence, seq_len_h + seq_len_x + 2, 2]
            filter_mask_random = repeat(filter_mask_random, "n l m -> n b l m", b=1)
            # [number_of_sampled_sequence, batch_size, seq_len_h + seq_len_x + 2, 2]

            L_sp_r, L_rp_r = self.get_metric_values(input_marks, input_time, input_mask, filter_mask_random, mean, std)

            mask = torch.zeros_like(mask_history, device=self.device) * mask_history
            # [batch_size, seq_len_h + 1]
        end = time.time()
        time_baseline_1_given_percentage = end - start
        time_baseline_1_given_percentage_to_ehd = time_baseline_1_given_percentage / time_ehd_mtpp
        L_sp_g1, L_rp_g1 = 0, 0

        start = time.time()
        mask = torch.zeros_like(mask_history) * mask_history  # [batch_size, seq_len_h + 1]
        for the_number_of_remained_marks_per_seq, the_number_of_historical_marks_per_seq in zip(
            the_number_of_remained_marks, mask_history.sum(dim=-1)
        ):
            # The first mark is always included.
            mask[:, 0] = 1  # [batch_size, seq_len_h + 1]
            full_mask = torch.ones_like(mask_history)  # [batch_size, seq_len_h + 1]
            seq_len_h_1 = full_mask.sum(dim=-1)  # [batch_size]
            masked_marks = 1

            initial_history_mask = F.one_hot(mask.long(), num_classes=2)  # [batch_size, seq_len_h, 2]
            check_tensor(initial_history_mask)
            future_mask = torch.ones(initial_history_mask.shape[0], self.seq_len_x + 1, 2, device=self.device)
            # [batch_size, seq_len_x + 1, 2]
            initial_filter_mask, _ = pack((initial_history_mask, future_mask), "b * m")
            # [batch_size, seq_len_h + seq_len_x + 2, 2]
            initial_filter_mask = rearrange(initial_filter_mask, "b l m -> () b l m")
            # [number_of_sampled_sequence, batch_size, seq_len_h + seq_len_x + 2, 2]
            selected_L_sp_given_marks, selected_L_rp_given_marks = self.get_metric_values(
                input_marks, input_time, input_mask, initial_filter_mask, mean, std
            )

            while masked_marks < the_number_of_historical_marks_per_seq - the_number_of_remained_marks_per_seq:
                number_of_masked_marks = mask.sum(dim=-1)  # [batch_size]
                generated_mask = repeat(mask, "b s -> nss b s", nss=seq_len_h_1)
                # [seq_len_h + 1, batch_size, seq_len_h + 1]
                added_mask = torch.diag_embed(torch.ones(seq_len_h_1, device=self.device)).unsqueeze(dim=1)
                # [seq_len_h + 1, 1, seq_len_h + 1]
                generated_mask = generated_mask.long() | added_mask.long()  # [seq_len_h + 1, batch_size, seq_len_h + 1]
                generated_mask = generated_mask[generated_mask.sum(dim=-1) != number_of_masked_marks]
                # [seq_len_h + 1 - number_of_masked_marks, seq_len_h + 1]

                history_mask_random = F.one_hot(generated_mask, num_classes=2)
                # [number_of_sampled_sequence, seq_len_h, 2]
                check_tensor(history_mask_random)

                future_mask_random = torch.ones(generated_mask.shape[0], self.seq_len_x + 1, 2, device=self.device)
                # [number_of_sampled_sequence, seq_len_x + 1, 2]

                filter_mask_random, _ = pack((history_mask_random, future_mask_random), "nss * m")
                # [number_of_sampled_sequence, seq_len_h + seq_len_x + 2, 2]
                filter_mask_random = repeat(filter_mask_random, "n l m -> n b l m", b=1)
                # [number_of_sampled_sequence, batch_size, seq_len_h + seq_len_x + 2, 2]

                L_sp_r_d, L_rp_r_d = self.get_metric_values(
                    input_marks, input_time, input_mask, filter_mask_random, mean, std, return_mean=False
                )
                # [number_of_sampled_sequence]

                selected_index = torch.argmin(L_rp_r_d)
                selected_L_sp_given_marks = L_sp_r_d[selected_index].item()
                selected_L_rp_given_marks = L_rp_r_d[selected_index].item()
                selected_mask = generated_mask[selected_index]  # [batch_size, seq_len_h + seq_len_x + 2, 2]

                mask = selected_mask.unsqueeze(dim=0)  # [batch_size, seq_len_h + seq_len_x + 2]
                masked_marks += 1

            greedy_remained_marks = (the_number_of_total_marks - mask.sum(dim=1)).item()

        end = time.time()
        time_greedy_given_percentage = end - start

        # detect where the random removal's performance could reach EHD's performance?
        # Caution: only works when batch_size = 1
        start = time.time()
        random_remained_marks = copy.deepcopy(the_number_of_total_marks)
        while True:
            if (random_remained_marks == 0).any():
                break

            for the_number_of_remained_marks_per_seq in random_remained_marks:
                # baseline 1, random removal.
                rand_mat = torch.rand(number_of_sampled_sequence, self.seq_len_h, device=self.device)
                # [number_of_sampled_sequence, seq_len_h]
                k_th_quant = torch.topk(rand_mat, the_number_of_remained_marks_per_seq - 1, largest=False)[0][:, -1:]
                # [number_of_sampled_sequence, 1]
                if the_number_of_remained_marks_per_seq == 1:
                    mask = torch.ones_like(rand_mat, device=self.device).long()
                else:
                    mask = (rand_mat > k_th_quant).long()  # [number_of_sampled_sequence, seq_len_h]
                generated_mask_probability_random = F.one_hot(mask, num_classes=2)
                # [number_of_sampled_sequence, seq_len_h, 2]
                check_tensor(generated_mask_probability_random)

                # Since we don't need gradient during evaluation, we simply use argmax() here to generate history_mask.
                history_mask_random = F.one_hot(torch.argmax(generated_mask_probability_random, dim=-1), num_classes=2)
                # [number_of_sampled_sequence, seq_len_h, 2]
                history_mask_random, _ = pack(
                    (torch.ones(number_of_sampled_sequence, 1, 2, device=self.device), history_mask_random), "nss * m"
                )
                # [number_of_sampled_sequence, seq_len_h, 2]
                check_tensor(history_mask_random)

                future_mask_random = torch.ones(number_of_sampled_sequence, self.seq_len_x + 1, 2, device=self.device)
                # [number_of_sampled_sequence, seq_len_x + 1, 2]

                filter_mask_random, _ = pack((history_mask_random, future_mask_random), "nss * m")
                # [number_of_sampled_sequence, seq_len_h + seq_len_x + 2, 2]
                filter_mask_random = repeat(filter_mask_random, "n l m -> n b l m", b=1)
                # [number_of_sampled_sequence, batch_size, seq_len_h + seq_len_x + 2, 2]

                L_sp_r_d, L_rp_r_d = self.get_metric_values(
                    input_marks, input_time, input_mask, filter_mask_random, mean, std
                )

            if L_rp_r_d < L_rp and L_sp_r_d > L_sp:
                break
            else:
                random_remained_marks = random_remained_marks - 1
        end = time.time()
        time_baseline_1 = end - start
        random_remained_marks = random_remained_marks.item()

        # baseline 2, greedy: remove historical marks until it meets EHD's performance.
        # Each time we remove the mark which perform highest reduction to the probability.
        # Only works when batch_size = 1
        start = time.time()
        mask = torch.zeros_like(mask_history, device=self.device) * mask_history
        # [batch_size, seq_len_h + 1]
        # The first mark is always included.
        mask[:, 0] = 1  # [batch_size, seq_len_h + 1]
        full_mask = torch.ones_like(mask_history, device=self.device)  # [batch_size, seq_len_h + 1]
        seq_len_h_1 = full_mask.sum(dim=-1)  # [batch_size]

        while True:
            # Break when the mask equals to the full_mask, meaning that no mark is left.
            if (mask == full_mask).all():
                break

            number_of_masked_marks = mask.sum(dim=-1)  # [batch_size]
            generated_mask = repeat(
                mask, "b s -> nss b s", nss=seq_len_h_1
            )  # [seq_len_h + 1, batch_size, seq_len_h + 1]
            added_mask = torch.diag_embed(torch.ones(seq_len_h_1, device=self.device)).unsqueeze(dim=1)
            # [seq_len_h + 1, 1, seq_len_h + 1]
            generated_mask = generated_mask.long() | added_mask.long()  # [seq_len_h + 1, batch_size, seq_len_h + 1]
            generated_mask = generated_mask[generated_mask.sum(dim=-1) != number_of_masked_marks]
            # [seq_len_h + 1 - number_of_masked_marks, seq_len_h + 1]

            history_mask_random = F.one_hot(generated_mask, num_classes=2)  # [number_of_sampled_sequence, seq_len_h, 2]
            check_tensor(history_mask_random)

            future_mask_random = torch.ones(generated_mask.shape[0], self.seq_len_x + 1, 2, device=self.device)
            # [number_of_sampled_sequence, seq_len_x + 1, 2]

            filter_mask_random, _ = pack((history_mask_random, future_mask_random), "nss * m")
            # [number_of_sampled_sequence, seq_len_h + seq_len_x + 2, 2]
            filter_mask_random = repeat(filter_mask_random, "n l m -> n b l m", b=1)
            # [number_of_sampled_sequence, batch_size, seq_len_h + seq_len_x + 2, 2]

            L_sp_r_d, L_rp_r_d = self.get_metric_values(
                input_marks, input_time, input_mask, filter_mask_random, mean, std, return_mean=False
            )
            # [number_of_sampled_sequence]

            selected_index = torch.argmin(L_rp_r_d)
            selected_L_sp = L_sp_r_d[selected_index]
            selected_L_rp = L_rp_r_d[selected_index]
            selected_mask = generated_mask[selected_index]  # [batch_size, seq_len_h + seq_len_x + 2, 2]

            if selected_L_rp < L_rp and selected_L_sp > L_sp:
                break
            else:
                mask = selected_mask.unsqueeze(dim=0)  # [batch_size, seq_len_h + seq_len_x + 2]

        greedy_remained_marks = (the_number_of_total_marks - mask.sum(dim=1)).item()
        end = time.time()
        time_baseline_2 = end - start

        # time evaluation
        time_greedy_given_percentage_to_ehd = time_greedy_given_percentage / time_ehd_mtpp
        time_baseline_1_given_percentage_to_ehd = time_baseline_1_given_percentage / time_ehd_mtpp
        time_baseline_1_to_ehd = time_baseline_1 / time_ehd_mtpp
        time_baseline_2_to_ehd = time_baseline_2 / time_ehd_mtpp

        return (
            percentage_remained_marks,
            random_remained_marks,
            greedy_remained_marks,
            L_sp,
            L_sp_r,
            L_sp_g1,
            selected_L_sp_given_marks,
            L_rp,
            L_rp_r,
            L_rp_g1,
            selected_L_rp_given_marks,
            time_baseline_1_given_percentage_to_ehd,
            time_baseline_1_to_ehd,
            time_baseline_2_to_ehd,
            time_greedy_given_percentage_to_ehd,
        )

    def get_unhinged_perplexity_gap_full_curve(self, input_data, opt):
        """
        Given the number of distilled marks, this function will sort the probability then assign 1 to marks with the top-N highest probability.
        Because the theoretical best is nearly impossible to calculate for the insanely huge search space, we only expect to perform comparison on
        several selected sequences.
        """
        input_time, input_marks, _, input_mask, _, _, _, _, mean, var = self.extract_minibatch(input_data)

        assert (input_mask == 1).all()
        (time_history, time_future), (marks_history, marks_future), (mask_history, mask_future) = (
            self.divide_history_and_future(input_time, input_marks, input_mask)
        )
        # ([batch_size, seq_len_h + 1], [batch_size, seq_len_x + 1]) * 3

        # Here, mask = 1: important. Removing them would cause counterfactual results.
        #       mask = 0: noises or unrelated marks. Keeping them makes no benefit for modeling the future.
        generated_mask_probability = self.model(
            time_history, time_future, marks_history, marks_future, mask_history, mask_future, mean, var
        )
        # [batch_size, seq_len_h + 1, 2]
        batch_size = generated_mask_probability.shape[0]
        check_tensor(generated_mask_probability)
        probability_of_distilled = generated_mask_probability[..., 1:, 1]  # [batch_size, seq_len_h]
        sorted_index = torch.argsort(probability_of_distilled)  # [batch_size, seq_len_h]
        sorted_index = torch.cat((torch.zeros(batch_size, 1), sorted_index + 1), dim=-1)
        # [batch_size, seq_len_h + 1]
        # Generate the history_mask based on sorted_index.
        history_mask = torch.zeros((batch_size, self.seq_len_h, self.seq_len_h), device=self.device)
        # [batch_size, number_of_sampled_sequence, seq_len_h]
        for batch_idx in range(batch_size):
            for diagonal in range(-self.seq_len_h + 1, self.seq_len_h):
                history_mask[batch_idx] += torch.diagnal(
                    sorted_index[batch_idx][: self.seq_len_h - 1 - abs(diagonal)], diagonal=diagonal
                )

        history_mask = rearrange(
            history_mask, "b nss sqh -> nss b sqh"
        )  # [number_of_sampled_sequence, batch_size, seq_len_h]

        the_number_of_remained_marks = range(1, self.seq_len_h + 1)  # [batch_size, seq_len_h]
        all_mask = []
        gap = []
        L_sp_model = []
        L_rp_model = []

        # Initial state
        mask = torch.ones(batch_size, self.seq_len_h, device=self.device)  # [batch_size, seq_len_h]
        generated_mask_probability = F.one_hot(mask.to(torch.int64), num_classes=2)
        # [batch_size, seq_len_h, 2]
        history_mask = F.one_hot(torch.argmax(generated_mask_probability, dim=-1), num_classes=2)
        # [batch_size, seq_len_h, 2]
        history_mask, _ = pack((torch.ones(batch_size, 1, 2, device=self.device), history_mask), "bs * m")
        # [batch_size, seq_len_h, 2]
        future_mask = torch.ones(batch_size, self.seq_len_x + 1, 2, device=self.device)
        # [batch_size, seq_len_x + 1, 2]
        filter_mask, _ = pack((history_mask, future_mask), "bs * m")  # [batch_size, seq_len_h + seq_len_x + 2, 2]
        filter_mask = repeat(
            filter_mask, "b l m -> n b l m", n=1
        )  # [number_of_sampled_sequence, batch_size, seq_len_h + seq_len_x + 2, 2]

        L_sp_m, L_rp_m = self.get_metric_values(input_marks, input_time, input_mask, filter_mask, mean, var)
        all_mask.append(filter_mask.tolist())
        gap.append(L_sp_m - L_rp_m)
        L_sp_model.append(L_sp_m)
        L_rp_model.append(L_rp_m)

        for the_number_of_remained_marks_per_seq in the_number_of_remained_marks:
            mask = torch.zeros(batch_size, self.seq_len_h, device=self.device)
            # [batch_size, seq_len_h]
            selected_index = sorted_index[..., the_number_of_remained_marks_per_seq:]
            # [batch_size, seq_len_h]
            mask.scatter_(dim=-1, index=selected_index, src=torch.ones_like(mask))
            # [batch_size, seq_len_h]
            generated_mask_probability = F.one_hot(mask.to(torch.int64), num_classes=2)
            # [batch_size, seq_len_h, 2]

            # Since we don't need gradient during evaluation, we simply use argmax() here to generate history_mask.
            history_mask = F.one_hot(torch.argmax(generated_mask_probability, dim=-1), num_classes=2)
            # [batch_size, seq_len_h, 2]
            history_mask, _ = pack((torch.ones(batch_size, 1, 2, device=self.device), history_mask), "bs * m")
            # [batch_size, seq_len_h, 2]
            future_mask = torch.ones(batch_size, self.seq_len_x + 1, 2, device=self.device)
            # [batch_size, seq_len_x + 1, 2]
            filter_mask, _ = pack((history_mask, future_mask), "bs * m")  # [batch_size, seq_len_h + seq_len_x + 2, 2]
            filter_mask = repeat(
                filter_mask, "b l m -> n b l m", n=1
            )  # [number_of_sampled_sequence, batch_size, seq_len_h + seq_len_x + 2, 2]

            L_sp_m, L_rp_m = self.get_metric_values(input_marks, input_time, input_mask, filter_mask, mean, var)

            all_mask.append(filter_mask.tolist())
            gap.append(L_sp_m - L_rp_m)
            L_sp_model.append(L_sp_m)
            L_rp_model.append(L_rp_m)

        return all_mask, gap, L_sp_model, L_rp_model, list(the_number_of_remained_marks)

    def lsp_and_lrp_trend(self, input_data, opt):
        """
        So this function verifies the assumption 1.
        """
        input_time, input_marks, input_mask, mean, std = self.extract_minibatch(input_data)

        assert (input_mask == 1).all()
        (time_history, time_future), (marks_history, marks_future), (mask_history, mask_future) = (
            self.divide_history_and_future(input_time, input_marks, input_mask)
        )
        # ([batch_size, seq_len_h + 1], [batch_size, seq_len_x + 1]) * 3

        the_number_of_remained_marks = range(1, self.seq_len_h + 1)
        number_of_sampled_sequence = 32
        L_sp_rs = []
        L_rp_rs = []
        for the_number_of_remained_marks_per_seq in the_number_of_remained_marks:
            # baseline 1, random removal.
            rand_mat = torch.rand(number_of_sampled_sequence, self.seq_len_h, device=self.device)
            # [number_of_sampled_sequence, seq_len_h]
            k_th_quant = torch.topk(rand_mat, the_number_of_remained_marks_per_seq - 1, largest=False)[0][:, -1:]
            # [number_of_sampled_sequence, 1]
            if the_number_of_remained_marks_per_seq == 1:
                mask = torch.ones_like(rand_mat, device=self.device).long()  # [number_of_sampled_sequence, seq_len_h]
            else:
                mask = (rand_mat > k_th_quant).long()  # [number_of_sampled_sequence, seq_len_h]
            generated_mask_probability_random = F.one_hot(mask, num_classes=2)
            # [number_of_sampled_sequence, seq_len_h, 2]
            check_tensor(generated_mask_probability_random)

            # Since we don't need gradient during evaluation, we simply use argmax() here to generate history_mask.
            history_mask_random = F.one_hot(torch.argmax(generated_mask_probability_random, dim=-1), num_classes=2)
            # [number_of_sampled_sequence, seq_len_h, 2]
            history_mask_random, _ = pack(
                (torch.ones(number_of_sampled_sequence, 1, 2, device=self.device), history_mask_random), "nss * m"
            )
            # [number_of_sampled_sequence, seq_len_h, 2]
            check_tensor(history_mask_random)

            future_mask_random = torch.ones(number_of_sampled_sequence, self.seq_len_x + 1, 2, device=self.device)
            # [number_of_sampled_sequence, seq_len_x + 1, 2]

            filter_mask_random, _ = pack((history_mask_random, future_mask_random), "nss * m")
            # [number_of_sampled_sequence, seq_len_h + seq_len_x + 2, 2]
            filter_mask_random = repeat(filter_mask_random, "n l m -> n b l m", b=1)
            # [number_of_sampled_sequence, batch_size, seq_len_h + seq_len_x + 2, 2]

            L_sp_r, L_rp_r = self.get_metric_values(input_marks, input_time, input_mask, filter_mask_random, mean, std)
            L_sp_rs.append(L_sp_r)
            L_rp_rs.append(L_rp_r)

            mask = torch.zeros_like(mask_history, device=self.device) * mask_history
            # [batch_size, seq_len_h + 1]

        def get_ratio(input_list):
            tmp = np.array(input_list)
            return (tmp - tmp.min()) / (tmp.max() - tmp.min())

        L_rp_rs_ratio = get_ratio(L_rp_rs)
        L_sp_rs_ratio = get_ratio(L_sp_rs)
        return L_rp_rs_ratio, L_sp_rs_ratio

    """
    All static methods
    """

    def train_step(self, minibatch):
        """
        Epoch operation in training phase.
        The input minibatch comprise time sequences.

        Args:
            minibatch: [batch_size, seq_len]
                       contains [time_seq, mark_seq, score, mask]
        """
        self.train()
        [time_seq, mark_seq, mask], (mean, std) = minibatch

        loss, loss_number_of_useful_events, loss_how_useful_events_are, avg_gap_between_p_1_and_p_0 = self.forward(
            task_name="train", input_time=time_seq, input_marks=mark_seq, input_mask=mask, mean=mean, std=std
        )
        loss.backward()

        loss = loss.item()
        loss_number_of_useful_events = loss_number_of_useful_events.item()
        loss_how_useful_events_are = loss_how_useful_events_are.item()
        avg_gap_between_p_1_and_p_0 = avg_gap_between_p_1_and_p_0.item()

        return loss, loss_number_of_useful_events, loss_how_useful_events_are, avg_gap_between_p_1_and_p_0

    def evaluation_step(self, minibatch):
        """Epoch operation in evaluation phase"""

        self.eval()
        [time_seq, mark_seq, mask], (mean, std) = minibatch

        (
            loss,
            loss_number_of_useful_events,
            loss_how_useful_events_are,
            avg_gap_between_p_1_and_p_0,
            the_number_of_left_events,
        ) = self.forward(
            task_name="evaluate", input_time=time_seq, input_marks=mark_seq, input_mask=mask, mean=mean, std=std
        )

        loss = loss.item()
        loss_number_of_useful_events = loss_number_of_useful_events.item()
        loss_how_useful_events_are = loss_how_useful_events_are.item()
        avg_gap_between_p_1_and_p_0 = avg_gap_between_p_1_and_p_0.item()
        the_number_of_left_events = the_number_of_left_events.item()

        return (
            loss,
            loss_number_of_useful_events,
            loss_how_useful_events_are,
            avg_gap_between_p_1_and_p_0,
            the_number_of_left_events,
        )

    def postprocess(self, input_data, procedure):
        def train_postprocess(input_data):
            """
            Training process
            [absolute loss, relative loss, marks loss]
            """
            return [input_data[0], input_data[1], input_data[2], input_data[3]]

        def test_postprocess(input_data):
            """
            Evaluation process
            [absolute loss, relative loss, marks loss, mae value]
            """
            return [
                input_data[0],
                input_data[1],
                input_data[2],
                input_data[3],
                input_data[4],
            ]

        return train_postprocess(input_data) if procedure == "training" else test_postprocess(input_data)

    def log_print_format(self, input_data, procedure):
        def train_log_print_format(input_data):
            format_dict = {}
            format_dict["Loss"] = pack_one_value_to_dict(input_data[0])
            format_dict["L_c"] = pack_one_value_to_dict(input_data[1])
            format_dict["L_p"] = pack_one_value_to_dict(input_data[2])
            format_dict["L_g"] = pack_one_value_to_dict(input_data[3])
            return format_dict

        def test_log_print_format(input_data):
            format_dict = {}
            format_dict["Loss"] = pack_one_value_to_dict(input_data[0])
            format_dict["L_n"] = pack_one_value_to_dict(input_data[1])
            format_dict["L_e"] = pack_one_value_to_dict(input_data[2])
            format_dict["L_g"] = pack_one_value_to_dict(input_data[3])
            format_dict["the_number_of_left_events"] = pack_one_value_to_dict(input_data[4])
            return format_dict

        return train_log_print_format(input_data) if procedure == "training" else test_log_print_format(input_data)

    format_dict_length = 5

    def choose_metric(self, evaluation_report_format_dict, test_report_format_dict):
        """
        [relative loss on evaluation dataset, relative loss on test dataset, mark loss on test dataset]
        """
        return [evaluation_report_format_dict["Loss"], test_report_format_dict["Loss"]], [
            "evaluation_Loss",
            "test_Loss",
        ]

    metric_number = 2  # metric number is the length of the output of choose_metric
