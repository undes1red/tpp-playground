import torch
from einops import pack, rearrange, repeat

from src.toolbox.algorithms import bisection
from src.toolbox.misc import check_should_we_stop_sampling, predict_mark
from src.TPP.model.basic_tpp_model import its_lower_bound, its_upper_bound
from src.TPP.model.utils import step_split


def sample_time(self, sampling_approach="its", task="mt", autoregressive=False, *args, **kwargs):
    """
    Sample time from the learned MTPP model using p(t) or p(t|m) using different methods.

    ### Args
        * ```str``` sampling_approach
          Use which method to sample from a distribution.
          its      -> Inverse Transform Sampling.
          thinning -> Thinning algorithm.
        * ```str``` task
          Use which distribution to sample time, p(t) or p(t|m)?
          mt -> p(t|m)
          tm -> p(t)
        * ```bool``` autoregressive
          If true, we autoregressively generate a sequence using the learned MTPP model.
          If false, we sample one next mark given a history sequence.

    ### Args required when sampling from p(t|m) using its.
        * ```torch.tensor``` marks_history
          shape: ```[batch_size, seq_len]```
          Historical mark sequences. Commonly, this sequence is a slice of the original mark sequence from 0 to seq_len - 1(included).
        * ```torch.tensor``` time_history
          shape: ```[batch_size, seq_len]```
          Historical time sequences. Similar to marks_history, we always generate this sequence as a slice of the original time sequence from 0 to seq_len - 1(included).
        * ```torch.tensor``` p_m
          shape: ```[batch_size, seq_len, num_marks]```
          The value of p(m) over the different mark m.
        * ```int``` resolution
          The number of interpolated points in a time interval between two adjoint marks for integration estimation.
          The number of interpolated points counts the start and end point of the interval.
        * ```int``` number_of_total_samples
          This tells how many time samples are generated from the time distribution.
        * ```int``` step
          This parameter controls how many samples are generated in one shot when sampling from p(t|m).
        * ```float``` inf_val
          the upper limit of the bisection method.
        * ```float``` mean
        * ```float``` std
          Used for input time scaling.

    ### Args required when sampling from p(t) using its.
        * ```torch.tensor``` marks_history
          shape: ```[batch_size, seq_len]```
          Historical mark sequences. Commonly, this sequence is a slice of the original mark sequence from 0 to seq_len - 1(included).
        * ```torch.tensor``` time_history
          shape: ```[batch_size, seq_len]```
          Historical time sequences. Similar to marks_history, we always generate this sequence as a slice of the original time sequence from 0 to seq_len - 1(included).
        * ```int``` number_of_total_samples
          This tells how many time samples are generated from the time distribution.
        * ```int``` step
          This parameter controls how many samples are generated in one shot when sampling from p(t|m).
        * ```float``` inf_val
          the upper limit of the bisection method.
        * ```float``` mean
        * ```float``` std
          Used for input time scaling.

    ### Args required when sampling from p(t|m) using thinning.
        Do not exist since it is impossible for now to sample from p(t|m) using thinning.

    ### Args required when sampling from p(t) using thinning.
        * ```torch.tensor``` marks_history
          shape: ```[batch_size, seq_len]```
          Historical mark sequences. Commonly, this sequence is a slice of the original mark sequence from 0 to seq_len - 1(included).
        * ```torch.tensor``` time_history
          shape: ```[batch_size, seq_len]```
          Historical time sequences. Similar to marks_history, we always generate this sequence as a slice of the original time sequence from 0 to seq_len - 1(included).
        * ```int``` number_of_total_samples
          This tells how many time samples are generated from the time distribution.
        * ```int``` step
          This parameter controls how many samples are generated in one shot when sampling from p(t|m).
        * ```float``` inf_val
          the upper limit of the bisection method.
        * ```float``` mean
        * ```float``` std
          Used for input time scaling.
    """
    if autoregressive:
        dict_sampling_apparoch = {
            "its": autoregressive_sampling_by_its,
            "thinning": autoregressive_sampling_by_thinning,
        }
    else:
        dict_sampling_apparoch = {"its": sampling_by_its, "thinning": sampling_by_thinning}

    return dict_sampling_apparoch[sampling_approach](self, task, *args, **kwargs)


# Sample marks from p^*(m, t) using inversed transform sampling in a autoregressive manner.
def autoregressive_sampling_by_its(self, task, *args, **kwargs):
    dict_apparoch_for_tasks = {"mt": autoregressive_sampling_by_its_for_mt, "tm": autoregressive_sampling_by_its_for_tm}

    return dict_apparoch_for_tasks[task](self, *args, **kwargs)


# Sample marks from p^*(m, t) using thinning algorithm in a autoregressive manner.
def autoregressive_sampling_by_thinning(self, task, *args, **kwargs):
    dict_apparoch_for_tasks = {
        "mt": autoregressive_sampling_by_thinning_for_mt,
        "tm": autoregressive_sampling_by_thinning_for_tm,
    }

    return dict_apparoch_for_tasks[task](self, *args, **kwargs)


# Sample marks from p^*(m, t) using inversed transform sampling.
def sampling_by_its(self, task, *args, **kwargs):
    dict_apparoch_for_tasks = {"mt": sampling_by_its_for_mt, "tm": sampling_by_its_for_tm}

    return dict_apparoch_for_tasks[task](self, *args, **kwargs)


# Sample marks from p^*(m, t) using thinning algorithm.
def sampling_by_thinning(self, task, *args, **kwargs):
    dict_apparoch_for_tasks = {"mt": sampling_by_thinning_for_mt, "tm": sampling_by_thinning_for_tm}

    return dict_apparoch_for_tasks[task](self, *args, **kwargs)


# For autoregressive_sampling_by_its.
def autoregressive_sampling_by_its_for_mt(
    self, time_history, marks_history, p_m, number_of_total_samples, step, inf_val, mean, std
):
    # Preprocess
    sample_rate_list = step_split(number_of_total_samples, step)

    def bisect_target(taus, probability_threshold):
        # \\int_{tau}^{+\\inf}{p(m, \\tau|\\mathcal{H})d\\tau}
        probability_integral_from_t_to_infinite = self.model.gamma_at_t_autoregressive(
            time_history, taus, marks_history, mean=mean, std=std, extend_input_time=False
        )
        # [sample_rate, num_marks]
        # \\int_{0}^{tau}{p(m, \\tau|\\mathcal{H})d\\tau}
        p_mt = p_m - probability_integral_from_t_to_infinite  # [sample_rate, num_marks]
        p_t_m = p_mt / p_m  # [sample_rate, num_marks]
        return p_t_m - probability_threshold  # [sample_rate, num_marks]

    # Preprocess
    tau_pred = []
    for sub_sample_rate in sample_rate_list:
        probability_threshold = torch.zeros((sub_sample_rate, self.num_marks), device=self.device)
        # [sub_sample_rate, num_marks]
        torch.nn.init.uniform_(probability_threshold, a=its_lower_bound, b=its_upper_bound)
        # [sub_sample_rate, num_marks]
        tau_pred.append(
            bisection(
                self.max_step, self.bisect_early_stop_threshold, bisect_target, probability_threshold, r_val=inf_val
            )
        )
        # [sub_sample_rate, num_marks]

    return torch.cat(tau_pred, dim=0)  # [sample_rate, num_marks]


def autoregressive_sampling_by_its_for_tm(self, time_history, marks_history, number_of_total_samples, step, mean, std):
    # Preprocess
    sample_rate_list = step_split(number_of_total_samples, step)

    def evaluate(taus, probability_threshold, integral_from_zero_to_inf):
        probability_integral_from_t_to_inf = self.model.gamma_at_t_autoregressive(
            time_history, taus, marks_history, mean, std
        )
        # [sample_rate, num_marks]
        # P_m(t) = \\int_{0}^{t}{p(t|m, \\mathcal{H})}
        probability_integral = integral_from_zero_to_inf - probability_integral_from_t_to_inf
        # [sample_rate, num_marks]
        probability_integral = torch.sum(probability_integral, dim=-1)  # [sample_rate]

        return probability_integral - probability_threshold

    tau_pred = []
    for sub_sample_rate in sample_rate_list:
        probability_threshold = torch.zeros(sub_sample_rate, device=self.device)
        # [sub_sample_rate]
        torch.nn.init.uniform_(probability_threshold, a=its_lower_bound, b=its_upper_bound)
        # [sub_sample_rate]

        time_next_zero = torch.zeros_like(probability_threshold)  # [sub_sample_rate]
        # [sub_sample_rate, num_marks]
        integral_from_zero_to_inf = self.model.gamma_at_t_autoregressive(
            time_history, time_next_zero, marks_history, mean=mean, std=std
        )
        # [sub_sample_rate, num_marks]

        tau_pred.append(
            bisection(
                self.max_step,
                self.bisect_early_stop_threshold,
                evaluate,
                probability_threshold,
                integral_from_zero_to_inf,
            )
        )
        # [sub_sample_rate]
    return torch.cat(tau_pred, dim=0)  # [sample_rate]


# For autoregressive_sampling_by_thinning.
def autoregressive_sampling_by_thinning_for_mt(self, *args, **kwargs):
    raise Exception("Thinning algorithm can not solve task MT. Please use ITS by setting sampling_approach = its.")


def autoregressive_sampling_by_thinning_for_tm(
    self, marks_history, time_history, mask_history, number_of_total_samples, step, mean, std
):
    raise Exception(
        "IFIB does not know intensity functions, which thinning algorithm requires. Please use ITS by setting sampling_approach = its."
    )


# For sampling_by_its.
def sampling_by_its_for_mt(
    self, time_history, marks_history, mask_history, p_m, number_of_total_samples, step, inf_val, mean, std
):
    # Preprocess
    sample_rate_list = step_split(number_of_total_samples, step)

    def bisect_target(taus, probability_threshold):
        # \\int_{tau}^{+\\inf}{p(m, \\tau|\\mathcal{H})d\\tau}
        probability_integral_from_t_to_infinite, _ = self.model(
            time_history, taus, marks_history, mask_history, mean, std, extend_input_time=False
        )
        # [sample_rate, batch_size, seq_len, num_marks]
        # \\int_{0}^{tau}{p(m, \\tau|\\mathcal{H})d\\tau}
        p_mt = p_m - probability_integral_from_t_to_infinite  # [sample_rate, batch_size, seq_len, num_marks]
        p_t_m = p_mt / p_m  # [sample_rate, batch_size, seq_len, num_marks]
        return p_t_m - probability_threshold  # [sample_rate, batch_size, seq_len, num_marks]

    # Preprocess
    tau_pred = []
    batch_size, seq_len = time_history.shape
    for sub_sample_rate in sample_rate_list:
        probability_threshold = torch.zeros((batch_size, seq_len, self.num_marks, sub_sample_rate), device=self.device)
        # [batch_size, seq_len, num_marks, sub_sample_rate]
        torch.nn.init.uniform_(probability_threshold, a=its_lower_bound, b=its_upper_bound)
        # [batch_size, seq_len, num_marks, sub_sample_rate]
        probability_threshold = rearrange(probability_threshold, "bs sl nm ssr -> ssr bs sl nm")
        # [sub_sample_rate, batch_size, seq_len, num_marks]
        tau_pred.append(
            bisection(
                self.max_step, self.bisect_early_stop_threshold, bisect_target, probability_threshold, r_val=inf_val
            )
        )
        # [sub_sample_rate, batch_size, seq_len, num_marks]

    return torch.cat(tau_pred, dim=0)  # [sample_rate, batch_size, seq_len, num_marks]


def sampling_by_its_for_tm(self, time_history, marks_history, mask_history, number_of_total_samples, step, mean, std):
    # Preprocess
    sample_rate_list = step_split(number_of_total_samples, step)

    def evaluate(taus, probability_threshold, integral_from_zero_to_inf):
        probability_integral_from_t_to_inf, _ = self.model(time_history, taus, marks_history, mask_history, mean, std)
        # [sample_rate, batch_size, seq_len, num_marks]
        # P_m(t) = \\int_{0}^{t}{p(t|m, \\mathcal{H})}
        probability_integral = integral_from_zero_to_inf - probability_integral_from_t_to_inf
        # [sample_rate, batch_size, seq_len, num_marks]
        probability_integral = torch.sum(probability_integral, dim=-1)  # [sample_rate, batch_size, seq_len]

        return probability_integral - probability_threshold

    tau_pred = []
    batch_size, seq_len = time_history.shape
    for sub_sample_rate in sample_rate_list:
        probability_threshold = torch.zeros((batch_size, seq_len, sub_sample_rate), device=self.device)
        # [batch_size, seq_len, sub_sample_rate]
        torch.nn.init.uniform_(probability_threshold, a=its_lower_bound, b=its_upper_bound)
        # [batch_size, seq_len, sub_sample_rate]
        probability_threshold = rearrange(probability_threshold, "bs sl ssr -> ssr bs sl")
        # [sub_sample_rate, batch_size, seq_len]

        time_next_zero = torch.zeros_like(probability_threshold)
        # [sub_sample_rate, batch_size, seq_len]
        integral_from_zero_to_inf, _ = self.model(
            time_history, time_next_zero, marks_history, mask_history, mean=mean, std=std
        )
        # [sub_sample_rate, batch_size, seq_len, num_marks]

        tau_pred.append(
            bisection(
                self.max_step,
                self.bisect_early_stop_threshold,
                evaluate,
                probability_threshold,
                integral_from_zero_to_inf,
            )
        )
        # [sub_sample_rate, batch_size, seq_len]
    return torch.cat(tau_pred, dim=0)  # [sample_rate, batch_size, seq_len]


# For sampling_by_thinning.
def sampling_by_thinning_for_mt(self, *args, **kwargs):
    raise Exception("Thinning algorithm can not solve task MT. Please use ITS by setting sampling_approach = its.")


def sampling_by_thinning_for_tm(
    self, marks_history, time_history, mask_history, number_of_total_samples, step, mean, std
):
    raise Exception(
        "IFIB does not know intensity functions, which thinning algorithm requires. Please use ITS by setting sampling_approach = its."
    )


# For autoregressive sampling.
def sample_time_mark(
    self, time_history_for_sampling, marks_history_for_sampling, mean, std, end_sampling_requirement="time", **kwargs
):
    """
    This function will sample x sequences by the learned probability distribution following the time-mark prediction procedure.
    Steps:
    1. Sample a time \\(t_s\\) from p^*(t) = \\sum{n \\in M}{p^*(m, t)} referring to existing history
    2. Judge the mark of this mark by comparing \\(\\lambda^*(m, t_s)\\).
    """
    if time_history_for_sampling is None and marks_history_for_sampling is None:
        number_of_sampled_sequences = kwargs["number_of_sampled_sequences"]
        time_history_for_sampling = torch.zeros((number_of_sampled_sequences, 1), device=self.device)
        # [number_of_sampled_sequences, 1]
        marks_history_for_sampling = (
            torch.ones((number_of_sampled_sequences, 1), device=self.device, dtype=torch.int32) * self.num_marks
        )
    # [number_of_sampled_sequences, 1]
    else:
        assert time_history_for_sampling is not None and marks_history_for_sampling is not None, (
            "How is it possible that one input history is not None while another one is?"
        )
        assert marks_history_for_sampling.shape[0] == time_history_for_sampling.shape[0], (
            f"time_history_for_sampling says we will sample {time_history_for_sampling.shape[0]} sequences, while marks_history_for_sampling suggests {marks_history_for_sampling.shape[0]}. So, how many sequences should we sample?"
        )
        number_of_sampled_sequences = marks_history_for_sampling.shape[0]

    sampled_mask = None

    while True:
        sampled_time = self.sample_time(
            "its",
            "tm",
            True,
            time_history_for_sampling,
            marks_history_for_sampling,
            number_of_sampled_sequences,
            number_of_sampled_sequences,
            mean,
            std,
        )
        # [number_of_sampled_sequences]
        repeated_sampled_time = repeat(
            sampled_time, "... -> ... ne", ne=self.num_marks
        )  # [number_of_sampled_sequences, num_marks]
        repeated_sampled_time.requires_grad = True
        integral_from_sampled_time_to_inf = self.model.gamma_at_t_autoregressive(
            time_history_for_sampling,
            repeated_sampled_time,
            marks_history_for_sampling,
            mean=mean,
            std=std,
            extend_input_time=False,
        )
        # [number_of_sampled_sequences, num_marks]
        probability_for_each_mark_at_pred_time = -torch.autograd.grad(
            outputs=integral_from_sampled_time_to_inf,
            inputs=repeated_sampled_time,
            grad_outputs=torch.ones_like(integral_from_sampled_time_to_inf),
        )[0]  # [number_of_sampled_sequences, num_marks]
        repeated_sampled_time.requires_grad = False

        sampled_marks = predict_mark(probability_for_each_mark_at_pred_time.log(), sample=True, logits=True)
        # [number_of_sampled_sequences]

        tmp_marks_history_for_sampling, _ = pack([marks_history_for_sampling, sampled_marks], "nss *")
        # [number_of_sampled_sequences, history_length + 1]
        tmp_time_history_for_sampling, _ = pack([time_history_for_sampling, sampled_time], "nss *")
        # [number_of_sampled_sequences, history_length + 1]

        should_we_stop, sampled_mask = check_should_we_stop_sampling(
            tmp_time_history_for_sampling, end_sampling_requirement, **kwargs
        )

        if should_we_stop:
            # Remove the mask of the temporarily added mark.
            sampled_mask = sampled_mask[..., :-1]
            break

        marks_history_for_sampling = tmp_marks_history_for_sampling  # [number_of_sampled_sequences, history_length + 1]
        time_history_for_sampling = tmp_time_history_for_sampling  # [number_of_sampled_sequences, history_length + 1]

    return time_history_for_sampling, marks_history_for_sampling, sampled_mask


def sample_mark_time(
    self, time_history_for_sampling, marks_history_for_sampling, mean, std, end_sampling_requirement="time", **kwargs
):
    """
    These two functions will sample a mark sequence from the learned p^*(m, t) following the mark-time prediction procedure.
    Steps:
    1. Sample the mark \\(m_p\\) from p^*(m) = \\int_{t_l}^{+\\infty}{p^*(m, \\tau)d\\tau}.
    2. Sample when a new \\(m_p\\) mark would happen in the future time by \\(p^*(t|m_p)\\).
    """
    if time_history_for_sampling is None and marks_history_for_sampling is None:
        number_of_sampled_sequences = kwargs["number_of_sampled_sequences"]
        time_history_for_sampling = torch.zeros((number_of_sampled_sequences, 1), device=self.device)
        # [number_of_sampled_sequences, 1]
        marks_history_for_sampling = (
            torch.ones((number_of_sampled_sequences, 1), device=self.device, dtype=torch.int32) * self.num_marks
        )
    # [number_of_sampled_sequences, 1]
    else:
        assert time_history_for_sampling is not None and marks_history_for_sampling is not None, (
            "How is it possible that one history is not None while another one is?"
        )
        assert marks_history_for_sampling.shape[0] == time_history_for_sampling.shape[0], (
            f"time_history_for_sampling says we will sample {time_history_for_sampling.shape[0]} sequences, while marks_history_for_sampling suggests {marks_history_for_sampling.shape[0]}. So, how many sequences should we sample?"
        )
        number_of_sampled_sequences = marks_history_for_sampling.shape[0]

    sampled_mask = None

    while True:
        time_next_zero = torch.zeros(number_of_sampled_sequences, device=self.device)
        # [number_of_sampled_sequences]
        integral_from_zero_to_inf = self.model.gamma_at_t_autoregressive(
            time_history_for_sampling, time_next_zero, marks_history_for_sampling, mean=mean, std=std
        )
        # [number_of_sampled_sequences]
        sampled_marks = predict_mark(integral_from_zero_to_inf.log(), sample=True, logits=True)
        # [number_of_sampled_sequences]
        all_sampled_time = self.sample_time(
            "its",
            "mt",
            True,
            time_history_for_sampling,
            marks_history_for_sampling,
            integral_from_zero_to_inf,
            number_of_sampled_sequences,
            number_of_sampled_sequences,
            1e6,
            mean,
            std,
        )
        # [number_of_sampled_sequences, num_marks]
        one_hot_mask_of_sampled_marks = torch.nn.functional.one_hot(sampled_marks, num_classes=self.num_marks)
        # [number_of_sampled_sequences, num_marks]
        sampled_time = torch.sum(all_sampled_time * one_hot_mask_of_sampled_marks, dim=-1)
        # [number_of_sampled_sequences]

        tmp_marks_history_for_sampling, _ = pack([marks_history_for_sampling, sampled_marks], "nss *")
        # [number_of_sampled_sequences, history_length + 1]
        tmp_time_history_for_sampling, _ = pack([time_history_for_sampling, sampled_time], "nss *")
        # [number_of_sampled_sequences, history_length + 1]

        should_we_stop, sampled_mask = check_should_we_stop_sampling(
            tmp_time_history_for_sampling, end_sampling_requirement, **kwargs
        )

        if should_we_stop:
            # Remove the mask of the temporarily added mark.
            sampled_mask = sampled_mask[..., :-1]
            break

        marks_history_for_sampling = tmp_marks_history_for_sampling  # [number_of_sampled_sequences, history_length + 1]
        time_history_for_sampling = tmp_time_history_for_sampling  # [number_of_sampled_sequences, history_length + 1]

    return time_history_for_sampling, marks_history_for_sampling, sampled_mask
