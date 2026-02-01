import torch
from einops import rearrange

from src.toolbox.algorithms import approximate_integration, bisection
from src.tpp.tpp_models.model.basic_tpp_model import its_lower_bound, its_upper_bound
from src.tpp.tpp_models.model.utils import step_split, thinning_sampling


@torch.no_grad
def sample_time(self, sampling_approach="its", task="mt", *args, **kwargs):
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
          The number of interpolated points in a time interval between two adjoint mark for integration estimation.
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
    dict_sampling_apparoch = {"its": sampling_by_its, "thinning": sampling_by_thinning}

    return dict_sampling_apparoch[sampling_approach](self, task=task, *args, **kwargs)


def sampling_by_its(self, task, *args, **kwargs):
    dict_apparoch_for_tasks = {"mt": sampling_by_its_for_mt, "tm": sampling_by_its_for_tm}

    return dict_apparoch_for_tasks[task](self, *args, **kwargs)


def sampling_by_its_for_mt(
    self,
    time_history,
    marks_history,
    mask_history,
    p_m,
    resolution,
    number_of_total_samples,
    step,
    inf_val,
    mean,
    std,
):
    sample_rate_list = step_split(number_of_total_samples, step)

    def evaluate_all_mark(taus):
        expanded_integral_across_mark, expanded_intensity_across_mark, timestamp = (
            self.model.integral_intensity_time_next_3d(
                time_history, taus, marks_history, mask_history, resolution, time_next_with_resolution_dim=True
            )
        )
        # 2 * [sample_rate, batch_size, seq_len, num_marks, resolution, num_marks] + [sample_rate, batch_size, seq_len, num_marks, resolution]
        expanded_integral_sum_across_mark = expanded_integral_across_mark.sum(dim=-1)
        # [sample_rate, batch_size, seq_len, num_marks, resolution]
        intensity_mark_mask = torch.diag(torch.ones(self.num_marks, device=self.device))
        # [num_marks, num_marks]
        intensity_mark_mask = rearrange(
            intensity_mark_mask, f"ne ne1 -> {'() ' * (len(expanded_intensity_across_mark.shape) - 3)}ne () ne1"
        )
        # [sample_rate, batch_size, seq_len, num_marks, resolution, num_marks]
        expanded_intensity_per_mark = (expanded_intensity_across_mark * intensity_mark_mask).sum(dim=-1)
        # [sample_rate, batch_size, seq_len, num_marks, resolution]
        expanded_probability_per_mark = expanded_intensity_per_mark * torch.exp(-expanded_integral_sum_across_mark)
        # [sample_rate, batch_size, seq_len, num_marks, resolution]
        integral_in_picked_interval = approximate_integration(expanded_probability_per_mark, timestamp, dim=-1)
        # [sample_rate, batch_size, seq_len, num_marks, resolution]

        # probability distribution integration offset.
        expanded_integral_across_marks, expanded_intensity_across_marks, timestamp = (
            self.model.integral_intensity_time_next_3d(
                time_history,
                taus[..., 0],
                marks_history,
                mask_history,
                resolution,
            )
        )
        expanded_integral_sum_across_marks = expanded_integral_across_marks.sum(dim=-1)
        # [sample_rate, batch_size, seq_len, num_marks, resolution]
        expanded_intensity_per_mark = (expanded_intensity_across_marks * intensity_mark_mask).sum(dim=-1)
        # [sample_rate, batch_size, seq_len, num_marks, resolution]
        expanded_probability_per_mark = expanded_intensity_per_mark * torch.exp(-expanded_integral_sum_across_marks)
        # [sample_rate, batch_size, seq_len, num_marks, resolution]
        integral_offset = approximate_integration(expanded_probability_per_mark, timestamp, dim=-1, only_integral=True)
        # [sample_rate, batch_size, seq_len, num_marks]

        return integral_in_picked_interval + integral_offset.unsqueeze(dim=-1)
        # [sample_rate, batch_size, seq_len, num_marks]

    def bisect_target(taus, probability_threshold):
        p_mt = evaluate_all_mark(taus)  # [sample_rate, batch_size, seq_len, num_marks, resolution]
        p_t_m = p_mt / p_m.unsqueeze(dim=-1)  # [sample_rate, batch_size, seq_len, num_marks, resolution]
        return p_t_m - probability_threshold  # [sample_rate, batch_size, seq_len, num_marks, resolution]

    tau_pred = []
    batch_size, seq_len = time_history.shape

    for sub_sample_rate in sample_rate_list:
        probability_threshold = torch.zeros((batch_size, seq_len, self.num_marks, sub_sample_rate), device=self.device)
        # [batch_size, seq_len, num_marks, sample_rate]
        torch.nn.init.uniform_(probability_threshold, a=its_lower_bound, b=its_upper_bound)
        # [batch_size, seq_len, num_marks, sample_rate]
        probability_threshold = rearrange(probability_threshold, "b sl ne sr -> sr b sl ne")
        # [sample_rate, batch_size, seq_len, num_marks]
        tau_pred.append(
            bisection(
                self.max_step,
                self.bisect_early_stop_threshold,
                bisect_target,
                probability_threshold,
                resolution=100,
                r_val=inf_val,
            )
        )
        # [sample_rate, batch_size, seq_len, num_marks]
    return torch.cat(tau_pred, dim=0)  # [sample_rate, batch_size, seq_len, num_marks]


def sampling_by_its_for_tm(
    self, time_history, marks_history, mask_history, resolution, number_of_total_samples, step, inf_val, mean, std
):
    sample_rate_list = step_split(number_of_total_samples, step)

    def bisect_target(taus, probability_threshold):
        """
        Args:
        1. taus: the probed timestamps. shape: [sample_rate, batch_size, seq_len, resolution]
        2. probability_threshold: the threshold. shape: [sample_rate, batch_size, seq_len, 1]
        """
        (
            expanded_integral_all_marks,
            _,
            _,
        ) = self.model.integral_intensity_time_next_2d(
            time_history,
            taus,
            marks_history,
            mask_history,
            integration_sample_rate=resolution,
            time_next_with_resolution_dim=True,
        )
        # [sample_rate, batch_size, seq_len, integration_sample_rate, num_marks]
        expanded_integral = expanded_integral_all_marks.sum(
            dim=-1
        )  # [sample_rate, batch_size, seq_len, integration_sample_rate]

        return expanded_integral + torch.log(1 - probability_threshold)

    tau_pred = []
    for sub_sample_rate in sample_rate_list:
        probability_threshold = torch.zeros((*time_history.shape, sub_sample_rate), device=self.device)
        # [batch_size, seq_len, sample_rate]
        torch.nn.init.uniform_(probability_threshold, a=its_lower_bound, b=its_upper_bound)
        # [batch_size, seq_len, sample_rate]
        probability_threshold = rearrange(probability_threshold, "b sl sr -> sr b sl")
        # [sample_rate, batch_size, seq_len]
        tau_pred.append(
            bisection(
                self.max_step,
                self.bisect_early_stop_threshold,
                bisect_target,
                probability_threshold,
                resolution=resolution,
                r_val=inf_val
            )
        )
        # [sample_rate, batch_size, seq_len]

    return torch.cat(tau_pred, dim=0)  # [sample_rate, batch_size, seq_len]


def sampling_by_thinning(self, task, *args, **kwargs):
    dict_apparoch_for_tasks = {"mt": sampling_by_thinning_for_mt, "tm": sampling_by_thinning_for_tm}

    return dict_apparoch_for_tasks[task](self, *args, **kwargs)


def sampling_by_thinning_for_mt(self, *args, **kwargs):
    raise Exception("Thinning algorithm can not solve task MT. Please use ITS by setting sampling_approach = its.")


def sampling_by_thinning_for_tm(
    self, marks_history, time_history, mask_history, number_of_total_samples, step, mean, std, autoregressive=False
):
    sample_rate_list = step_split(number_of_total_samples, step)
    batch_size, seq_len = time_history.shape
    maximum_thinning_loops = 50
    max_sample_time_limit = mean + 10 * std

    def get_intensity(tau, time_history, marks_history, mask_history):
        return self.model(time_history, tau, marks_history, mask_history)[-1].sum(dim=-1)

    def find_maximum_intensity_values_in_one_interval(
        interval_left, interval_right, time_history, marks_history, mask_history
    ):
        _, intensity_between_interval_left_and_right, _ = self.model.integral_intensity_time_next_2d(
            marks_history,
            time_history,
            interval_right,
            mask_history,
            self.integration_sample_rate,
            time_next_start=interval_left,
        )
        # [sample_rate, batch_size, seq_len, integration_sample_rate, num_marks]
        intensity_between_interval_left_and_right = intensity_between_interval_left_and_right.sum(dim=-1)
        # [sample_rate, batch_size, seq_len, integration_sample_rate]

        return intensity_between_interval_left_and_right.max(dim=-1)[0]

    sampled_time = []
    for each_step in sample_rate_list:
        sampled_time.append(
            thinning_sampling(
                maximum_thinning_loops,
                max_sample_time_limit,
                (each_step, batch_size, seq_len),
                self.device,
                get_intensity,
                find_maximum_intensity_values_in_one_interval,
                time_history,
                marks_history,
                mask_history,
            )
        )
        # [sample_rate, batch_size, seq_len]

    return torch.cat(sampled_time, dim=0)
