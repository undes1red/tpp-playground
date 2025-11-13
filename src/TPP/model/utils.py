import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from einops import repeat
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score

from src.toolbox.misc import move_from_tensor_to_ndarray

default_figure_kwargs = {'font.size': 18,
                         'figure.figsize': (8, 4)}


'''
This function returns a list consisting of the step size of each operation.
For example:
(total_rate: 40, step_size: 15) -> [15, 15, 10]
'''
def step_split(total_rate, step_size):
    substep_rate_list = []
    while total_rate > 0:
        substep_rate_list.append(step_size)
        total_rate -= step_size
    substep_rate_list[-1] += total_rate

    return substep_rate_list


'''
Thinning algorithm.
'''
def thinning_sampling(maximum_thinning_loops, max_sample_time_limit, sample_output_shape, device, intensity_func, \
                      find_maximum_intensity_values_in_one_interval, *args, **kwargs):
    sample_rate, batch_size, seq_len = sample_output_shape
    thinning_unit_interval_length = max_sample_time_limit / maximum_thinning_loops

    predicted_time = torch.zeros(sample_rate, batch_size, seq_len, dtype = torch.int32, device = device)
                                                                               # [sample_rate, batch_size, seq_len]
    # The initial mask tensor contains only zero.
    # Zero means we have got a valid time sample.
    # One means we need a resample
    rejected_mask = torch.ones(sample_rate, batch_size, seq_len, dtype = torch.int32, device = device)
                                                                               # [sample_rate, batch_size, seq_len]
    thinning_loops = 0
    while(rejected_mask.sum() > 0):
        thinning_loops += 1
        if thinning_loops > maximum_thinning_loops:
            break

        sampling_interval_left_side = torch.ones_like(rejected_mask) * thinning_unit_interval_length * (thinning_loops - 1)
                                                                               # [sample_rate, batch_size, seq_len]
        sampling_interval_right_side = torch.ones_like(rejected_mask) * thinning_unit_interval_length * thinning_loops
                                                                               # [sample_rate, batch_size, seq_len]
        intensity_values_for_thinning_upper_bound = find_maximum_intensity_values_in_one_interval(sampling_interval_left_side, sampling_interval_right_side, *args, **kwargs) * 1.05
                                                                               # [sample_rate, batch_size, seq_len]
        # Exponential distribution: F(x) = 1 - exp(-\\lambda x) => x = ln(1 - F(x)) / (-\\lambda)
        probability_threshold_for_exp = torch.zeros_like(intensity_values_for_thinning_upper_bound)
                                                                               # [sample_rate, batch_size, seq_len]
        torch.nn.init.uniform_(probability_threshold_for_exp)                  # [sample_rate, batch_size, seq_len]
        probability_threshold_for_thinning = torch.zeros_like(intensity_values_for_thinning_upper_bound)
                                                                               # [sample_rate, batch_size, seq_len]
        torch.nn.init.uniform_(probability_threshold_for_thinning)             # [sample_rate, batch_size, seq_len]
        sampled_time = - torch.log(1 - probability_threshold_for_exp) / (intensity_values_for_thinning_upper_bound + 1e-20)
                                                                               # [sample_rate, batch_size, seq_len]
        # Part 1: exclude time exceeding the limit.
        sampled_time_exceeding_limit = sampled_time > thinning_unit_interval_length
                                                                               # [sample_rate, batch_size, seq_len]
        # Part 2: exclude time rejected by the learned MTPP.
        intensity_values_at_sampled_time = intensity_func(sampled_time, *args, **kwargs)
                                                                               # [sample_rate, batch_size, seq_len]
        sampled_time_rejected = probability_threshold_for_thinning > intensity_values_at_sampled_time / intensity_values_for_thinning_upper_bound
                                                                               # [sample_rate, batch_size, seq_len]
        rejected_in_this_loop = sampled_time_rejected | sampled_time_exceeding_limit
                                                                               # [sample_rate, batch_size, seq_len]
        accept_mask = rejected_mask & (~rejected_in_this_loop)                 # [sample_rate, batch_size, seq_len]
        rejected_mask = rejected_mask & rejected_in_this_loop                  # [sample_rate, batch_size, seq_len]
        predicted_time = predicted_time + accept_mask * sampled_time + rejected_mask * thinning_unit_interval_length
                                                                               # [sample_rate, batch_size, seq_len]
    return predicted_time


'''
Sample a mark from a (unnormalized) probability distribution.
'''
def predict_mark(probability, sample = False):
    # The shape of the input probability is [..., num_marks].
    if sample:
        distribution_of_marks = torch.distributions.categorical.Categorical(probability)
        sampled_marks = distribution_of_marks.sample()                         # [...]
    else:
        sampled_marks = torch.argmax(probability, dim = -1)                    # [...]

    return sampled_marks


'''
resolution_inf and resolution_between_marks.
'''
def decide_resolution_inf_and_resolution_between_marks(time, memory_ceiling, num_marks, mean, std):
    # Suggested batch_size: 1

    max_ = time.mean() + 10 * time.std() if mean == 0 and std == 1 else mean + 10 * std

    if mean == 0:
        resolution_between_marks = max(min(int(time.mean().item() // 0.005), 500), 10)
    else:
        resolution_between_marks = max(min(int(mean // 0.005), 500), 10)

    max_ = min(1e6, max_)
    resolution_inf = max(int(max_ // 0.005), 100)

    batch_size, seq_len = time.shape
    if batch_size * seq_len * resolution_inf * num_marks > memory_ceiling:
        resolution_inf = int(memory_ceiling // (seq_len * num_marks * batch_size))

    if batch_size * seq_len * resolution_between_marks * num_marks * num_marks > memory_ceiling:
        resolution_between_marks = int(memory_ceiling // (seq_len * num_marks * num_marks * batch_size))

    return max_, resolution_inf, resolution_between_marks


'''
custom metrics
'''
def get_f1_and_top_k_acc_in_mae_e(marks_true, p_m, input_mask, num_marks):
    f1 = []
    top_k_acc = []
    for (marks_true_per_seq, probability_integral_per_seq, input_mask_per_seq) in zip(marks_true, p_m, input_mask):
        marks_true_per_seq, probability_integral_per_seq, input_mask_per_seq \
            = move_from_tensor_to_ndarray(marks_true_per_seq, probability_integral_per_seq, input_mask_per_seq)
        y_pred = np.argmax(probability_integral_per_seq, axis = -1)

        selected_marks_true_per_seq = marks_true_per_seq[input_mask_per_seq == 1]
        selected_y_pred = y_pred[input_mask_per_seq == 1]
        selected_probability_integral_per_seq = probability_integral_per_seq[input_mask_per_seq == 1]

        f1.append(f1_score(y_true = selected_marks_true_per_seq, y_pred = selected_y_pred, average = 'macro'))
        top_k_acc_single_mark_seq = []
        if num_marks > 2:
            for k in range(1, num_marks):
                top_k_acc_single_mark_seq.append(
                    top_k_accuracy_score(y_true = selected_marks_true_per_seq,
                                         y_score = selected_probability_integral_per_seq,
                                         k = k,
                                         labels = np.arange(num_marks))
                )
        else:
            top_k_acc_single_mark_seq.append(
                accuracy_score(
                    y_true = selected_marks_true_per_seq, y_pred = selected_y_pred
                )
            )
        top_k_acc.append(top_k_acc_single_mark_seq)

    return f1, top_k_acc


'''
Plotting.
'''
def draw_intensity_integral_and_probability(df, df_mark, annotation, figure_type, color_palette, num_marks, figure_kwargs = {}):
    figure_kwargs = dict(default_figure_kwargs, **figure_kwargs)
    no_ground_truth = len(df.columns) == 2

    df_plot = pd.melt(df, 'Time')
    df_plot.columns = ['Time', ' ', figure_type]

    with mpl.rc_context(figure_kwargs):
        fig, ax = plt.subplots()
        sns.lineplot(x = 'Time', y = figure_type, hue = ' ', data = df_plot, ax = ax)

        handles, labels = ax.get_legend_handles_labels()
        lineplot_legend = ax.legend(handles = handles, labels = labels, loc = 'lower left')
        ax.add_artist(lineplot_legend)

        sns.scatterplot(x = 'Time', y = 'Point', data = df_mark, palette = color_palette, \
                        hue = 'Mark', hue_order = [f'Mark {item}' for item in range(num_marks)], ax = ax)

        handles, labels = ax.get_legend_handles_labels()
        lineplot_legend = ax.legend(handles = handles[1 if no_ground_truth else 2:], labels = labels[1 if no_ground_truth else 2:])
        lineplot_legend.set_title('Mark')
        ax.add_artist(lineplot_legend)

        if annotation is not None:
            props = {'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.5}
            ax.text(0.05, 0.95, annotation, transform = ax.transAxes, fontsize = 14, verticalalignment = 'top', bbox=props)

    return fig


def legend_format(num_marks):
    import math

    format_parameter = {'ncol': 1, 'fontsize': 18}

    if num_marks > 10:
        format_parameter['ncol'] = 2

    num_marks_per_column = math.ceil(num_marks / format_parameter['ncol'])
    format_parameter['fontsize'] = format_parameter['fontsize'] * (-0.1 * max(num_marks_per_column - 5, 0) + 1)

    return format_parameter


def draw_intensity_integral_per_mark(df, df_mark, figure_type, color_palette, num_marks, figure_kwargs = {}):
    figure_kwargs = dict(default_figure_kwargs, **figure_kwargs)

    with mpl.rc_context(figure_kwargs):
        fig, ax = plt.subplots()

        sns.lineplot(x = 'Time', y = figure_type, hue = 'Mark', data = df, palette = color_palette, \
                     hue_order = [f'Mark {item}' for item in range(num_marks)], ax = ax)

        sns.scatterplot(x = 'Time', y = 'Point', data = df_mark, palette = color_palette, \
                        hue = 'Mark', hue_order =  [f'Mark {item}' for item in range(num_marks)], ax = ax)

        handles, labels = ax.get_legend_handles_labels()
        lineplot_legend = ax.legend(handles = [(handles[idx], handles[idx + num_marks]) for idx in range(num_marks)], 
                                    labels = labels[:num_marks], **legend_format(num_marks),
                                    handler_map = {tuple: mpl.legend_handler.HandlerTuple(ndivide = None)})
        lineplot_legend.set_title('Mark')

    return fig


def draw_heatmap(df_matrix, index_name, column_name, value_name, figure_kwargs):
    figure_kwargs = dict(default_figure_kwargs, **figure_kwargs)
    index, column = df_matrix.shape

    # The index and column list
    index_list = [ele for ele in range(index) for _ in range(column)]
    column_list = list(range(column)) * index

    df = pd.DataFrame.from_dict({
        index_name: index_list,
        column_name: column_list,
        value_name: df_matrix.flatten()
    })
    df = df.pivot(index = index_name, columns = column_name, values = value_name)

    with mpl.rc_context(figure_kwargs):
        fig, ax = plt.subplots()
        sns.heatmap(data = df, cmap = "YlGnBu", vmin = 0, vmax = max(1, np.max(df_matrix)), annot = False, ax = ax)

    return fig


def draw_lineplot(*args, figure_kwargs = {}, **kwargs):
    figure_kwargs = dict(default_figure_kwargs, **figure_kwargs)

    with mpl.rc_context(figure_kwargs):
        fig, ax = plt.subplots()
        sns.lineplot(*args, **kwargs, ax = ax)

    return fig


'''
For EHD.
'''
def pick_log_probability(log_probability, last_index, seq_len_x):
    device = last_index.device
    batch_size = last_index.shape[0]

    start_idx = torch.clamp(last_index - 1 - seq_len_x, min = 0)
                                                                           # [batch_size]
    index_indices = torch.arange(seq_len_x, device = device)               # [seq_len_x]
    index_indices = repeat(index_indices, '... -> b ...', b = batch_size) + start_idx.unsqueeze(dim = -1)
                                                                           # [batch_size, seq_len_x]
    log_probability_x = log_probability.gather(-1, index_indices)          # [batch_size, seq_len_x]

    return log_probability_x