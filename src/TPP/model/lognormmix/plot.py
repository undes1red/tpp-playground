import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.TPP.model.utils import move_from_tensor_to_ndarray, L1_distance_between_two_funcs, stable_palette, figure_instruction_generator
from src.TPP.resources.syn_tpp_utils import expand_true_probability

large_graph_length = 18
large_graph_height = 10


def plot_probability(data, timestamp, opt):
    '''
    '''

    num_events = opt.info_dict['num_events']
    color_palette = stable_palette([f'Mark {i}' for i in range(num_events)])

    plot_instruction = {}
    '''
    Part 1: the sum of probability distributions over all markers.
    '''
    expand_probability = data['expand_probability']                            # [batch_size, seq_len, resolution, num_events]
    mask_next = data['mask_next']                                              # [batch_size, seq_len]
    events_next = data['events_next']                                          # [batch_size, seq_len]
    time_next = data['time_next']                                              # [batch_size, seq_len]
    input_intensity = data['input_intensity']                                  # [batch_size, seq_len + 1]

    true_probability = expand_true_probability(time_next, input_intensity, opt)# [batch_size, seq_len, resolution] or batch_size * None

    packed_data = zip(*move_from_tensor_to_ndarray(expand_probability, events_next, time_next, mask_next, timestamp, true_probability))
    for idx, (expand_probability_per_seq, events_next_per_seq, time_next_per_seq, mask_next_per_seq, timestamp_per_seq, true_probability_per_seq) \
        in enumerate(packed_data):
        seq_len = mask_next_per_seq.sum()

        df_event = pd.DataFrame.from_dict(
                {'Time': time_next_per_seq.cumsum(axis = -1), 'Point': np.zeros_like(events_next_per_seq), \
                 'Mark': [f'Mark {item}' for item in events_next_per_seq]}
        )

        if true_probability_per_seq is not None:
            df = pd.DataFrame.from_dict(
                {'Time': timestamp_per_seq.flatten().cumsum(axis = -1),
                 'Predicted Probability': expand_probability_per_seq[:seq_len, :].flatten(),
                 'Truth': true_probability_per_seq[:seq_len, :].flatten()}
            )

            # Spearman correlation
            rho = spearmanr(a = true_probability_per_seq[:seq_len, :].flatten(), b = expand_probability_per_seq[:seq_len, :].flatten())[0]
            # Pearson correlation
            r = np.corrcoef(x = true_probability_per_seq[:seq_len, :].flatten(), y = expand_probability_per_seq[:seq_len, :].flatten())[0, 1]
            # L1 distance
            L1 = L1_distance_between_two_funcs(x = true_probability_per_seq[:seq_len, :], y = expand_probability_per_seq[:seq_len, :], \
                                               timestamp = timestamp_per_seq, resolution = opt.resolution)

            annotation = fr'r = {r}, \(\rho\) = {rho}, \(L^1\) = {L1}'
        else:
            df = pd.DataFrame.from_dict(
                {'Time': timestamp_per_seq.flatten().cumsum(axis = -1),
                 'Predicted Probability': expand_probability_per_seq[:seq_len, :].flatten()}
            )
            annotation = ''

        df_probability_plot = pd.melt(df, 'Time')
        df_probability_plot.columns = ['Time', ' ', 'Probability']

        subplot_instruction = \
        [
            {
                'plot_type': 'lineplot',
                'kwargs':
                {
                    'x':'Time',
                    'y': 'Probability',
                    'hue': ' ',
                    'data': df_probability_plot
                }
            },
            {
                'plot_type': 'scatterplot',
                'kwargs':
                {
                    'x': 'Time',
                    'y': 'Point',
                    'data': df_event,
                    'palette': color_palette,
                    'hue': 'Mark',
                    'hue_order': [f'Mark {item}' for item in range(num_events)]
                }
            },
            {
                'plot_type': 'text',
                'kwargs':
                {
                    'x': -1, 
                    'y': -0.75,
                    'verticalalignment': 'top',
                    'horizontalalignment': 'left',
                    's': annotation,
                    'fontsize': 12,
                }
            }
        ]

        plot_instruction[f'probability_{idx}'] \
         = figure_instruction_generator(subplot_instruction,
                                        figure_kwargs = {
                                            'figsize': (large_graph_length, large_graph_height),
                                        })

    return plot_instruction


def plot_debug(data, timestamp, opt):
    '''
    What is inside dict data?
    1. expand_intensity_for_each_event  shape: [batch_size, seq_len, resolution, num_events]
    2. expand_integral_for_each_event   shape: [batch_size, seq_len, resolution, num_events]
    3. spearman, pearson, and L1 distance matrix if self.event_toggle = True
    4. macro-f1: measure the event prediction performance without time prediction.
    5. top_k: measure the event prediction performance without time prediction.
    6. probability_sum: the value of \\int_{t_l}^{+infty}{p(m, \\tau)d\\tau}
    7. tau_pred_all_event: The time prediction of all events, with p(m) known.
    8. mae_before_event: as known as MAE.
    9. maes_after_event_avg: contains mae_per_event_with_predict_index_avg and mae_per_event_with_event_next_avg
    10. maes_after_event: contains mae_per_event_with_predict_index and mae_per_event_with_event_next
    11. event_next: 
    12. time_next:
    '''

    plot_instruction = {}

    '''
    Only MAE is available.
    '''
    mae = data['mae_before_event']                                     # [batch_size, seq_len]
    mask_next = data['mask_next']                                      # [batch_size, seq_len]

    packed_data = zip(*move_from_tensor_to_ndarray(mae, mask_next))

    for idx, (mae_per_seq, mask_next_per_seq) in enumerate(packed_data):
        seq_len = mask_next_per_seq.sum()

        data_maes_per_seq = {
            'x': list(range(seq_len)),
            'y': np.log(1 + mae_per_seq[:seq_len]),
            'marks': ['MAE'] * seq_len
        }
        df_data_maes_per_seq = pd.DataFrame.from_dict(data_maes_per_seq)

        subplot_instruction = [
            {
                'plot_type': 'lineplot',
                'kwargs':
                {
                    'x': 'x',
                    'y': 'y',
                    'hue': 'marks',
                    'data': df_data_maes_per_seq,
                    'markers': True
                }
            }
        ]
        plot_instruction[f'log_mae_k_{idx}'] = figure_instruction_generator(subplot_instruction)
    
    return plot_instruction
