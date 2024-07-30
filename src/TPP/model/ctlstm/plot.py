import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.toolbox.misc import move_from_tensor_to_ndarray, stable_palette, figure_instruction_generator
from src.toolbox.metrics import L1_distance_between_two_funcs

from src.TPP.resources.syn_tpp_utils import expand_true_intensity, expand_true_probability

large_graph_length = 18
large_graph_height = 10


def plot_intensity(data, timestamp, opt):
    '''
    '''

    num_events = opt.info_dict['num_events']
    color_palette = stable_palette([f'Mark {i}' for i in range(num_events)])

    plot_instruction = {}
    '''
    Part 1: the sum of intensity functions over all markers.
    '''
    expand_intensity = data['expand_intensity']                                # [batch_size, seq_len, resolution, num_events]
    mask_next = data['mask_next']                                              # [batch_size, seq_len]
    events_next = data['events_next']                                          # [batch_size, seq_len]
    time_next = data['time_next']                                              # [batch_size, seq_len]
    input_intensity = data['input_intensity']                                  # [batch_size, seq_len + 1]

    expand_intensity = expand_intensity.sum(dim = -1)                          # [batch_size, seq_len, resolution]
    true_intensity = expand_true_intensity(time_next, input_intensity, opt)    # [batch_size, seq_len, resolution]

    packed_data = zip(*move_from_tensor_to_ndarray(expand_intensity, events_next, time_next, mask_next, timestamp, true_intensity))
    for idx, (expand_intensity_per_seq, events_next_per_seq, time_next_per_seq, mask_next_per_seq, timestamp_per_seq, true_intensity_per_seq) \
        in enumerate(packed_data):
        seq_len = mask_next_per_seq.sum()

        df_event = pd.DataFrame.from_dict(
                {'Time': time_next_per_seq.cumsum(axis = -1), 'Point': np.zeros_like(events_next_per_seq), \
                 'Mark': [f'Mark {item}' for item in events_next_per_seq]}
        )

        if true_intensity_per_seq is not None:
            df_intensity = pd.DataFrame.from_dict(
                    {'Time': timestamp_per_seq.flatten().cumsum(axis = -1),
                     'Intensity': expand_intensity_per_seq[:seq_len, :].flatten(),
                     'Truth': true_intensity_per_seq[:seq_len, :].flatten()}
            )

            # Spearman correlation
            rho = spearmanr(a = true_intensity_per_seq[:seq_len, :].flatten(), b = expand_intensity_per_seq[:seq_len, :].flatten())[0]
            # Pearson correlation
            r = np.corrcoef(x = true_intensity_per_seq[:seq_len, :].flatten(), y = expand_intensity_per_seq[:seq_len, :].flatten())[0, 1]
            # L1 distance
            L1 = L1_distance_between_two_funcs(x = true_intensity_per_seq[:seq_len, :], y = expand_intensity_per_seq[:seq_len, :], \
                                               timestamp = timestamp_per_seq, resolution = opt.resolution)

            annotation = fr'r = {r}, \(\rho\) = {rho}, \(L^1\) = {L1}'
        else:
            df_intensity = pd.DataFrame.from_dict(
                    {'Time': timestamp_per_seq.flatten().cumsum(axis = -1),
                     'Intensity': expand_intensity_per_seq[:seq_len, :].flatten()})
            annotation = ''

        df_intensity_plot = pd.melt(df_intensity, 'Time')
        df_intensity_plot.columns = ['Time', ' ', 'Intensity']

        subplot_instruction = [
            {
                'plot_type': 'lineplot',
                'kwargs':
                {
                    'x':'Time',
                    'y': 'Intensity',
                    'hue': ' ',
                    'data': df_intensity_plot
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

        packed_instruction \
            = figure_instruction_generator(subplot_instruction,
                                           figure_kwargs = {
                                               'figsize': (large_graph_length, large_graph_height),
                                           })
        plot_instruction[f'intensity_{idx}'] = packed_instruction

    return plot_instruction


def plot_integral(data, timestamp, opt):
    '''
    '''

    num_events = opt.info_dict['num_events']
    color_palette = stable_palette([f'Mark {i}' for i in range(num_events)])

    plot_instruction = {}
    '''
    Part 1: the sum of intensity integrals over all markers.
    '''
    expand_integral = data['expand_integral']                                  # [batch_size, seq_len, resolution]
    mask_next = data['mask_next']                                              # [batch_size, seq_len]
    events_next = data['events_next']                                          # [batch_size, seq_len]
    time_next = data['time_next']                                              # [batch_size, seq_len]
    expand_integral = expand_integral.sum(dim = -1)                            # [batch_size, seq_len, resolution]

    packed_data = zip(*move_from_tensor_to_ndarray(expand_integral, events_next, time_next, mask_next, timestamp))
    for idx, (expand_integral_per_seq, events_next_per_seq, time_next_per_seq, mask_next_per_seq, timestamp_per_seq) \
        in enumerate(packed_data):
        seq_len = mask_next_per_seq.sum()

        df_event = pd.DataFrame.from_dict(
                {'Time': time_next_per_seq.cumsum(axis = -1), 'Point': np.zeros_like(events_next_per_seq), \
                 'Mark': [f'Mark {item}' for item in events_next_per_seq]}
        )

        df_integral = pd.DataFrame.from_dict(
                {'Time': timestamp_per_seq.flatten().cumsum(axis = -1),
                 'Integral': expand_integral_per_seq[:seq_len, :].flatten()}
        )

        subplot_instruction = [
            {
                'plot_type': 'lineplot',
                'kwargs':
                {
                    'x':'Time',
                    'y': 'Integral',
                    'data': df_integral
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
            }
        ]
        packed_instruction \
            = figure_instruction_generator(subplot_instruction,
                                           figure_kwargs = {
                                               'figsize': (large_graph_length, large_graph_height),
                                           })
        plot_instruction[f'integral_{idx}'] = packed_instruction

    return plot_instruction


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

    expand_probability = expand_probability.sum(dim = -1)                      # [batch_size, seq_len, resolution]
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

        subplot_instruction = [
            {
                'plot_type': 'lineplot',
                'kwargs':
                {
                    'x':'Time',
                    'y': 'Probability',
                    'hue': ' ',
                    'data': df_probability_plot,
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
        packed_instruction \
            = figure_instruction_generator(subplot_instruction,
                                           figure_kwargs = {
                                               'figsize': (large_graph_length, large_graph_height),
                                           })

        plot_instruction[f'probability_{idx}'] = packed_instruction

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

    num_events = opt.info_dict['num_events']
    resolution = opt.resolution
    color_palette = stable_palette([f'Mark {i}' for i in range(num_events)])

    plot_instruction = {}
    '''
    Part 1: expand intensity and expand integral
    Required plots: lineplot and scatterplot
    '''
    events_next = data['events_next']                                          # [batch_size, seq_len]
    time_next = data['time_next']                                              # [batch_size, seq_len]
    mask_next = data['mask_next']                                              # [batch_size, seq_len]
    expand_intensity = data['expand_intensity_for_each_event']                 # [batch_size, seq_len, resolution, num_events] if self.event_toggle else [batch_size, seq_len, resolution, 1]
    expand_integral = data['expand_integral_for_each_event']                   # [batch_size, seq_len, resolution, num_events] if self.event_toggle else [batch_size, seq_len, resolution, 1]
    expand_timestamp = timestamp                                               # [batch_size, seq_len, resolution]

    packed_data = zip(*move_from_tensor_to_ndarray(events_next, time_next, mask_next, expand_intensity, expand_integral, expand_timestamp))
    for idx, (events_next_per_seq, time_next_per_seq, mask_next_per_seq, expand_intensity_per_seq, \
              expand_integral_per_seq, timestamp_per_seq) in enumerate(packed_data):
        seq_len = mask_next_per_seq.sum()

        df_event = pd.DataFrame.from_dict(
                {'Time': time_next_per_seq.cumsum(axis = -1), 'Point': np.zeros_like(events_next_per_seq), \
                 'Mark': [f'Mark {item}' for item in events_next_per_seq]}
        )

        event_list = [f'Event {i}' for i in range(num_events)]
    
        df_intensity = pd.DataFrame.from_dict(
                {'Time': timestamp_per_seq.flatten().cumsum(axis = -1).repeat(num_events), 
                 'Intensity': expand_intensity_per_seq[:seq_len, :, :].flatten(), 
                 'Mark': event_list * (seq_len * resolution)}
            )
        df_integral = pd.DataFrame.from_dict(
                {'Time': timestamp_per_seq.flatten().cumsum(axis = -1).repeat(num_events), 
                 'Integral': expand_integral_per_seq[:seq_len, :, :].flatten(),
                 'Mark': event_list * (seq_len * resolution)}
            )
        
        for df, y in [(df_intensity, 'Intensity'), (df_integral, 'Integral')]:
            subplot_instruction = [
                {
                    'plot_type': 'lineplot',
                    'kwargs':
                    {
                        'x':'Time',
                        'y': y,
                        'hue': 'Mark',
                        'data': df
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
                }
            ]
            packed_instruction \
                = figure_instruction_generator(subplot_instruction,
                                               figure_kwargs = {
                                                   'figsize': (large_graph_length, large_graph_height),
                                               })
            plot_instruction[f'sub{y.lower()}_{idx}'] = packed_instruction

        '''
        Part 2: plot for spearman, pearson, and L1 distance matrix
        Required plots: heatmap
        '''
        def matrix_to_pd(matrix, index_name, column_name, value_name):
            index, column = matrix.shape
        
            # The index and column list
            index_list = [ele for ele in range(index) for _ in range(column)]
            column_list = list(range(column)) * index
        
            df = pd.DataFrame.from_dict({
                index_name: index_list,
                column_name: column_list,
                value_name: matrix.flatten()
            })
        
            df = df.pivot(index = index_name, columns = column_name, values = value_name)
        
            return df
    
        for value in ['spearman', 'pearson', 'L1']:
            selected_matrices = data[f'{value}_matrix']
            for idx, each_matrix in enumerate(selected_matrices):
                df_matrix = \
                    matrix_to_pd(each_matrix, index_name = 'Mark type', column_name = 'Mark type ', value_name = value)
                subplot_instruction = [
                    {
                        'plot_type': 'heatmap',
                        'kwargs':
                        {
                            'data': df_matrix,
                            'cmap': "YlGnBu",
                            'vmin': 0,
                            'vmax': max(1, np.max(df_matrix.values)),
                            'annot': True
                        }
                    },
                ]
                packed_instruction = figure_instruction_generator(subplot_instruction)
                plot_instruction[f'{value}_matrix_{idx}'] = packed_instruction


        '''
        Part 3: plot for Top-K accuracy
        Required plots: lineplot
        '''
        top_k = data['top_k']                                                  # [batch_size, num_events - 1]
        for idx, top_k_per_seq in enumerate(top_k):
            data_top_k_per_seq = {
                'x': np.arange(1, max(num_events, 2)),
                'y': top_k_per_seq,
            }
            df_data_top_k_per_seq = pd.DataFrame.from_dict(data_top_k_per_seq)
            subplot_instruction = [
                {
                    'set_ylim': {'bottom': -0.05, 'top': 1.05},
                },
                {
                    'plot_type': 'lineplot',
                    'kwargs':
                    {
                        'x': 'x',
                        'y': 'y',
                        'data': df_data_top_k_per_seq,
                        'markers': True
                    }
                }
            ]
            packed_instruction = figure_instruction_generator(subplot_instruction)
            plot_instruction[f'top_k_accuracy_{idx}'] = packed_instruction


        '''
        Part 4: Logarithm of MAE-E and MAE at each event
        '''
        mae_per_event_with_predict_index, mae_per_event_with_event_next = data['maes_after_event']
                                                                               # [batch_size, seq_len]
        mae = data['mae_before_event']                                         # [batch_size, seq_len]
        mask_next = data['mask_next']                                          # [batch_size, seq_len]
    
        packed_data = zip(*move_from_tensor_to_ndarray(mae, mae_per_event_with_predict_index, mae_per_event_with_event_next, mask_next))
    
        for idx, (mae_per_seq, mae_per_event_with_predict_index_per_seq, mae_per_event_with_event_next_per_seq, mask_next_per_seq) in enumerate(packed_data):
            seq_len = mask_next_per_seq.sum()
    
            data_maes_per_seq = {
                'x': list(range(seq_len)) * 3,
                'y': np.concatenate(
                    (np.log(1 + mae_per_event_with_predict_index_per_seq[:seq_len]),
                     np.log(1 + mae_per_event_with_event_next_per_seq[:seq_len]),
                     np.log(1 + mae_per_seq[:seq_len]))
                ),
                'Mark': ['MAE_k against prediction'] * seq_len +  ['MAE_k against real events'] * seq_len + ['MAE'] * seq_len
            }
            df_data_maes_per_seq = pd.DataFrame.from_dict(data_maes_per_seq)
    
            subplot_instruction = [
                {
                    'plot_type': 'lineplot',
                    'kwargs':
                    {
                        'x': 'x',
                        'y': 'y',
                        'hue': 'Mark',
                        'data': df_data_maes_per_seq,
                        'markers': True
                    }
                }
            ]
            packed_instruction = figure_instruction_generator(subplot_instruction)
            plot_instruction[f'log_mae_k_{idx}'] = packed_instruction
    

        '''
        Part 5: the value of \\sum_{m \\in M}{p^*(m)} given different history.
        '''
        probability_sum = data['probability_sum']                              # [batch_size, seq_len]
        mask_next = data['mask_next']                                          # [batch_size, seq_len]
    
        packed_data = zip(*move_from_tensor_to_ndarray(probability_sum, mask_next))
    
        for idx, (probability_sum_per_seq, mask_next_per_seq) in enumerate(packed_data):
            seq_len = mask_next_per_seq.sum()
    
            data_probability_sum_per_seq = {
                'x': np.arange(1, seq_len + 1),
                'y': probability_sum_per_seq[:seq_len]
            }
            df_data_probability_sum_per_seq = pd.DataFrame.from_dict(data_probability_sum_per_seq)
    
            subplot_instruction = [
                {
                    'set_ylim': {'bottom': -0.05, 'top': 1.05},
                },
                {
                    'plot_type': 'lineplot',
                    'kwargs':
                    {
                        'x': 'x',
                        'y': 'y',
                        'data': df_data_probability_sum_per_seq,
                        'markers': True
                    }
                }
            ]
            packed_instruction = figure_instruction_generator(subplot_instruction)
            plot_instruction[f'probability_sum_{idx}'] = subplot_instruction


        '''
        Part 6: The Logarithm of time prediction against all events
    
        '''
        tau_pred_all_event = data['tau_pred_all_event']                        # [batch_size, seq_len, num_events]
        mask_next = data['mask_next']                                          # [batch_size, seq_len]
        tau_pred_all_event, mask_next = move_from_tensor_to_ndarray(tau_pred_all_event, mask_next)
                                                                               # [batch_size, seq_len, num_events] + [batch_size, seq_len]
    
        for idx, (tau_pred_all_event_per_seq, mask_next) in enumerate(zip(tau_pred_all_event, mask_next)):
            seq_len = mask_next_per_seq.sum()
    
            data_tau_pred_all_event_per_seq = {
                'x': [ele for ele in range(seq_len) for _ in range(num_events)],
                'y': np.log(1 + tau_pred_all_event_per_seq[:seq_len, :]).flatten(),
                'Mark': [f'Mark {i}' for i in range(num_events)] * seq_len
            }
            df_data_tau_pred_all_event_per_seq = pd.DataFrame.from_dict(data_tau_pred_all_event_per_seq)
            subplot_instruction = [
                {
                    'plot_type': 'lineplot',
                    'kwargs':
                    {
                        'x': 'x',
                        'y': 'y',
                        'hue': 'Mark',
                        'data': df_data_tau_pred_all_event_per_seq,
                        'markers': True,
                        'hue_order': [f'Mark {item}' for item in range(num_events)]
                    }
                }
            ]
            packed_instruction = figure_instruction_generator(subplot_instruction)
            plot_instruction[f't_pred_all_event_{idx}'] = packed_instruction


    return plot_instruction