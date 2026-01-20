import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from einops import pack, repeat

from src.tpp.tpp_models.utils import move_from_tensor_to_ndarray, L1_distance_between_two_funcs
from src.tpp.resources.syn_tpp_utils import expand_true_probability

large_graph_length = 18
large_graph_height = 10


def plot_removed_events(data, opt):
    '''
    For simplicity, this function assumes that batch_size = 1
    '''
    
    plot_instruction = {}
    '''
    Part 1: which events are being selected and which events are discarded?
    '''
    padded_filtered_labels = data['padded_filtered_labels'].squeeze()
    padded_filtered_input_x = data['padded_filtered_input_x'].squeeze()
    padded_filtered_input_y = data['padded_filtered_input_y'].squeeze()
    padded_selected_filtered_labels = data['padded_selected_filtered_labels'].squeeze()
    padded_selected_filtered_input_x = data['padded_selected_filtered_input_x'].squeeze()
    padded_selected_filtered_input_y = data['padded_selected_filtered_input_y'].squeeze()
    expand_x = data['expand_x']
    expand_y = data['expand_y']

    df_filtered_events = pd.DataFrame.from_dict(
                {'x': padded_filtered_input_x, 'y': padded_filtered_input_y, \
                 'label': padded_filtered_labels, 'status': ['Filtered',] * padded_filtered_input_y.shape[0]}
        )

    df_selected_events = pd.DataFrame.from_dict(
                {'x': padded_selected_filtered_input_x, 'y': padded_selected_filtered_input_y, \
                 'label': padded_selected_filtered_labels, 'status': ['Selected',] * padded_selected_filtered_input_y.shape[0]}
        )

    df_model = pd.DataFrame.from_dict(
                {'x': expand_x, 'y': expand_y}
        )
    
    df_events = pd.concat((df_filtered_events, df_selected_events), ignore_index = True)
   
    instruction = [
        {
            'plot_type': 'scatterplot',
            'length': large_graph_length,
            'height': large_graph_height,
            'kwargs':
            {
                'data': df_events,
                'x': 'x',
                'y': 'y',
                'hue': 'label',
                'hue_order': [i for i in range(opt.info_dict['num_events'])],
                'style': 'status',
                's': 100,
                'markers': True,
                'palette': sns.color_palette("husl", opt.info_dict['num_events'])
            }
        },
        {
            'plot_type': 'lineplot',
            'length': large_graph_length,
            'height': large_graph_height,
            'kwargs':
            {
                'data': df_model,
                'x': 'x',
                'y': 'y',
                'color': 'b'
            }
        }
    ]

    plot_instruction['selected_and_filtered_event'] = instruction

    return plot_instruction


def plot_debug(data, timestamp, opt):
    '''
    What is inside dict data?
    1. expand_intensity_for_each_event  shape: [batch_size, seq_len, resolution, num_events]
    2. expand_integral_for_each_event   shape: [batch_size, seq_len, resolution, num_events]
    3. spearman, pearson, and L1 distance matrix.
    4. macro-f1: measure the event prediction performance without time prediction.
    5. top_k: measure the event prediction performance without time prediction.
    6. probability_sum: the value of \int_{t_l}^{+infty}{p(m, \tau)d\tau}
    7. tau_pred_all_event: The time prediction of all events, with p(m) known.
    8. mae_before_event: as known as MAE.
    9. maes_after_event_avg: contains mae_per_event_with_predict_index_avg and mae_per_event_with_event_next_avg
    10. maes_after_event: contains mae_per_event_with_predict_index and mae_per_event_with_event_next
    11. event_next: 
    12. time_next:
    '''

    plot_instruction = {}
    num_events = data['expand_probability_for_each_event'].shape[-1]
    resolution = data['expand_probability_for_each_event'].shape[-2]

    '''
    Part 1: expand intensity and expand integral
    Required plots: lineplot and scatterplot
    '''
    events_next = data['events_next']                                          # [batch_size, seq_len]
    time_next = data['time_next']                                              # [batch_size, seq_len]
    mask_next = data['mask_next']                                              # [batch_size, seq_len]
    expand_probability = data['expand_probability_for_each_event']             # [batch_size, seq_len, resolution, num_events]
    expand_timestamp = timestamp                                               # [batch_size, seq_len, resolution]

    packed_data = zip(*move_from_tensor_to_ndarray(events_next, time_next, mask_next, expand_probability, expand_timestamp))
    for idx, (events_next_per_seq, time_next_per_seq, mask_next_per_seq, expand_probability_per_seq, \
              timestamp_per_seq) in enumerate(packed_data):
        seq_len = mask_next_per_seq.sum()

        df_event = pd.DataFrame.from_dict(
                {'Time': time_next_per_seq.cumsum(axis = -1), 'Point': np.zeros_like(events_next_per_seq), \
                 'Event': [f'Event {item}' for item in events_next_per_seq]}
        )

        event_list = [f'Event {i}' for i in range(num_events)]
    
        df_probability = pd.DataFrame.from_dict(
                {'Time': timestamp_per_seq.flatten().cumsum(axis = -1).repeat(num_events), 
                 'Probability': expand_probability_per_seq[:seq_len, :, :].flatten(), 
                 'Event': event_list * (seq_len * resolution)}
            )
        
        for df, y in [(df_probability, 'Probability'),]:
            subplot_instruction = [
                {
                    'plot_type': 'lineplot',
                    'length': large_graph_length,
                    'height': large_graph_height,
                    'kwargs':
                    {
                        'x':'Time',
                        'y': y,
                        'hue': 'Event',
                        'data': df
                    }
                },
                {
                    'plot_type': 'scatterplot',
                    'length': large_graph_length,
                    'height': large_graph_height,
                    'kwargs':
                    {
                        'x': 'Time',
                        'y': 'Point',
                        'data': df_event,
                        'palette': 'pastel',
                        'hue': 'Event'
                    }
                }
            ]
            plot_instruction[f'sub{y.lower()}_{idx}'] = subplot_instruction

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
                matrix_to_pd(each_matrix, index_name = 'Event type', column_name = 'Event type ', value_name = value)
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
            plot_instruction[f'{value}_matrix_{idx}'] = subplot_instruction

    '''
    Part 3: plot for Top-K accuracy
    Required plots: lineplot
    '''
    top_k = data['top_k']                                                      # [batch_size, num_events - 1]
    for idx, top_k_per_seq in enumerate(top_k):
        data_top_k_per_seq = {
            'x': np.arange(1, num_events),
            'y': top_k_per_seq,
            'marks': 'Top-K accuracy'
        }
        df_data_top_k_per_seq = pd.DataFrame.from_dict(data_top_k_per_seq)
        sub_plot_instruction = [
            {
                'plot_type': 'lineplot',
                'kwargs':
                {
                    'x': 'x',
                    'y': 'y',
                    'hue': 'marks',
                    'data': df_data_top_k_per_seq,
                    'markers': True
                }
            }
        ]
        plot_instruction[f'top_k_accuracy_{idx}'] = sub_plot_instruction

    '''
    Part 4: The Logarithm of time prediction against all events

    '''
    tau_pred_all_event = data['tau_pred_all_event']                            # [batch_size, seq_len, num_events]
    mask_next = data['mask_next']                                              # [batch_size, seq_len]
    tau_pred_all_event, mask_next = move_from_tensor_to_ndarray(tau_pred_all_event, mask_next)
                                                                               # [batch_size, seq_len, num_events] + [batch_size, seq_len]

    for idx, (tau_pred_all_event_per_seq, mask_next) in enumerate(zip(tau_pred_all_event, mask_next)):
        seq_len = mask_next_per_seq.sum()

        data_tau_pred_all_event_per_seq = {
            'x': [ele for ele in range(seq_len) for _ in range(num_events)],
            'y': np.log(1 + tau_pred_all_event_per_seq[:seq_len, :]).flatten(),
            'marks': [f'Event {i}' for i in range(num_events)] * seq_len
        }
        df_data_tau_pred_all_event_per_seq = pd.DataFrame.from_dict(data_tau_pred_all_event_per_seq)
        sub_plot_instruction = [
            {
                'plot_type': 'lineplot',
                'kwargs':
                {
                    'x': 'x',
                    'y': 'y',
                    'hue': 'marks',
                    'data': df_data_tau_pred_all_event_per_seq,
                    'markers': True
                }
            }
        ]
        plot_instruction[f't_pred_all_event_{idx}'] = sub_plot_instruction


    '''
    Part 5: Logarithm of MAE-E and MAE at each event
    '''
    mae_per_event_with_predict_index, mae_per_event_with_event_next = data['maes_after_event']
                                                                               # [batch_size, seq_len]
    mae = data['mae_before_event']                                             # [batch_size, seq_len]
    mask_next = data['mask_next']                                              # [batch_size, seq_len]

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
            'marks': ['MAE_k against prediction'] * seq_len +  ['MAE_k against real events'] * seq_len + ['MAE'] * seq_len
        }
        df_data_maes_per_seq = pd.DataFrame.from_dict(data_maes_per_seq)

        sub_plot_instruction = [
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
        plot_instruction[f'log_mae_k_{idx}'] = sub_plot_instruction
    

    '''
    Part 6: the value of \sum_{m \in M}{p^*(m)} given different history.
    '''
    probability_sum = data['probability_sum']                                  # [batch_size, seq_len]
    mask_next = data['mask_next']                                              # [batch_size, seq_len]

    packed_data = zip(*move_from_tensor_to_ndarray(probability_sum, mask_next))

    for idx, (probability_sum_per_seq, mask_next_per_seq) in enumerate(packed_data):
        seq_len = mask_next_per_seq.sum()

        data_probability_sum_per_seq = {
            'x': np.arange(1, seq_len + 1),
            'y': probability_sum_per_seq[:seq_len]
        }
        df_data_probability_sum_per_seq = pd.DataFrame.from_dict(data_probability_sum_per_seq)

        sub_plot_instruction = [
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
        plot_instruction[f'probability_sum_{idx}'] = sub_plot_instruction
    
    '''
    Part 7: expand intensity and expand integral on sampled event sequences.
    Required plots: lineplot and scatterplot
    '''
    sampled_events_next_event_time = data['sampled_events_next_event_time']    # [batch_size, seq_len]
    sampled_time_next_event_time = data['sampled_time_next_event_time']        # [batch_size, seq_len]
    sampled_mask_next_event_time = data['sampled_mask_next_event_time']        # [batch_size, seq_len]
    sampled_expand_subprobability_event_time = data['sampled_subprobability_event_time']
                                                                               # [batch_size, seq_len, resolution, num_events]
    sampled_expand_timestamp_event_time = data['sampled_timestamp_event_time'] # [batch_size, seq_len, resolution]

    sampled_expand_probability_event_time = sampled_expand_subprobability_event_time.sum(dim = -1)
                                                                               # [batch_size, seq_len, resolution]

    packed_data = zip(*move_from_tensor_to_ndarray(sampled_events_next_event_time, sampled_time_next_event_time, \
                                                   sampled_mask_next_event_time, sampled_expand_probability_event_time, \
                                                   sampled_expand_subprobability_event_time, sampled_expand_timestamp_event_time))
    for idx, (sampled_events_next_per_seq, sampled_time_next_per_seq, sampled_mask_next_per_seq, sampled_expand_probability_per_seq, \
              sampled_expand_subprobability_per_seq, sampled_timestamp_per_seq) in enumerate(packed_data):
        seq_len = sampled_mask_next_per_seq.sum()

        df_event = pd.DataFrame.from_dict(
                {'Time': sampled_time_next_per_seq.cumsum(axis = -1), 'Point': np.zeros_like(sampled_events_next_per_seq), \
                 'Event': [f'Event {item}' for item in sampled_events_next_per_seq]}
        )

        event_list = [f'Event {i}' for i in range(num_events)]
    
        df_subprobability = pd.DataFrame.from_dict(
                {'Time': sampled_timestamp_per_seq.flatten().cumsum(axis = -1).repeat(num_events), 
                 'Probability': sampled_expand_subprobability_per_seq[:seq_len, :, :].flatten(), 
                 'Event': event_list * (seq_len * resolution)}
            )

        df_probability = pd.DataFrame.from_dict(
                {'Time': sampled_timestamp_per_seq.flatten().cumsum(axis = -1), 
                 'Probability': sampled_expand_probability_per_seq[:seq_len, :].flatten()}
            )

        df_probability_plot = pd.melt(df_probability, 'Time')
        df_probability_plot.columns = ['Time', ' ', 'Probability']
        
        '''
        Probability distribution of the sampled sequence.
        '''
        subplot_instruction = [
            {
                'plot_type': 'lineplot',
                'length': large_graph_length,
                'height': large_graph_height,
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
                'length': large_graph_length,
                'height': large_graph_height,
                'kwargs':
                {
                    'x': 'Time',
                    'y': 'Point',
                    'data': df_event,
                    'palette': 'pastel',
                    'hue': 'Event'
                }
            }
        ]
        plot_instruction[f'sampled_probability_{idx}_event_time'] = subplot_instruction

        '''
        sub-probability distribution of the sampled sequence.
        '''
        for df, y in [(df_subprobability, 'Probability'),]:
            subplot_instruction = [
                {
                    'plot_type': 'lineplot',
                    'length': large_graph_length,
                    'height': large_graph_height,
                    'kwargs':
                    {
                        'x':'Time',
                        'y': y,
                        'hue': 'Event',
                        'data': df
                    }
                },
                {
                    'plot_type': 'scatterplot',
                    'length': large_graph_length,
                    'height': large_graph_height,
                    'kwargs':
                    {
                        'x': 'Time',
                        'y': 'Point',
                        'data': df_event,
                        'palette': 'pastel',
                        'hue': 'Event'
                    }
                }
            ]
            plot_instruction[f'sampled_sub{y.lower()}_{idx}_event_time'] = subplot_instruction


    sampled_events_next_time_event = data['sampled_events_next_time_event']    # [batch_size, seq_len]
    sampled_time_next_time_event = data['sampled_time_next_time_event']        # [batch_size, seq_len]
    sampled_mask_next_time_event = data['sampled_mask_next_time_event']        # [batch_size, seq_len]
    sampled_expand_subprobability_time_event = data['sampled_subprobability_time_event']
                                                                               # [batch_size, seq_len, resolution, num_events]
    sampled_expand_timestamp_time_event = data['sampled_timestamp_time_event'] # [batch_size, seq_len, resolution]

    sampled_expand_probability_time_event = sampled_expand_subprobability_time_event.sum(dim = -1)
                                                                               # [batch_size, seq_len, resolution]

    packed_data = zip(*move_from_tensor_to_ndarray(sampled_events_next_time_event, sampled_time_next_time_event, \
                                                   sampled_mask_next_time_event, sampled_expand_probability_time_event, \
                                                   sampled_expand_subprobability_time_event, sampled_expand_timestamp_time_event))
    for idx, (sampled_events_next_per_seq, sampled_time_next_per_seq, sampled_mask_next_per_seq, sampled_expand_probability_per_seq, \
              sampled_expand_subprobability_per_seq, sampled_timestamp_per_seq) in enumerate(packed_data):
        seq_len = sampled_mask_next_per_seq.sum()

        df_event = pd.DataFrame.from_dict(
                {'Time': sampled_time_next_per_seq.cumsum(axis = -1), 'Point': np.zeros_like(sampled_events_next_per_seq), \
                 'Event': [f'Event {item}' for item in sampled_events_next_per_seq]}
        )

        event_list = [f'Event {i}' for i in range(num_events)]
    
        df_subprobability = pd.DataFrame.from_dict(
                {'Time': sampled_timestamp_per_seq.flatten().cumsum(axis = -1).repeat(num_events), 
                 'Probability': sampled_expand_subprobability_per_seq[:seq_len, :, :].flatten(), 
                 'Event': event_list * (seq_len * resolution)}
            )

        df_probability = pd.DataFrame.from_dict(
                {'Time': sampled_timestamp_per_seq.flatten().cumsum(axis = -1), 
                 'Probability': sampled_expand_probability_per_seq[:seq_len, :].flatten()}
            )

        df_probability_plot = pd.melt(df_probability, 'Time')
        df_probability_plot.columns = ['Time', ' ', 'Probability']
        
        '''
        Probability distribution of the sampled sequence.
        '''
        subplot_instruction = [
            {
                'plot_type': 'lineplot',
                'length': large_graph_length,
                'height': large_graph_height,
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
                'length': large_graph_length,
                'height': large_graph_height,
                'kwargs':
                {
                    'x': 'Time',
                    'y': 'Point',
                    'data': df_event,
                    'palette': 'pastel',
                    'hue': 'Event'
                }
            }
        ]
        plot_instruction[f'sampled_probability_{idx}_time_event'] = subplot_instruction

        '''
        sub-probability distribution of the sampled sequence.
        '''
        for df, y in [(df_subprobability, 'Probability'),]:
            subplot_instruction = [
                {
                    'plot_type': 'lineplot',
                    'length': large_graph_length,
                    'height': large_graph_height,
                    'kwargs':
                    {
                        'x':'Time',
                        'y': y,
                        'hue': 'Event',
                        'data': df
                    }
                },
                {
                    'plot_type': 'scatterplot',
                    'length': large_graph_length,
                    'height': large_graph_height,
                    'kwargs':
                    {
                        'x': 'Time',
                        'y': 'Point',
                        'data': df_event,
                        'palette': 'pastel',
                        'hue': 'Event'
                    }
                }
            ]
            plot_instruction[f'sampled_sub{y.lower()}_{idx}_time_event'] = subplot_instruction


    return plot_instruction
