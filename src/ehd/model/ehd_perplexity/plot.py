import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from einops import pack, repeat

from src.TPP.model.utils import move_from_tensor_to_ndarray, L1_distance_between_two_funcs
from src.TPP.resources.syn_tpp_utils import expand_true_probability

large_graph_length = 18
large_graph_height = 10


def plot_removed_events(data, opt):
    '''
    For simplicity, this function assumes that batch_size = 1
    '''
    
    plot_instruction = {}
    '''
    Part 1: the sum of probability distributions over all markers.
    '''
    percentage_remained_events = data['percentage_remained_events']
    L_sp = data['L_sp']
    L_sp_r = data['L_sp_r']
    L_rp = data['L_rp']
    L_rp_r = data['L_rp_r']
    events_history = data['events_history']
    events_future = data['events_future']
    time_history = data['time_history']
    time_future = data['time_future']
    filter_mask = data['filter_mask']

    filter_for_removal = filter_mask[:, :, 1]                                  # [batch_size, seq_len_h + seq_len_x + 2]
    filter_for_left = filter_mask[:, :, 0]                                     # [batch_size, seq_len_h + seq_len_x + 2]

    filter_for_removal[:, 0] = 1                                               # [batch_size, seq_len_h + seq_len_x + 2]
    filter_for_removal[:, -opt.info_dict['length_of_x'] - 1:] = 0              # [batch_size, seq_len_h + seq_len_x + 2]

    filter_for_left[:, -opt.info_dict['length_of_x'] - 1:] = 0                 # [batch_size, seq_len_h + seq_len_x + 2]
    filter_for_left[:, 0] = 1                                                  # [batch_size, seq_len_h + seq_len_x + 2]

    generated_mask_probability = data['generated_mask_probability']            # [batch_size, seq_len_h + 1, 2]

    input_events, _ = pack((events_history, events_future), 'b *')             # [batch_size, seq_len_h + seq_len_x + 2]
    input_time, _ = pack((time_history, time_future), 'b *')                   # [batch_size, seq_len_h + seq_len_x + 2]
    cumulative_input_time = input_time.cumsum(axis = -1)                       # [batch_size, seq_len_h + seq_len_x + 2]

    if opt.log_time:
        cumulative_input_time = cumulative_input_time.sqrt()                   # [batch_size, seq_len_h + seq_len_x + 2]

    # All sequence starts with a fake event. This fake event will appear in both padded_filtered_events and padded_filtered_removed_events.
    # Please carefully remove these fake events.
    removed_events = input_events[filter_for_removal == 1]                     # [removed_events]
    time_of_removed_events = cumulative_input_time[filter_for_removal == 1]    # [removed_events]

    left_events = input_events[filter_for_left == 1]                           # [left_events]
    time_of_left_events = cumulative_input_time[filter_for_left == 1]          # [left_events]

    time_of_x = cumulative_input_time[0, -opt.info_dict['length_of_x'] - 1:]   # [seq_len_x + 1]

    df_removed_event = pd.DataFrame.from_dict(
                {'Time': time_of_removed_events, 'Point': np.ones_like(time_of_removed_events) * (-0.05) , \
                 'Event': [f'Event {item}' for item in removed_events], 'filter': [r'$\\in \\mathcal{H}_{r,o,t_l}$'] * filter_for_removal.sum().int().item()}
        )

    df_left_event = pd.DataFrame.from_dict(
                {'Time': time_of_left_events, 'Point': np.ones_like(time_of_left_events) * (-0.1), \
                 'Event': [f'Event {item}' for item in left_events], 'filter': [r'$\\in \\mathcal{H}_{l,o,t_l}$'] * filter_for_left.sum().int().item()}
        )

    df_x = pd.DataFrame.from_dict(
                {'Time': time_of_x, 'Point': np.ones_like(time_of_x) * (-0.075), \
                 'Event': [f'Event {item}' for item in events_future.squeeze()], \
                 'filter': [r'$\\in \\mathbf{x}_{o}$'] * (opt.info_dict['length_of_x'] + 1)}
        )

    df_event = pd.concat((df_left_event, df_removed_event, df_x), axis = 0, ignore_index = True)
    
    df_removed_probability = pd.DataFrame.from_dict(
                {'Time': cumulative_input_time[0, :opt.info_dict['length_of_h'] + 1], \
                 'Probability': generated_mask_probability[:, :, 1].squeeze()}
        )
    
    df_half_probability = pd.DataFrame.from_dict(
                {'Time': cumulative_input_time[0, :opt.info_dict['length_of_h'] + 1], \
                 'Probability': np.ones(opt.info_dict['length_of_h'] + 1) * 0.5
                }
        )
    
    annotation = fr'$dppl_r$ = {L_sp}, $dppl_{{rr}}$ = {L_sp_r}, $dppl_l$ = {L_rp}, $dppl_{{lr}}$ = {L_rp_r} with ${percentage_remained_events * 100}\%$ events left. len($\\mathcal{{H}}_{{o,t_l}}$) = {opt.info_dict["length_of_h"]}, len($\\mathbf{{x}}_{{o}}$) = {opt.info_dict["length_of_x"]}'
    
    instruction = [
        {
            'plot_type': 'lineplot',
            'length': large_graph_length,
            'height': large_graph_height,
            'kwargs':
            {
                'data': df_removed_probability,
                'x': 'Time',
                'y': 'Probability',
                'marker': 'o'
            }
        },
        {
            'plot_type': 'lineplot',
            'length': large_graph_length,
            'height': large_graph_height,
            'kwargs':
            {
                'data': df_half_probability,
                'x': 'Time',
                'y': 'Probability'
            }
        },
        {
            'plot_type': 'scatterplot',
            'length': large_graph_length,
            'height': large_graph_height,
            'kwargs':
            {
                'data': df_event,
                'x': 'Time',
                'y': 'Point',
                'hue': 'Event',
                'hue_order': [f'Event {i}' for i in range(opt.info_dict['num_events'] + 1)],
                'style': 'filter',
                's': 100,
                'markers': True,
                'palette': sns.color_palette("husl", opt.info_dict['num_events'] + 1)
            }
        },
        {
            'plot_type': 'text',
            'kwargs':
            {
                'x': 0.5, 
                'y': -0.125,
                'verticalalignment': 'center',
                'horizontalalignment': 'center',
                's': annotation,
                'fontsize': 12,
            }
        }
    ]

    plot_instruction['removed_and_left_event'] = instruction

    return plot_instruction


def plot_debug(data, timestamp, opt):
    '''
    What is inside dict data?
    1. expand_intensity_for_each_event  shape: [batch_size, seq_len, resolution, num_events]
    2. expand_integral_for_each_event   shape: [batch_size, seq_len, resolution, num_events]
    3. spearman, pearson, and L1 distance matrix.
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
    Part 6: the value of \\sum_{m \\in M}{p^*(m)} given different history.
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
