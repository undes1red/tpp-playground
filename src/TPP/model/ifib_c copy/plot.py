import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.toolbox.misc import move_from_tensor_to_ndarray, stable_palette, save_fig, get_logger
from src.toolbox.metrics import L1_distance_between_two_funcs

from src.TPP.model.utils import draw_intensity_integral_and_probability, draw_intensity_integral_per_mark, draw_heatmap, draw_lineplot
from src.TPP.resources.syn_tpp_utils import expand_true_probability

logger = get_logger(__name__)


'''
What is a plot_instruction?
plot_instruction is a dict. Each key in the plot_instruction is the plot name, also the file name of the stored plot. 
The value corresponding to the key is a list. This list comprises dicts. Each dict is an instruction defining a plot.
The framework will loop over the instruction list, so you can use multiple dicts to draw a rather complicated plot.
'''

def generate_probability_figure(data, timestamp, opt):
    '''

    '''
    num_events = opt.info_dict['num_events']
    color_palette = stable_palette([f'Mark {i}' for i in range(num_events)])

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
        start_time = time_next_per_seq[:seq_len].cumsum(axis = -1)
        timestamp_offset = np.concatenate((np.array([0.]), start_time[:-1]), axis = -1)
        timestamp_per_seq[:, 0] = timestamp_per_seq[:, 0] + 1e-30
        timestamp_per_seq = timestamp_per_seq + np.expand_dims(timestamp_offset, axis = -1)

        df_event = pd.DataFrame.from_dict(
                {'Time': start_time, 'Point': np.zeros_like(events_next_per_seq), \
                 'Mark': [f'Mark {item}' for item in events_next_per_seq]}
        )

        annotation = None
        if true_probability_per_seq is not None:
            df_probability_plot = pd.DataFrame.from_dict(
                {'Time': timestamp_per_seq.flatten(),
                 'Predicted': expand_probability_per_seq[:seq_len, :].flatten(),
                 'Truth': true_probability_per_seq[:seq_len, :].flatten()}
            )

            # Spearman correlation
            rho = spearmanr(a = true_probability_per_seq[:seq_len, :].flatten(), b = expand_probability_per_seq[:seq_len, :].flatten())[0]
            # Pearson correlation
            r = np.corrcoef(x = true_probability_per_seq[:seq_len, :].flatten(), y = expand_probability_per_seq[:seq_len, :].flatten())[0, 1]
            # L1 distance
            L1 = L1_distance_between_two_funcs(x = true_probability_per_seq[:seq_len, :], y = expand_probability_per_seq[:seq_len, :], \
                                               timestamp = timestamp_per_seq)

            annotation = '\n'.join((fr'$r = {r}$', fr'$\rho = {rho}$', fr'$L^1 = {L1}$'))
        else:
            df_probability_plot = pd.DataFrame.from_dict(
                {'Time': timestamp_per_seq.flatten(),
                 'Predicted': expand_probability_per_seq[:seq_len, :].flatten()}
            )

        fig = draw_intensity_integral_and_probability(df_probability_plot, df_event, annotation, 'Probability', color_palette, num_events)
        save_fig(fig, opt.plot_store_dir_for_this_batch, f'probability_{idx}.pdf')
        logger.info(f'probability_{idx} drawed and saved in {opt.plot_store_dir_for_this_batch}!')

    return 0


def generate_debug_figure(data, timestamp, opt):
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
        start_time = time_next_per_seq[:seq_len].cumsum(axis = -1)
        timestamp_offset = np.concatenate((np.array([0.]), start_time[:-1]), axis = -1)
        timestamp_per_seq[:, 0] = timestamp_per_seq[:, 0] + 1e-30
        timestamp_per_seq = timestamp_per_seq + np.expand_dims(timestamp_offset, axis = -1)

        '''
        Figure 1 and 2: Mark-wise intensity and integral function.
        Required plots: lineplot
        '''
        df_event = pd.DataFrame.from_dict(
                {'Time': start_time, 'Point': np.zeros_like(events_next_per_seq), \
                 'Mark': [f'Mark {item}' for item in events_next_per_seq]}
        )

        event_list = [f'Mark {i}' for i in range(num_events)]
    
        df_intensity = pd.DataFrame.from_dict(
                {'Time': timestamp_per_seq.flatten().repeat(num_events), 
                 'Intensity': expand_intensity_per_seq[:seq_len, :, :].flatten(), 
                 'Mark': event_list * (seq_len * resolution)}
            )
        df_integral = pd.DataFrame.from_dict(
                {'Time': timestamp_per_seq.flatten().repeat(num_events), 
                 'Integral': expand_integral_per_seq[:seq_len, :, :].flatten(),
                 'Mark': event_list * (seq_len * resolution)}
            )
        
        fig1 = draw_intensity_integral_per_mark(df_intensity, df_event, 'Intensity', color_palette, num_events)
        save_fig(fig1, opt.plot_store_dir_for_this_batch, f'mark_wise_intensity_{idx}.pdf')
        logger.info(f'mark_wise_intensity_{idx} drawed and saved in {opt.plot_store_dir_for_this_batch}!')

        fig2 = draw_intensity_integral_per_mark(df_integral, df_event, 'Integral', color_palette, num_events)
        save_fig(fig2, opt.plot_store_dir_for_this_batch, f'mark_wise_integral_{idx}.pdf')
        logger.info(f'mark_wise_integral_{idx} drawed and saved in {opt.plot_store_dir_for_this_batch}!')


    '''
    Part 3, 4, 5: plot for spearman, pearson, and L1 distance matrix
    Required plots: heatmap
    '''
    for value in ['spearman', 'pearson', 'L1']:
        matrices = data[f'{value}_matrix']
        for idx, matrix in enumerate(matrices):
            fig = draw_heatmap(matrix, 'Mark type', 'Mark type ', value, {'font.size': 18, 'figure.figsize': (5, 5)})
            save_fig(fig, opt.plot_store_dir_for_this_batch, f'{value}_{idx}.pdf')
            logger.info(f'{value}_{idx} drawed and saved in {opt.plot_store_dir_for_this_batch}!')


    '''
    Part 6: plot for Top-K accuracy
    Required plots: lineplot
    '''
    top_k = data['top_k']                                                      # [batch_size, num_events - 1]
    for idx, top_k_per_seq in enumerate(top_k):
        data_top_k_per_seq = {
            'K': np.arange(1, max(num_events, 2)),
            'Accuracy': top_k_per_seq,
        }

        fig6 = draw_lineplot(data = data_top_k_per_seq, x = 'K', y = 'Accuracy', figure_kwargs = {'font.size': 18, 'figure.figsize': (5, 5)})
        ax = fig6.gca()
        ax.set_ylim(bottom = -0.05, top = 1.05)

        save_fig(fig6, opt.plot_store_dir_for_this_batch, f'topk_{idx}.pdf')
        logger.info(f'Top-K_{idx} drawed and saved in {opt.plot_store_dir_for_this_batch}!')


    '''
    Part 7: Logarithm of MAE at each event
    '''
    mae_per_event_with_predict_index, mae_per_event_with_event_next = data['maes_after_event']
                                                                               # [batch_size, seq_len]
    mae = data['mae_before_event']                                             # [batch_size, seq_len]
    mask_next = data['mask_next']                                              # [batch_size, seq_len]
    
    packed_data = zip(*move_from_tensor_to_ndarray(mae, mae_per_event_with_predict_index, mae_per_event_with_event_next, mask_next))
    for idx, (mae_per_seq, mae_per_event_with_predict_index_per_seq, mae_per_event_with_event_next_per_seq, mask_next_per_seq) in enumerate(packed_data):
        seq_len = mask_next_per_seq.sum()

        data_maes_per_seq = {
            'Event Index': list(range(seq_len)) * 3,
            r'$\log(1 + \mathrm{MAE})$': np.concatenate(
                (np.log(1 + mae_per_event_with_predict_index_per_seq[:seq_len]),
                 np.log(1 + mae_per_event_with_event_next_per_seq[:seq_len]),
                 np.log(1 + mae_per_seq[:seq_len]))
            ),
            'Mark': ['MAE against prediction'] * seq_len +  ['MAE against real events'] * seq_len + ['MAE'] * seq_len
        }

        fig7 = draw_lineplot(data = data_maes_per_seq, x = 'Event Index', y = r'$\log(1 + \mathrm{MAE})$', hue = 'Mark', \
                             figure_kwargs = {'font.size': 18, 'figure.figsize': (5, 5)})
        save_fig(fig7, opt.plot_store_dir_for_this_batch, f'MAE_{idx}.pdf')
        logger.info(f'MAE_{idx} drawed and saved in {opt.plot_store_dir_for_this_batch}!')


    '''
    Part 8: the value of \\sum_{m \\in M}{p^*(m)} given different history.
    '''
    probability_sum = data['probability_sum']                                  # [batch_size, seq_len]
    mask_next = data['mask_next']                                              # [batch_size, seq_len]

    packed_data = zip(*move_from_tensor_to_ndarray(probability_sum, mask_next))
    for idx, (probability_sum_per_seq, mask_next_per_seq) in enumerate(packed_data):
        seq_len = mask_next_per_seq.sum()

        data_probability_sum_per_seq = {
            'Event Index': np.arange(1, seq_len + 1),
            r'$\sum_{m \in M}{p(m)}$': probability_sum_per_seq[:seq_len]
        }

        fig8 = draw_lineplot(data = data_probability_sum_per_seq, x = 'Event Index', y = r'$\sum_{m \in M}{p(m)}$', \
                             figure_kwargs = {'font.size': 18, 'figure.figsize': (5, 5)})
        ax = fig8.gca()
        ax.set_ylim(bottom =  -0.05, top = 1.05)
        save_fig(fig8, opt.plot_store_dir_for_this_batch, f'sum_of_p_{idx}.pdf')
        logger.info(f'sum_of_p_m_{idx} drawed and saved in {opt.plot_store_dir_for_this_batch}!')


    '''
    Part 9: The Logarithm of time prediction against all events
    '''
    tau_pred_all_event = data['tau_pred_all_event']                            # [batch_size, seq_len, num_events]
    mask_next = data['mask_next']                                              # [batch_size, seq_len]
    tau_pred_all_event, mask_next = move_from_tensor_to_ndarray(tau_pred_all_event, mask_next)
                                                                               # [batch_size, seq_len, num_events] + [batch_size, seq_len]
    for idx, (tau_pred_all_event_per_seq, mask_next) in enumerate(zip(tau_pred_all_event, mask_next)):
        seq_len = mask_next_per_seq.sum()

        data_tau_pred_all_event_per_seq = {
            'Event Index': [ele for ele in range(seq_len) for _ in range(num_events)],
            r'$\log(1 + t_p)$': np.log(1 + tau_pred_all_event_per_seq[:seq_len, :]).flatten(),
            'Mark': [f'Mark {i}' for i in range(num_events)] * seq_len
        }

        fig9 = draw_lineplot(data = data_tau_pred_all_event_per_seq, x = 'Event Index', y = r'$\log(1 + t_p)$', \
                             hue = 'Mark', figure_kwargs = {'font.size': 18, 'figure.figsize': (5, 5)})
        save_fig(fig9, opt.plot_store_dir_for_this_batch, f'log_pred_time_{idx}.pdf')
        logger.info(f'log_pred_time_{idx} drawed and saved in {opt.plot_store_dir_for_this_batch}!')


    '''
    Part 10 and 11: expand intensity and expand integral on sampled event sequences.
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
        start_time = sampled_time_next_per_seq[:seq_len].cumsum(axis = -1)
        timestamp_offset = np.concatenate((np.array([0.]), start_time[:-1]), axis = -1)
        sampled_timestamp_per_seq[:, 0] = sampled_timestamp_per_seq[:, 0] + 1e-30
        sampled_timestamp_per_seq = sampled_timestamp_per_seq + np.expand_dims(timestamp_offset, axis = -1)

        df_event = pd.DataFrame.from_dict(
                {'Time': start_time, 'Point': np.zeros_like(sampled_events_next_per_seq), \
                 'Mark': [f'Mark {item}' for item in sampled_events_next_per_seq]}
        )

        event_list = [f'Mark {i}' for i in range(num_events)]
    
        df_subprobability = pd.DataFrame.from_dict(
                {'Time': sampled_timestamp_per_seq.flatten().repeat(num_events), 
                 'Probability': sampled_expand_subprobability_per_seq[:seq_len, :, :].flatten(), 
                 'Mark': event_list * (seq_len * resolution)}
            )

        df_probability = pd.DataFrame.from_dict(
                {'Time': sampled_timestamp_per_seq.flatten(), 
                 'Probability': sampled_expand_probability_per_seq[:seq_len, :].flatten()}
            )

        fig10 = draw_intensity_integral_per_mark(df_subprobability, df_event, 'Probability', color_palette, num_events)
        save_fig(fig10, opt.plot_store_dir_for_this_batch, f'sample_mark_time_mark_wise_probability_{idx}.pdf')
        logger.info(f'mark_wise_probability_{idx} drawed and saved in {opt.plot_store_dir_for_this_batch}!')

        fig11 = draw_intensity_integral_and_probability(df_probability, df_event, None, 'Probability', color_palette, num_events)
        save_fig(fig11, opt.plot_store_dir_for_this_batch, f'sample_mark_time_probability_{idx}.pdf')
        logger.info(f'mark_wise_integral_{idx} drawed and saved in {opt.plot_store_dir_for_this_batch}!')


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
        start_time = sampled_time_next_per_seq[:seq_len].cumsum(axis = -1)
        timestamp_offset = np.concatenate((np.array([0.]), start_time[:-1]), axis = -1)
        sampled_timestamp_per_seq[:, 0] = sampled_timestamp_per_seq[:, 0] + 1e-30
        sampled_timestamp_per_seq = sampled_timestamp_per_seq + np.expand_dims(timestamp_offset, axis = -1)

        df_event = pd.DataFrame.from_dict(
                {'Time': start_time, 'Point': np.zeros_like(sampled_events_next_per_seq), \
                 'Mark': [f'Mark {item}' for item in sampled_events_next_per_seq]}
        )

        event_list = [f'Mark {i}' for i in range(num_events)]
    
        df_subprobability = pd.DataFrame.from_dict(
                {'Time': sampled_timestamp_per_seq.flatten().repeat(num_events),
                 'Probability': sampled_expand_subprobability_per_seq[:seq_len, :, :].flatten(), 
                 'Mark': event_list * (seq_len * resolution)}
            )

        df_probability = pd.DataFrame.from_dict(
                {'Time': sampled_timestamp_per_seq.flatten(),
                 'Probability': sampled_expand_probability_per_seq[:seq_len, :].flatten()}
            )

        fig12 = draw_intensity_integral_per_mark(df_subprobability, df_event, 'Probability', color_palette, num_events)
        save_fig(fig12, opt.plot_store_dir_for_this_batch, f'sample_time_mark_mark_wise_probability_{idx}.pdf')
        logger.info(f'mark_wise_probability_{idx} drawed and saved in {opt.plot_store_dir_for_this_batch}!')

        fig13 = draw_intensity_integral_and_probability(df_probability, df_event, None, 'Probability', color_palette, num_events)
        save_fig(fig13, opt.plot_store_dir_for_this_batch, f'sample_time_mark_probability_{idx}.pdf')
        logger.info(f'mark_wise_integral_{idx} drawed and saved in {opt.plot_store_dir_for_this_batch}!')


    return 0