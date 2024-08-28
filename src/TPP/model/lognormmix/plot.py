import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.toolbox.misc import move_from_tensor_to_ndarray, stable_palette, save_fig, get_logger
from src.toolbox.metrics import L1_distance_between_two_funcs

from src.TPP.model.utils import draw_intensity_integral_and_probability, draw_lineplot
from src.TPP.resources.syn_tpp_utils import expand_true_intensity, expand_true_probability

logger = get_logger(__name__)


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
    '''

    '''
    Only MAE is available.
    '''
    mae = data['mae_before_event']                                     # [batch_size, seq_len]
    mask_next = data['mask_next']                                      # [batch_size, seq_len]

    packed_data = zip(*move_from_tensor_to_ndarray(mae, mask_next))

    for idx, (mae_per_seq, mask_next_per_seq) in enumerate(packed_data):
        seq_len = mask_next_per_seq.sum()

        data_maes_per_seq = {
             'Event Index': list(range(seq_len)),
            r'$\log(1 + \mathrm{MAE})$': np.log(1 + mae_per_seq[:seq_len]),
             'marks': ['MAE'] * seq_len
        }

        fig1 = draw_lineplot(data = data_maes_per_seq, x = 'Event Index', y = r'$\log(1 + \mathrm{MAE})$', hue = 'Mark', \
                             figure_kwargs = {'font.size': 18, 'figure.figsize': (5, 5)})
        save_fig(fig1, opt.plot_store_dir_for_this_batch, f'MAE_{idx}.pdf')
        logger.info(f'MAE_{idx} drawed and saved in {opt.plot_store_dir_for_this_batch}!')

    return 0