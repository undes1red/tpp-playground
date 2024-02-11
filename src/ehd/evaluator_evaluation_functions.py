import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pickle as pkl
import gc

from einops import pack
from tqdm import tqdm
from src.taskhost_utils import getLogger


logger = getLogger(name = __file__)


def draw(model, minibatch, desc, batch_idx, opt):
    '''
    This function will be called when task_name = graph

    In the new pipeline, each plot is defined as a instruction list. draw_features() should extract and
    call correct seaborn APIs with expected kwargs. The structure of the dict goes as follows:
    {
        ...
        '[plot name]':
        [
            ...
            {
                'plot_type': '[plot_type]'
                'length': [diagram length],
                'height': [diagram height],
                'kwargs':
                {
                    ...'[arguments sent to seaborn APIs.]'
                }
            }
            ...
        ]
        ...
    }
    '''

    plots = model('graph', minibatch, opt)
    
    # Create the plot storing directory if not exist.
    plot_store_dir_for_this_batch = os.path.join(opt.store_dir, opt.plot_type, desc, str(batch_idx))
    if not os.path.exists(plot_store_dir_for_this_batch):
        os.makedirs(plot_store_dir_for_this_batch)
    
    plt.rcParams.update({'font.size': 22, 'figure.figsize': (9, 7)})
    for plot_name, plot_instructions in plots.items():
        fig = plt.figure()
        ax = None
        for instruction in plot_instructions:
            if instruction.get('plot_type') == 'text':
                ax.text(transform = ax.transAxes if ax is not None else None, **instruction['kwargs'])
            else:
                if instruction.get('length') and instruction.get('height'):
                    fig.set_size_inches(instruction.get('length'), instruction.get('height'))
                ax = getattr(sns, instruction['plot_type'])(ax = ax, **instruction['kwargs'])
        
        logger.info(f'{plot_name} for No.{batch_idx} minibatch in {desc} dataset finished drawing!')
        plt.savefig(os.path.join(plot_store_dir_for_this_batch, plot_name + '.png'), dpi = 1000)
        fig.clf()
        plt.close(fig = fig)
        del ax
        gc.collect()


def spearman_and_l1(model, dataset, desc, opt):
    '''
    This function is called when task_name = spearman_and_l1.

    This function calculates the average of spearman and L^1 distance between the learned probability distribution
    and the ground truth on all synthetic event sequences.
    '''
    spearman = 0
    l1 = 0
    size_of_dataset = len(dataset)
    for minibatch in tqdm(dataset, desc = f'Spearman and L1 for {desc}'):
        spearman_for_this_batch, l1_for_this_batch = model('spearman_and_l1', minibatch, opt)               
                                                                               # [batch_size, seq_len * resolution]
        spearman += spearman_for_this_batch
        l1 += l1_for_this_batch
    
    spearman = spearman / size_of_dataset
    l1 = l1 / size_of_dataset

    if not os.path.exists(opt.store_dir):
        os.makedirs(opt.store_dir)
    result_file = os.path.join(opt.store_dir, f'{desc}_spearman_and_l1.txt')
    f = open(result_file, 'w')
    f.write(f'For the {desc} of {opt.dataset_name}, we announce that the average spearman coefficient is {spearman} and average L1 distance is {l1}.')
    f.close()


def mae_and_f1(model, dataset, desc, opt):
    '''
    This function is called when task_name = mae_and_f1.

    This function calculates the average of mae and macro-f1 between the model prediction based on history
    and the ground truth on all available event sequences.
    We dump all mae values for calculating Q1, Q2, and Q3 later.
    '''
    mae = None
    f1 = 0
    elapsed_time = 0
    data_size = 0
    size_of_dataset = len(dataset)
    
    with tqdm(dataset, desc = f'MAE and macro-f1 for {desc}') as progress_bar:
        for minibatch in progress_bar:
            mae_per_seq, f1_per_seq = model('mae_and_f1', minibatch, opt)
                                                                               # [batch_size, seq_len]
            if mae is None:
                mae = mae_per_seq.flatten()
            else:
                mae, mae_ps = pack((mae, mae_per_seq.flatten()), '*')
            f1 += f1_per_seq
        elapsed_time = progress_bar.format_dict['elapsed']
        data_size = progress_bar.format_dict['total']

    f1 = f1 / size_of_dataset
    mean_mae = mae.mean().item()

    if not os.path.exists(opt.store_dir):
        os.makedirs(opt.store_dir)

    '''
    Report the average of mae and f1.
    '''
    result_file = os.path.join(opt.store_dir, f'{desc}_mae_and_macro-f1.txt')
    f = open(result_file, 'w')
    f.write(f'For the {desc} of {opt.dataset_name}, we announce that the average MAE is {mean_mae} and average macro-F1 is {f1}.\n Evaluation speed: {elapsed_time/data_size}s per sequence.')
    f.close()

    '''
    Dump the detailed distribution of mae for further usage.
    '''
    mae_dist_file = os.path.join(opt.store_dir, f'{desc}_mae.pkl')
    f = open(mae_dist_file, 'wb')
    pkl.dump(mae, f)
    f.close()


def mae_e_and_f1(model, dataset, desc, opt):
    '''
    This function is called when task_name = mae_e_and_f1.

    This function calculates the average of mae_e and macro-f1 between the model prediction based on history
    and the ground truth on all available event sequences.
    We dump all mae_e values for calculating Q1, Q2, and Q3 later.
    '''
    mae_e = None
    f1 = []
    probability_sum = None
    elapsed_time = 0
    data_size = 0

    with tqdm(dataset, desc = f'MAE-E and macro-f1 for {desc}') as progress_bar:
        for minibatch in progress_bar:
            mae_e_per_seq, f1_per_seq, probability_sum_per_seq = model('mae_e_and_f1', minibatch, opt)
                                                                               # [batch_size, seq_len]
            if mae_e is None:
                mae_e = mae_e_per_seq.flatten()
            else:
                mae_e, mae_e_ps = pack((mae_e, mae_e_per_seq.flatten()), '*')

            if probability_sum is None:
                probability_sum = probability_sum_per_seq.flatten()
            else:
                probability_sum, probability_sum_ps = pack((probability_sum, probability_sum_per_seq.flatten()), '*')

            f1 += f1_per_seq
        elapsed_time = progress_bar.format_dict['elapsed']
        data_size = progress_bar.format_dict['total']

    f1 = np.array(f1).mean()
    mean_mae_e = mae_e.mean().item()
    mean_probability_sum = probability_sum.mean().item()

    if not os.path.exists(opt.store_dir):
        os.makedirs(opt.store_dir)
    
    '''
    Report the average of mae-e and f1.
    '''
    result_file = os.path.join(opt.store_dir, f'{desc}_mae_e_and_macro-f1.txt')
    f = open(result_file, 'w')
    f.write(f'For the {desc} of {opt.dataset_name}, we announce that the average MAE-E is {mean_mae_e} and average macro-F1 is {f1}. The sum of p(t) is {mean_probability_sum}. \n Evaluation speed: {elapsed_time/data_size}s per sequence.')
    f.close()

    '''
    Dump the detailed distribution of mae-e for further usage.
    '''
    mae_e_dist_file = os.path.join(opt.store_dir, f'{desc}_mae_e.pkl')
    f = open(mae_e_dist_file, 'wb')
    pkl.dump(mae_e, f)
    f.close()


def lsp_and_lrp(model, dataset, desc, opt):
    '''
    This function is called when task_name = lsp_and_lrp.
    '''
    metric_list = None

    elapsed_time = 0
    data_size = 0

    with tqdm(dataset, desc = f'lsp and lrp for {desc}') as progress_bar:
        for minibatch in progress_bar:
            '''
            percentage_remained_events_per_seq, random_percentage_remained_events_per_seq, greedy_percentage_remained_events_per_seq, \
            l_sp_per_seq, l_sp_random_per_seq, l_sp_g1_per_seq, \
            l_rp_per_seq, l_rp_random_per_seq, l_rp_g1_per_seq, \
            time_baseline_1_given_percentage_to_ehd_per_seq, time_baseline_1_to_ehd_per_seq, time_baseline_2_to_ehd_per_seq
                = model('lsp_and_lrp', minibatch, opt)
            '''
            metrics_per_seq = model('lsp_and_lrp', minibatch, opt)
            if metric_list is None:
                metric_list = [[] for _ in range(len(metrics_per_seq))]
            
            for metric_value_per_seq, metric_values in zip(metrics_per_seq, metric_list):
                metric_values.append(metric_value_per_seq)

        elapsed_time = progress_bar.format_dict['elapsed']
        data_size = progress_bar.format_dict['total']

    metric_list = np.array(metric_list)

    the_mean_of_metric = metric_list.mean(axis = -1).tolist()

    if not os.path.exists(opt.store_dir):
        os.makedirs(opt.store_dir)

    # Metric Translator
    metric_name = [
        'percentage_remained_events', 'random_percentage_remained_events', 'greedy_percentage_remained_events',
        'l_sp', 'l_sp_random', 'l_sp_g1', 'selected_L_sp_given_events', 'l_rp', 'l_rp_random', 'l_rp_g1', 'selected_L_rp_given_events', 
        'time_baseline_1_given_percentage_to_ehd_per_seq', 'time_baseline_1_to_ehd_per_seq', 'time_baseline_2_to_ehd_per_seq',
        'time_greedy_given_percentage_to_ehd'
        ]
    assert len(metric_name) == len(the_mean_of_metric)
    dict_metric_name = {name: value for name, value in zip(metric_name, the_mean_of_metric)}
    
    # Report the average of mae-e and f1.
    result_file = os.path.join(opt.store_dir, f'{desc}_lsp_and_lrp.txt')
    f = open(result_file, 'w')
    f.write(f"For the {desc} of {opt.dataset_name}, we announce that the average percentage of remained events is {dict_metric_name['percentage_remained_events']}.\n")
    f.write(f"For random selection, the average percentage of remained events is {dict_metric_name['random_percentage_remained_events']}.\n")
    f.write(f"For greedy, the average percentage of remained events is {dict_metric_name['greedy_percentage_remained_events']}.\n")
    f.write(f"The average ratio between probability of selected events and originals is {dict_metric_name['l_sp']}, while random removal achieved {dict_metric_name['l_sp_random']}, greedy achieved {dict_metric_name['selected_L_sp_given_events']}.\n")
    f.write(f"The average ratio between probability of remained events and originals is {dict_metric_name['l_rp']}, while random removal achieved {dict_metric_name['l_rp_random']}, greedy achieved {dict_metric_name['selected_L_rp_given_events']}.\n")
    f.write(f"The average ratio between the running time of EHD and random selection with given K is {dict_metric_name['time_baseline_1_given_percentage_to_ehd_per_seq']}.\n")
    f.write(f"The average ratio between the running time of EHD and greedy selection with given K is {dict_metric_name['time_greedy_given_percentage_to_ehd']}.\n")
    f.write(f"The average ratio between the running time of EHD and random selection with given L_sp and L_rp is {dict_metric_name['time_baseline_1_to_ehd_per_seq']}.\n")
    f.write(f"The average ratio between the running time of EHD and greedy selection with given L_sp and L_rp is {dict_metric_name['time_baseline_2_to_ehd_per_seq']}.\n") 
    f.write(f"Evaluation speed: {elapsed_time/data_size}s per sequence.") 
    f.close()

    # Dump the detailed distribution of mae-e for further usage.
    data = {metric_name: metric_values for (metric_name, metric_values) in zip(metric_name, metric_list.tolist())}

    target_file = os.path.join(opt.store_dir, f'{desc}_data.pkl')
    f = open(target_file, 'wb')
    pkl.dump(data, f)
    f.close()


def lsp_and_lrp_fast(model, dataset, desc, opt):
    '''
    This function is called when task_name = lsp_and_lrp.
    '''
    output_list = None

    elapsed_time = 0
    data_size = 0

    with tqdm(dataset, desc = f'lsp and lrp fast for {desc}') as progress_bar:
        for minibatch in progress_bar:
            '''
            percentage_remained_events, L_sp, L_sp_r, L_rp, L_rp_r, time_baseline_1_given_percentage_to_ehd, \
            history_mask, time_history, time_future, events_history, events_future \
                = model('lsp_and_lrp', minibatch, opt, fast = True)
            '''
            metrics_per_seq = model('lsp_and_lrp', minibatch, opt, fast = True)
            if output_list is None:
                output_list = [[] for _ in range(len(metrics_per_seq))]
            
            for metric_value_per_seq, metric_values in zip(metrics_per_seq, output_list):
                metric_values.append(metric_value_per_seq)

        elapsed_time = progress_bar.format_dict['elapsed']
        data_size = progress_bar.format_dict['total']

    metric_list = np.array(output_list[:-5])

    the_mean_of_metric = metric_list.mean(axis = -1).tolist()

    if not os.path.exists(opt.store_dir):
        os.makedirs(opt.store_dir)

    # Metric Translator
    output_name = [
        'percentage_remained_events', 'L_sp', 'l_sp_random', 'L_rp',
        'l_rp_random', 'time_baseline_1_given_percentage_to_ehd', 
        'history_mask', 'time_history', 'time_future', 'events_history', 'events_future'
        ]
    metric_name = [
        'percentage_remained_events', 'L_sp', 'l_sp_random', 'L_rp',
        'l_rp_random', 'time_baseline_1_given_percentage_to_ehd', 
        # 'history_mask', 'time_history', 'time_future', 'events_history', 'events_future'
        ]
    assert len(metric_name) == len(the_mean_of_metric)
    dict_metric_name = {name: value for name, value in zip(metric_name, the_mean_of_metric)}
    
    # Report the average of mae-e and f1.
    result_file = os.path.join(opt.store_dir, f'{desc}_lsp_and_lrp_fast.txt')
    f = open(result_file, 'w')
    f.write(f"For the {desc} of {opt.dataset_name}, we announce that the average percentage of remained events is {dict_metric_name['percentage_remained_events']}.\n")
    f.write(f"The average ratio between probability of selected events and originals is {dict_metric_name['L_sp']}, while random removal achieved {dict_metric_name['l_sp_random']}.\n")
    f.write(f"The average ratio between probability of remained events and originals is {dict_metric_name['L_rp']}, while random removal achieved {dict_metric_name['l_rp_random']}.\n")
    f.write(f"The average ratio between the running time of EHD and random selection with given K is {dict_metric_name['time_baseline_1_given_percentage_to_ehd']}.\n")
    f.write(f"Evaluation speed: {elapsed_time/data_size}s per sequence.") 
    f.close()

    # Dump the detailed distribution of mae-e for further usage.
    data = {metric_name: metric_values for (metric_name, metric_values) in zip(output_name, output_list)}

    target_file = os.path.join(opt.store_dir, f'{desc}_data_fast.pkl')
    f = open(target_file, 'wb')
    pkl.dump(data, f)
    f.close()


def lsp_and_lrp_trend(model, dataset, desc, opt):
    '''
    This function is called when task_name = lsp_and_lrp.
    '''
    L_rp_rs_ratios, L_sp_rs_ratios = [], []

    with tqdm(dataset, desc = f'lsp and lrp trend for {desc}') as progress_bar:
        for minibatch in progress_bar:
            '''
            percentage_remained_events, L_sp, L_sp_r, L_rp, L_rp_r, time_baseline_1_given_percentage_to_ehd, \
            history_mask, time_history, time_future, events_history, events_future \
                = model('lsp_and_lrp', minibatch, opt, fast = True)
            '''
            L_rp_rs_ratio_per_seq, L_sp_rs_ratio_per_seq = model('lsp_and_lrp_trend', minibatch, opt)
            L_rp_rs_ratios.append(L_rp_rs_ratio_per_seq)
            L_sp_rs_ratios.append(L_sp_rs_ratio_per_seq)

    L_rp_rs_ratios = np.stack(L_rp_rs_ratios, axis = 0)
    L_sp_rs_ratios = np.stack(L_sp_rs_ratios, axis = 0)

    if not os.path.exists(opt.store_dir):
        os.makedirs(opt.store_dir)

    target_file = os.path.join(opt.store_dir, f'{desc}_dppl_l_distribution.pkl')
    f = open(target_file, 'wb')
    pkl.dump(L_rp_rs_ratios, f)
    f.close()

    target_file = os.path.join(opt.store_dir, f'{desc}_dppl_d_distribution.pkl')
    f = open(target_file, 'wb')
    pkl.dump(L_sp_rs_ratios, f)
    f.close()