import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import gc

from torch.utils.flop_counter import FlopCounterMode
from tqdm import tqdm
from src.taskhost_utils import get_logger, mkdir_if_not_exist, dump_to_pkl, write_to_txt
from src.TPP.resources.evaluation_functions_utils import flatten, free_model_from_gpu

logger = get_logger(name = __file__)


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
    # Create the plot storing directory if not exist.
    plot_store_dir_for_this_batch = os.path.join(opt.store_dir, opt.plot_type, desc, str(batch_idx))
    opt.plot_store_dir_for_this_batch = plot_store_dir_for_this_batch
    mkdir_if_not_exist(plot_store_dir_for_this_batch)

    plots = model('graph', minibatch, opt)
    
    plt.rcParams.update({'font.size': 22, 'figure.figsize': (9, 7)})
    for plot_name, plot_instructions in plots.items():
        fig = plt.figure()
        ax = None
        for instruction in plot_instructions:
            if instruction.get('plot_type') == 'text':
                ax.text(**instruction['kwargs'])
            else:
                if instruction.get('length') and instruction.get('height'):
                    fig.set_size_inches(instruction.get('length'), instruction.get('height'))
                ax = getattr(sns, instruction['plot_type'])(ax = ax, **instruction['kwargs'])
        
        logger.info(f'{plot_name} for No.{batch_idx} minibatch in {desc} dataset finished drawing!')
        plt.savefig(os.path.join(plot_store_dir_for_this_batch, plot_name + '.png'), dpi = 1000, bbox_inches = "tight")
        fig.clear()
        plt.close(fig = fig)
        del ax
        gc.collect()


def spearman_and_l1_postprocess(all_evaluation_results, desc, opt):
    '''
    This function is called when task_name = spearman_and_l1.

    This function calculates the average of spearman and L^1 distance between the learned probability distribution
    and the ground truth on all synthetic event sequences.
    '''
    spearman, l1 = all_evaluation_results
    spearman = np.mean(spearman)
    l1 = np.mean(l1)

    result_file = os.path.join(opt.store_dir, f'{desc}_spearman_and_l1.txt')
    strings = f'For the {desc} of {opt.dataset_name}, we announce that the average spearman coefficient is {spearman} and average L1 distance is {l1}.'
    write_to_txt(strings, result_file)


def mae_and_f1_postprocess(all_evaluation_results, desc, opt):
    '''
    This function is called when task_name = mae_and_f1.

    This function calculates the average of mae and macro-f1 between the model prediction based on history
    and the ground truth on all available event sequences.
    We dump all mae values for calculating Q1, Q2, and Q3 later.
    '''
    mae, f1 = all_evaluation_results
    f1 = np.mean(f1)
    mean_mae = np.mean(flatten(mae))

    result_file = os.path.join(opt.store_dir, f'{desc}_mae_and_macro-f1.txt')
    strings = f'For the {desc} of {opt.dataset_name}, we announce that the average MAE is {mean_mae} and average macro-F1 is {f1}.'
    write_to_txt(strings, result_file)

    '''
    Dump the detailed distribution of mae for further usage.
    '''
    mae_dist_file = os.path.join(opt.store_dir, f'{desc}_mae.pkl')
    dump_to_pkl(mae, mae_dist_file, compression = 'bz2')


def mae_e_and_f1_postprocess(all_evaluation_results, desc, opt):
    '''
    This function is called when task_name = mae_e_and_f1.

    This function calculates the average of mae_e and macro-f1 between the model prediction based on history
    and the ground truth on all available event sequences.
    We dump all mae_e values for calculating Q1, Q2, and Q3 later.
    '''
    capable_of_sending_event_next = ['fenn', 'fullynn', 'sahp', 'thp', 'marked_lognormmix']
    if opt.model_name == 'ifib_c':
        '''
        mae_e, macro-f1, sum of p^*(m), p^*(m), events_next
        '''
        mae_e, f1, sum_of_pm, pm, event_next = all_evaluation_results

        mae_e_dist_file = os.path.join(opt.store_dir, f'{desc}_mae_e_data.pkl')
        data = {'mae_e': mae_e, 'events_next': event_next, 'pm': pm}
        dump_to_pkl(data, mae_e_dist_file, compression = 'bz2')
    elif opt.model_name in capable_of_sending_event_next:
        '''
        mae_e, macro-f1, sum of p^*(m), events_next
        '''
        mae_e, f1, sum_of_pm, event_next = all_evaluation_results

        mae_e_dist_file = os.path.join(opt.store_dir, f'{desc}_mae_e_data.pkl')
        data = {'mae_e': mae_e, 'events_next': event_next}
        dump_to_pkl(data, mae_e_dist_file, compression = 'bz2')
    else:
        '''
        mae_e, macro-f1, sum of p^*(m)
        '''
        mae_e, f1, sum_of_pm = all_evaluation_results

        mae_e_dist_file = os.path.join(opt.store_dir, f'{desc}_mae_e_data.pkl')
        data = {'mae_e': mae_e}
        dump_to_pkl(data, mae_e_dist_file, compression = 'bz2')

    mean_mae_e = np.mean(flatten(mae_e))
    f1 = np.mean(f1)
    mean_probability_sum = np.mean(sum_of_pm)

    '''
    Report the average of mae-e and f1.
    '''
    result_file = os.path.join(opt.store_dir, f'{desc}_mae_e_and_macro-f1.txt')
    strings = f'For the {desc} of {opt.dataset_name}, we announce that the average MAE-E is {mean_mae_e} and average macro-F1 is {f1}. The sum of p(t) is {mean_probability_sum}.'
    write_to_txt(strings, result_file)


def mae_e_and_f1_by_time_event_postprocess(all_evaluation_results, desc, opt):
    mae_e, f1, events_pred_index, events_next = all_evaluation_results

    f1 = np.mean(f1)
    mean_mae_e = np.mean(flatten(mae_e))

    '''
    Report the average of mae-e and f1.
    '''
    result_file = os.path.join(opt.store_dir, f'{desc}_mae_e_and_macro-f1_by_time_event.txt')
    strings = f'For the {desc} of {opt.dataset_name}, we announce that the average MAE-E is {mean_mae_e} and average macro-F1 is {f1}'
    write_to_txt(strings, result_file)

    '''
    Dump the detailed distribution of mae-e for further usage.
    '''
    mae_e_dist_file = os.path.join(opt.store_dir, f'{desc}_mae_e_by_time_event.pkl')
    data = {'mae_e': mae_e, 'f1': f1, 'events_pred_index': events_pred_index, 'event_next': events_next}
    dump_to_pkl(data, mae_e_dist_file, compression = 'bz2')


def which_event_occurs_first_postprocess(all_evaluation_results, desc, opt):
    '''
    This function is called when task_name = which_event_occurs_first.
    '''
    mae, f1 = all_evaluation_results
    f1 = np.mean(f1)
    mean_mae = np.mean(flatten(mae))

    '''
    Report the average of mae-e and f1.
    '''
    result_file = os.path.join(opt.store_dir, f'{desc}_which_event_first.txt')
    strings = f'For the {desc} of {opt.dataset_name}, we announce that the average MAE-E is {mean_mae} and average macro-F1 is {f1}.'
    write_to_txt(strings, result_file)

    '''
    Dump the detailed distribution of mae-e for further usage.
    '''
    mae_e_dist_file = os.path.join(opt.store_dir, f'{desc}_which_event_first.pkl')
    dump_to_pkl(mae, mae_e_dist_file, compression = 'bz2')


def samples_from_et_postprocess(all_evaluation_results, desc, opt):
    '''
    Dump the detailed distribution of mae-e for further usage.
    '''
    samples, p_ms = all_evaluation_results

    mae_e_dist_file = os.path.join(opt.store_dir, f'{desc}_samples_for_every_point.pkl')
    data = {'samples': samples, 'p_ms': p_ms}
    dump_to_pkl(data, mae_e_dist_file, compression = 'bz2')


desc_funcs = {
    'spearman_and_l1': ['Spearman and L1 for {0}', spearman_and_l1_postprocess],
    'mae_and_f1': ['MAE and macro-f1 for {0}', mae_and_f1_postprocess],
    'mae_e_and_f1': ['MAE-E and macro-f1 for {0}', mae_e_and_f1_postprocess],
    'mae_e_and_f1_by_time_event': ['MAE-E and macro-f1 for {0} following NER', mae_e_and_f1_by_time_event_postprocess],
    'which_event_occurs_first': ['Predict the next event by finding which event occurs first for {0}', which_event_occurs_first_postprocess],
    'samples_from_et': [f'Samples of {0} for each mark', ]
}


def basic_evaluation_loop(model, dataset, desc, opt, early_offload = True):
    task_name = opt.task_name
    desc_string, postprocess_func = desc_funcs[task_name]

    elapsed_time = 0
    list_output_results = None

    with tqdm(dataset, desc = desc_string.format(desc)) as progress_bar:
        with FlopCounterMode(display = False) as counter:
            for minibatch in progress_bar:
                results_per_minibatch = model(task_name, minibatch, opt)
                
                if list_output_results is None:
                    result_length = len(results_per_minibatch)
                    list_output_results = [[] for _ in range(result_length)]
                
                [a.append(b) for a, b in zip(list_output_results, results_per_minibatch)]

        flops = sum(counter.flop_counts['Global'].values())
        elapsed_time = progress_bar.format_dict['elapsed']
        data_size = progress_bar.format_dict['total']
    
    if early_offload:
        # How to remove a model and free its memory?
        free_model_from_gpu(model)

    mkdir_if_not_exist(opt.store_dir)
    result_file = os.path.join(opt.store_dir, f'{desc}_{task_name}_misc.txt')
    strings = [f'Evaluation speed: {elapsed_time/data_size}s per sequence.\n', 
               f'Computation: {flops / 1000**4} TFlops.']
    write_to_txt(strings, result_file)

    # call user's postprocess function for evaluation results.
    postprocess_func(list_output_results, desc, opt)


def mae_and_f1_of_imputated_events(model, dataset, desc, opt):
    '''
    This function is called when task_name = mae_e_and_f1.

    This function calculates the average of mae_e and macro-f1 between the model prediction based on history
    and the ground truth on all available event sequences.
    We dump all mae_e values for calculating Q1, Q2, and Q3 later.
    '''
    elapsed_time = 0
    data_size = 0
    list_mae = []
    f1 = []

    with tqdm(dataset, desc = f'MAE and macro-f1 for imputated events in {desc}:') as progress_bar:
        for minibatch in progress_bar:
            mae_per_seq, f1_per_seq = model('mae_and_f1_imputated_events', minibatch, opt)
                                                                               # [batch_size, seq_len]
            list_mae.append(mae_per_seq.flatten().tolist())
            f1.append(f1_per_seq)

        elapsed_time = progress_bar.format_dict['elapsed']
        data_size = progress_bar.format_dict['total']

    f1 = np.array(f1).mean()
    mean_mae = np.concatenate(list_mae).mean().item()

    mkdir_if_not_exist(opt.store_dir)
    '''
    Report the average of mae-e and f1.
    '''
    result_file = os.path.join(opt.store_dir, f'{desc}_mae_e_and_macro-f1_of_imputated_events.txt')
    strings = [f'For the {desc} of {opt.dataset_name}, we announce that the average MAE is {mean_mae} and average macro-F1 is {f1}.\n', 
               f'Evaluation speed: {elapsed_time/data_size}s per sequence.']
    write_to_txt(strings, result_file)

    '''
    Dump the detailed distribution of mae-e for further usage.
    '''
    mae_e_dist_file = os.path.join(opt.store_dir, f'{desc}_mae_e_of_imputated_events.pkl')
    data = {'mae_e': list_mae, 'f1': f1}
    dump_to_pkl(data, mae_e_dist_file, compression = 'bz2')