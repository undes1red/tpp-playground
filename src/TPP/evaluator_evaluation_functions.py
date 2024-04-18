import matplotlib.pyplot as plt
import seaborn as sns
import os
import lzma
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
    # Create the plot storing directory if not exist.
    plot_store_dir_for_this_batch = os.path.join(opt.store_dir, opt.plot_type, desc, str(batch_idx))
    opt.plot_store_dir_for_this_batch = plot_store_dir_for_this_batch
    if not os.path.exists(plot_store_dir_for_this_batch):
        os.makedirs(plot_store_dir_for_this_batch)

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
        plt.savefig(os.path.join(plot_store_dir_for_this_batch, plot_name + '.png'), dpi = 1000)
        fig.clear()
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
    elapsed_time = 0
    data_size = 0
    size_of_dataset = len(dataset)
    with tqdm(dataset, desc = f'Spearman and L1 for {desc}') as progress_bar:
        for minibatch in progress_bar:
            spearman_for_this_batch, l1_for_this_batch = model('spearman_and_l1', minibatch, opt)               
                                                                               # [batch_size, seq_len * resolution]
            spearman += spearman_for_this_batch
            l1 += l1_for_this_batch

        elapsed_time = progress_bar.format_dict['elapsed']
        data_size = progress_bar.format_dict['total']
    
    spearman = spearman / size_of_dataset
    l1 = l1 / size_of_dataset

    if not os.path.exists(opt.store_dir):
        os.makedirs(opt.store_dir)
    result_file = os.path.join(opt.store_dir, f'{desc}_spearman_and_l1.txt')
    f = open(result_file, 'w')
    f.write(f'For the {desc} of {opt.dataset_name}, we announce that the average spearman coefficient is {spearman} and average L1 distance is {l1}.\n Evaluation speed: {elapsed_time/data_size}s per sequence.')
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
    list_mae_e = []
    f1 = []
    list_probability_sum = []
    events_next = []
    if opt.model_name == 'ifib_c':
        probability_integral_from_zero_to_infinite = []
    elapsed_time = 0
    data_size = 0
    capable_of_sending_event_next = ['fenn', 'fullynn', 'sahp', 'thp', 'marked_lognormmix']

    with tqdm(dataset, desc = f'MAE-E and macro-f1 for {desc}') as progress_bar:
        for minibatch in progress_bar:
            if opt.model_name == 'ifib_c':
                mae_e_per_seq, f1_per_seq, probability_sum_per_seq, \
                    probability_integral_from_zero_to_infinite_per_seq, events_next_per_seq = model('mae_e_and_f1', minibatch, opt)
                                                                               # [batch_size, seq_len, num_events]
            elif opt.model_name in capable_of_sending_event_next:
                mae_e_per_seq, f1_per_seq, probability_sum_per_seq, events_next_per_seq = model('mae_e_and_f1', minibatch, opt)
                                                                               # [batch_size, seq_len]
            else:
                mae_e_per_seq, f1_per_seq, probability_sum_per_seq = model('mae_e_and_f1', minibatch, opt)
                                                                               # [batch_size, seq_len]
                events_next_per_seq = np.array([])

            list_mae_e.append(mae_e_per_seq.flatten().tolist())
            list_probability_sum.append(probability_sum_per_seq.flatten().tolist())
            events_next.append(events_next_per_seq.flatten().tolist())
            f1 += f1_per_seq
            if opt.model_name == 'ifib_c':
                probability_integral_from_zero_to_infinite.append(probability_integral_from_zero_to_infinite_per_seq.tolist())

        elapsed_time = progress_bar.format_dict['elapsed']
        data_size = progress_bar.format_dict['total']

    f1 = np.array(f1).mean()
    mae_e = np.concatenate(list_mae_e)
    probability_sum = np.concatenate(list_probability_sum)
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

    mae_e_dist_file = os.path.join(opt.store_dir, f'{desc}_mae_e_data.pkl')
    f = open(mae_e_dist_file, 'wb')
    if opt.model_name == 'ifib_c':
        pkl.dump({'mae_e': list_mae_e, 'events_next': events_next, 'pm': probability_integral_from_zero_to_infinite}, f)
    else:
        pkl.dump({'mae_e': list_mae_e, 'events_next': events_next}, f)
    f.close()


def which_event_occurs_first(model, dataset, desc, opt):
    '''
    This function is called when task_name = mae_e_and_f1.

    This function calculates the average of mae_e and macro-f1 between the model prediction based on history
    and the ground truth on all available event sequences.
    We dump all mae_e values for calculating Q1, Q2, and Q3 later.
    '''
    mae_e = None
    f1 = []
    elapsed_time = 0
    data_size = 0

    with tqdm(dataset, desc = f'Predict the next event by finding which event occurs first for {desc}') as progress_bar:
        for minibatch in progress_bar:
            mae_e_per_seq, f1_per_seq = model('which_event_first', minibatch, opt)
                                                                               # [batch_size, seq_len]
            if mae_e is None:
                mae_e = mae_e_per_seq.flatten()
            else:
                mae_e, _ = pack((mae_e, mae_e_per_seq.flatten()), '*')

            f1.append(f1_per_seq)
        elapsed_time = progress_bar.format_dict['elapsed']
        data_size = progress_bar.format_dict['total']

    f1 = np.array(f1).mean()
    mean_mae_e = mae_e.mean().item()

    if not os.path.exists(opt.store_dir):
        os.makedirs(opt.store_dir)
    
    '''
    Report the average of mae-e and f1.
    '''
    result_file = os.path.join(opt.store_dir, f'{desc}_which_event_first.txt')
    f = open(result_file, 'w')
    f.write(f'For the {desc} of {opt.dataset_name}, we announce that the average MAE-E is {mean_mae_e} and average macro-F1 is {f1}. \n Evaluation speed: {elapsed_time/data_size}s per sequence.')
    f.close()

    '''
    Dump the detailed distribution of mae-e for further usage.
    '''
    mae_e_dist_file = os.path.join(opt.store_dir, f'{desc}_which_event_first.pkl')
    f = open(mae_e_dist_file, 'wb')
    pkl.dump(mae_e, f)
    f.close()


def mae_e_and_f1_by_time_event(model, dataset, desc, opt):
    '''
    This function is called when task_name = mae_e_and_f1.

    This function calculates the average of mae_e and macro-f1 between the model prediction based on history
    and the ground truth on all available event sequences.
    We dump all mae_e values for calculating Q1, Q2, and Q3 later.
    '''
    list_mae_e = []
    f1 = []
    events_pred_index = []
    event_next = []
    elapsed_time = 0
    data_size = 0


    with tqdm(dataset, desc = f'MAE-E and macro-f1 for {desc} following time event paradigm') as progress_bar:
        for minibatch in progress_bar:
            mae_e_per_seq, f1_per_seq, events_pred_index_per_seq, events_next_per_seq = model('mae_e_and_f1_by_time_event', minibatch, opt)
                                                                               # [batch_size, seq_len]
            list_mae_e.append(mae_e_per_seq.flatten().tolist())
            event_next.append(events_next_per_seq.flatten().tolist())
            events_pred_index.append(events_pred_index_per_seq.tolist())
            f1.append(f1_per_seq)

        elapsed_time = progress_bar.format_dict['elapsed']
        data_size = progress_bar.format_dict['total']


    f1 = np.array(f1).mean()
    mean_mae_e = np.concatenate(list_mae_e).mean().item()

    if not os.path.exists(opt.store_dir):
        os.makedirs(opt.store_dir)
    
    '''
    Report the average of mae-e and f1.
    '''
    result_file = os.path.join(opt.store_dir, f'{desc}_mae_e_and_macro-f1_by_time_event.txt')
    f = open(result_file, 'w')
    f.write(f'For the {desc} of {opt.dataset_name}, we announce that the average MAE-E is {mean_mae_e} and average macro-F1 is {f1}. Evaluation speed: {elapsed_time/data_size}s per sequence.')
    f.close()

    '''
    Dump the detailed distribution of mae-e for further usage.
    '''
    mae_e_dist_file = os.path.join(opt.store_dir, f'{desc}_mae_e_by_time_event.pkl')
    f = open(mae_e_dist_file, 'wb')
    pkl.dump({'mae_e': list_mae_e, 'event_next': event_next, 'f1': f1, 'events_pred_index': events_pred_index}, f)
    f.close()


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

    if not os.path.exists(opt.store_dir):
        os.makedirs(opt.store_dir)
    
    '''
    Report the average of mae-e and f1.
    '''
    result_file = os.path.join(opt.store_dir, f'{desc}_mae_e_and_macro-f1_of_imputated_events.txt')
    f = open(result_file, 'w')
    f.write(f'For the {desc} of {opt.dataset_name}, we announce that the average MAE is {mean_mae} and average macro-F1 is {f1}. Evaluation speed: {elapsed_time/data_size}s per sequence.')
    f.close()

    '''
    Dump the detailed distribution of mae-e for further usage.
    '''
    mae_e_dist_file = os.path.join(opt.store_dir, f'{desc}_mae_e_of_imputated_events.pkl')
    f = open(mae_e_dist_file, 'wb')
    pkl.dump({'mae_e': list_mae, 'f1': f1}, f)
    f.close()


def samples_from_et(model, dataset, desc, opt):
    '''
    This function is called when task_name = mae_e_and_f1.

    This function calculates the average of mae_e and macro-f1 between the model prediction based on history
    and the ground truth on all available event sequences.
    We dump all mae_e values for calculating Q1, Q2, and Q3 later.
    '''
    samples = []
    p_ms = []

    with tqdm(dataset, desc = f'Samples of {desc} from ET') as progress_bar:
        for minibatch in progress_bar:
            samples_per_seq, p_ms_per_seq = model('samples_from_et', minibatch, opt)
                                                                               # [batch_size, seq_len]
            samples.append(samples_per_seq.tolist())
            p_ms.append(p_ms_per_seq.tolist())
    
    if not os.path.exists(opt.store_dir):
        os.makedirs(opt.store_dir)

    '''
    Dump the detailed distribution of mae-e for further usage.
    '''
    mae_e_dist_file = os.path.join(opt.store_dir, f'{desc}_samples_for_every_point.pkl.lzma')
    f = lzma.open(mae_e_dist_file, 'wb')
    pkl.dump({'samples': samples, 'p_ms': p_ms}, f)
    f.close()