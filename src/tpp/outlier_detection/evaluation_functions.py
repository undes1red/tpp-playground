import os
import numpy as np
from tqdm import tqdm
from src.toolbox.misc import get_logger, mkdir_if_not_exist, dump_to_pkl, write_to_txt, flatten


logger = get_logger(name = __file__)


def omission_outlier_postprocess(all_evaluation_results, desc, opt):
    '''
    This function is called when task_name = spearman_and_l1.

    This function calculates the average of spearman and L^1 distance between the learned probability distribution
    and the ground truth on all synthetic event sequences.
    '''
    all_aurocs = all_evaluation_results
    auroc = np.mean(all_aurocs)

    result_file = os.path.join(opt.store_dir, f'{desc}_auroc.txt')
    strings = f'For the {desc} of {opt.dataset_name}, we announce that the average auroc is {auroc}.'
    write_to_txt(strings, result_file)
    
    auroc_file = os.path.join(opt.store_dir, f'{desc}_auroc.pkl')
    data = {'auroc': all_aurocs}
    dump_to_pkl(data, auroc_file, compression = 'bz2')


def mae_and_f1_of_imputated_events(model, dataset, desc, opt, early_offload):
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


desc_funcs = {
    'omission_outlier': {'desc_string': 'AUROC of omission on {0}', 'postprocess_func': omission_outlier_postprocess},

    # Custom evaluation function.
    'mae_and_f1_of_imputated_events': mae_and_f1_of_imputated_events
}