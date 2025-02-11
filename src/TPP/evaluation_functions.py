import os
import numpy as np
from tqdm import tqdm
from src.toolbox.misc import get_logger, mkdir_if_not_exist, dump_to_pkl, write_to_txt, flatten, write_yaml


logger = get_logger(name = __file__)


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

    result_dist_file = os.path.join(opt.store_dir, f'{desc}_spearman_and_l1_result.pkl')
    dump_to_pkl({'spearman': spearman, 'l1': l1}, result_dist_file, compression = 'bz2')


def mae_and_f1_postprocess(all_evaluation_results, desc, opt):
    '''
    This function is called when task_name = mae_and_f1.

    This function calculates the average of mae and macro-f1 between the model prediction based on history
    and the ground truth on all available event sequences.
    We dump all mae values for calculating Q1, Q2, and Q3 later.
    '''
    mae, f1, events_next = all_evaluation_results
    f1 = np.mean(f1)
    mean_mae = np.mean(flatten(mae))
    
    mae_dist_file = os.path.join(opt.store_dir, f'{desc}_mae_data.pkl')
    data = {'mae': mae, 'events_next': events_next}
    dump_to_pkl(data, mae_dist_file, compression = 'bz2')

    result_file = os.path.join(opt.store_dir, f'{desc}_mae_and_macro-f1.txt')
    strings = f'For the {desc} of {opt.dataset_name}, we announce that the average MAE is {mean_mae} and average macro-F1 is {f1}.'
    write_to_txt(strings, result_file)
    result_dist_file = os.path.join(opt.store_dir, f'{desc}_mae_and_f1_result.pkl')
    dump_to_pkl({'mae': mean_mae, 'f1': f1}, result_dist_file, compression = 'bz2')

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
    capable_of_sending_event_next = ['ctlstm', 'ifib_c', 'fenn', 'fullynn', 'sahp', 'thp', 'marked_lognormmix', 'llmtpp']
    if opt.model_name in capable_of_sending_event_next:
        '''
        mae_e, macro-f1, sum of p^*(m), p^*(m), events_next
        '''
        mae_e, f1, sum_of_pm, pm, event_next = all_evaluation_results

        mae_e_dist_file = os.path.join(opt.store_dir, f'{desc}_mae_e_data.pkl')
        data = {'mae_e': mae_e, 'events_next': event_next, 'pm': pm}
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
    mean_probability_sum = np.mean(flatten(sum_of_pm))

    '''
    Report the average of mae-e and f1.
    '''
    result_file = os.path.join(opt.store_dir, f'{desc}_mae_e_and_macro-f1.txt')
    strings = f'For the {desc} of {opt.dataset_name}, we announce that the average MAE-E is {mean_mae_e} and average macro-F1 is {f1}. The sum of p(m) is {mean_probability_sum}.'
    write_to_txt(strings, result_file)
    result_dist_file = os.path.join(opt.store_dir, f'{desc}_mae_e_and_f1_result.pkl')
    dump_to_pkl({'mae_e': mean_mae_e, 'f1': f1}, result_dist_file, compression = 'bz2')


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
    result_dist_file = os.path.join(opt.store_dir, f'{desc}_mae_e_and_f1_by_time_event_result.pkl')
    dump_to_pkl({'mae_e': mean_mae_e, 'f1': f1}, result_dist_file, compression = 'bz2')

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
    result_dist_file = os.path.join(opt.store_dir, f'{desc}_which_event_occurs_first_result.pkl')
    dump_to_pkl({'mae': mean_mae, 'f1': f1}, result_dist_file, compression = 'bz2')

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


def cppod_evaluation_postprocess(all_evaluation_results, desc, opt):
    rocs = all_evaluation_results
    rocs = np.nanmean(rocs).item()
    
    result_file = os.path.join(opt.store_dir, f'{desc}_roc_mean.txt')
    strings = f'For the {desc} of {opt.dataset_name}, we announce that the average roc of outlier detection is {rocs}.'
    write_to_txt(strings, result_file)
    
    mae_e_dist_file = os.path.join(opt.store_dir, f'{desc}_cppod_rocs.pkl')
    data = {'rocs': rocs}
    dump_to_pkl(data, mae_e_dist_file, compression = 'bz2')


def cppod_commission_evaluation_postprocess(all_evaluation_results, desc, opt):
    rocs = all_evaluation_results
    rocs = np.nanmean(rocs).item()

    result_file = os.path.join(opt.store_dir, f'{desc}_roc_commission_mean.txt')
    strings = f'For the {desc} of {opt.dataset_name}, we announce that the average roc of outlier detection is {rocs}.'
    write_to_txt(strings, result_file)
    
    mae_e_dist_file = os.path.join(opt.store_dir, f'{desc}_cppod_commission_rocs.pkl')
    data = {'rocs': rocs}
    dump_to_pkl(data, mae_e_dist_file, compression = 'bz2')


def generate_hypro_dataset_postprocess(all_evaluation_results, desc, opt):
    '''
    Dump the detailed distribution of mae-e for further usage.
    '''
    input_time, input_events, tau_sampled, events_sampled = all_evaluation_results

    mae_e_dist_file = os.path.join(opt.store_dir, f'{desc}_hypro_sample.pkl')
    data = {'input_time': input_time, 'input_events': input_events, 'tau_sampled': tau_sampled, 'events_sampled': events_sampled}
    dump_to_pkl(data, mae_e_dist_file, compression = 'bz2')
    write_yaml({**opt.info_dict, 'hypro_length': opt.number_of_events_hypro, 'hypro_negative_samples': opt.number_of_negative_samples}, 
               opt.store_dir, 'dataset_card.yml')


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
    'spearman_and_l1': {'desc_string': 'Spearman and L1 for {0}', 'postprocess_func': spearman_and_l1_postprocess},
    'mae_and_f1': {'desc_string': 'MAE and macro-f1 for {0}', 'postprocess_func': mae_and_f1_postprocess},
    'mae_e_and_f1': {'desc_string': 'MAE-E and macro-f1 for {0}', 'postprocess_func': mae_e_and_f1_postprocess},
    'mae_e_and_f1_by_time_event': {'desc_string': 'MAE-E and macro-f1 for {0} following NER', 'postprocess_func': mae_e_and_f1_by_time_event_postprocess},
    'which_event_occurs_first': {'desc_string': 'Predict the next event by finding which event occurs first for {0}', 'postprocess_func': which_event_occurs_first_postprocess},
    'samples_from_et': {'desc_string': 'Samples of {0} for each mark', 'postprocess_func': samples_from_et_postprocess},
    'generate_hypro_dataset': {'desc_string': 'Generate HYPRO dataset for {0}', 'postprocess_func': generate_hypro_dataset_postprocess},

    # CPPOD task.
    'cppod_evaluation': {'desc_string': 'Obtaining CPPOD score for {0}', 'postprocess_func': cppod_evaluation_postprocess},
    'cppod_commission_evaluation': {'desc_string': 'Obtaining CPPOD score on commission outlier for {0}', 'postprocess_func': cppod_commission_evaluation_postprocess},


    # Custom evaluation function.
    'mae_and_f1_of_imputated_events': mae_and_f1_of_imputated_events
}