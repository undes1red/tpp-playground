import os
import numpy as np
from src.toolbox.misc import get_logger, mkdir_if_not_exist, dump_to_pkl, write_to_txt, flatten


logger = get_logger(name = __file__)


def probability_nhps_nhpf_postprocess(all_evaluation_results, desc, opt):
    '''
    This function is called when task_name = spearman_and_l1.

    This function calculates the average of spearman and L^1 distance between the learned probability distribution
    and the ground truth on all synthetic event sequences.
    '''
    log_p_z_x, log_q_z_con_x_on_all_samples = all_evaluation_results
    log_p_z_x = np.concatenate(log_p_z_x)
    log_p_z_x_mean = log_p_z_x.mean()
    log_q_z_con_x_on_all_samples = np.array(log_q_z_con_x_on_all_samples)
    log_q_z_con_x_on_all_samples_mean = log_q_z_con_x_on_all_samples.mean()

    result_file = os.path.join(opt.store_dir, f'{desc}_prob_nhpf_nhps.txt')
    strings = f'For the {desc} of {opt.dataset_name}, we announce that the probability on missing events by NHPF is {log_p_z_x_mean}, while the probability on missing events by NHPS is {log_q_z_con_x_on_all_samples_mean}.'
    write_to_txt(strings, result_file)
    
    result = {'log_p_z_x': log_p_z_x_mean, 'log_q_z_con_x_on_all_samples': log_q_z_con_x_on_all_samples_mean}
    prob_comparison_result = os.path.join(opt.store_dir, f'{desc}_prob_nhpf_nhps_result.pkl')
    dump_to_pkl(result, prob_comparison_result, compression = 'bz2')
    
    data = {'log_p_z_x': log_p_z_x, 'log_q_z_con_x_on_all_samples': log_q_z_con_x_on_all_samples}
    prob_comparison_result = os.path.join(opt.store_dir, f'{desc}_prob_nhpf_nhps.pkl')
    dump_to_pkl(data, prob_comparison_result, compression = 'bz2')


def imputed_seq_by_nhpf_postprocess(all_evaluation_results, desc, opt):
    '''
    This function is called when task_name = spearman_and_l1.

    This function calculates the average of spearman and L^1 distance between the learned probability distribution
    and the ground truth on all synthetic event sequences.
    '''
    complete_time, complete_events, complete_mask, weights, imputed_seqs = all_evaluation_results
    costs = opt.event_del_costs
    
    data = {'complete_time': complete_time, 'complete_events': complete_events, 'complete_mask': complete_mask, 
            'weights': weights, 'imputed_seqs': imputed_seqs, 'costs': costs}
    prob_comparison_result = os.path.join(opt.store_dir, f'{desc}_imputed_seq_by_nhpf.pkl')
    dump_to_pkl(data, prob_comparison_result, compression = 'bz2')


def imputed_seq_by_nhps_postprocess(all_evaluation_results, desc, opt):
    '''
    This function is called when task_name = spearman_and_l1.

    This function calculates the average of spearman and L^1 distance between the learned probability distribution
    and the ground truth on all synthetic event sequences.
    '''
    complete_time, complete_events, complete_mask, weights, imputed_seqs = all_evaluation_results
    costs = opt.event_del_costs
    
    data = {'complete_time': complete_time, 'complete_events': complete_events, 'complete_mask': complete_mask, 
            'weights': weights, 'imputed_seqs': imputed_seqs, 'costs': costs}
    prob_comparison_result = os.path.join(opt.store_dir, f'{desc}_imputed_seq_by_nhps.pkl')
    dump_to_pkl(data, prob_comparison_result, compression = 'bz2')


def otd_nhpf_postprocess(all_evaluation_results, desc, opt):
    '''
    This function is called when task_name = spearman_and_l1.

    This function calculates the average of spearman and L^1 distance between the learned probability distribution
    and the ground truth on all synthetic event sequences.
    '''
    otds = all_evaluation_results[0]
    otds_all = np.stack(otds, axis = 0)
    otd_mean = otds_all.mean(axis = 0)

    result_file = os.path.join(opt.store_dir, f'{desc}_nhpf_otds.txt')
    strings = f'For the {desc} of {opt.dataset_name}, we announce that otd between imputed and original sequences is {otd_mean}.'
    write_to_txt(strings, result_file)
    
    result = {'otds': otds_all}
    prob_comparison_result = os.path.join(opt.store_dir, f'{desc}_nhpf_otd_result.pkl')
    dump_to_pkl(result, prob_comparison_result, compression = 'bz2')
    

def otd_nhps_postprocess(all_evaluation_results, desc, opt):
    '''
    This function is called when task_name = spearman_and_l1.

    This function calculates the average of spearman and L^1 distance between the learned probability distribution
    and the ground truth on all synthetic event sequences.
    '''
    otds = all_evaluation_results[0]
    otds_all = np.stack(otds, axis = 0)
    otd_mean = otds_all.mean(axis = 0)

    result_file = os.path.join(opt.store_dir, f'{desc}_nhps_otds.txt')
    strings = f'For the {desc} of {opt.dataset_name}, we announce that otd between imputed and original sequences is {otd_mean}.'
    write_to_txt(strings, result_file)
    
    result = {'otds': otds_all}
    prob_comparison_result = os.path.join(opt.store_dir, f'{desc}_nhps_otd_result.pkl')
    dump_to_pkl(result, prob_comparison_result, compression = 'bz2')
    

desc_funcs = {
    'probability_nhps_nhpf': {'desc_string': 'Comparison probability at missing events between NHPF and NHPS on {0}', 'postprocess_func': probability_nhps_nhpf_postprocess},
    
    'get_imputed_seq_by_nhpf': {'desc_string': 'Get imputed sequences by NHPF for {0}', 'postprocess_func': imputed_seq_by_nhpf_postprocess},
    'get_imputed_seq_by_nhps': {'desc_string': 'Get imputed sequences by NHPS for {0}', 'postprocess_func': imputed_seq_by_nhps_postprocess},

    'otd_nhpf': {'desc_string': 'Calculating OTD on NHPF samples on {0}', 'postprocess_func': otd_nhpf_postprocess},
    'otd_nhps': {'desc_string': 'Calculating OTD on NHPS samples on {0}', 'postprocess_func': otd_nhps_postprocess},
}