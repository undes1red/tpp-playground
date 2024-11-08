import os
import numpy as np
from src.toolbox.misc import get_logger, mkdir_if_not_exist, dump_to_pkl, write_to_txt, flatten


logger = get_logger(name = __file__)


def long_horizon_pred_postprocess(all_evaluation_results, desc, opt):
    '''
    This function is called when task_name = spearman_and_l1.

    This function calculates the average of spearman and L^1 distance between the learned probability distribution
    and the ground truth on all synthetic event sequences.
    '''
    otd_picked_by_hypro, otd_average_sample = all_evaluation_results
    otd_picked_by_hypro = np.array(otd_picked_by_hypro).mean(axis = 0)
    otd_average_sample = np.array(otd_average_sample).mean(axis = 0)

    result_file = os.path.join(opt.store_dir, f'{desc}_otd.txt')
    strings = f'For the {desc} of {opt.dataset_name}, we announce that the OTD of sequences selected by HYPRO is {otd_picked_by_hypro}, while the average is {otd_average_sample}.'
    write_to_txt(strings, result_file)


desc_funcs = {
    'long_horizon_pred': {'desc_string': 'Evaluate sequences picked by HYPRO on {0}', 'postprocess_func': long_horizon_pred_postprocess},
}