# You can use this file if you are too lazy to create and modify script files.
# Just pack numerous tasks and run them one by one automatically.

import subprocess, os, argparse, itertools, importlib
from batch_task_worker_utils import task_index_generator, task_counting_generator, remove_empty_str
from src.taskhost import getLogger


logger = getLogger(__name__)
root_path = os.path.dirname(os.path.abspath(__file__))
logger.info(f'project root is {root_path}.')
logger.info(f'Please ensure the root_path is correct!')

parser = argparse.ArgumentParser()
parser.add_argument('--script_type', type = str, choices = ['train', 'plot'], default = 'train',\
                                     help = 'You can use this only argument to select what you want to do.')
parser.add_argument('--procedure_name', type = str, choices = ['TPP'], \
                                     help = 'You need this argument to select the proper parameter set.')
parser.add_argument('--GPU', type = int, default = None, help='How many GPU you want to use? Set it to None to use all GPUs, \
                                                                 or set it to negative number or None for CPU learning.')
parser.add_argument('--dataset', type = str, help = 'The dataset name to select correct parameter collection from the parameter dict.')
parser.add_argument('--model', type = str, help = 'The model name to select correct parameter collection from the parameter dict.')
parser.add_argument('--style', type = str, default = 'counting', help = 'How do we enumerate hyperparameters from the hyperparameter list?')

opt = parser.parse_args()
# Environment variables
do_not_use_gpu = False
if opt.GPU is not None and opt.GPU >= 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.GPU)
else:
    do_not_use_gpu = True


def task_generator(hyperparameter_list):
    '''
    [
        (other single hyperparameters),
        "counting": 
        [
            (hyperparameter lists)
        ],
        "index":
        [
            (hyperparameter lists)
        ]
    ]
    '''
    file_name = os.path.join(root_path, hyperparameter_list['file_name'])
    argparser = [opt.procedure_name + '_' + opt.script_type]

    single_hyperparameters = hyperparameter_list['single']
    index_hyperparameters = hyperparameter_list.get('index')
    counting_hyperparameters = hyperparameter_list.get('counting')

    index_hyperparameters_list = task_index_generator(index_hyperparameters)
    counting_hyperparameters_list = task_counting_generator(counting_hyperparameters)

    logger.info(f'We have planned {len(index_hyperparameters_list) * len(counting_hyperparameters_list)} tasks!')

    for index_hyperparameter_list in index_hyperparameters_list:
        for counting_hyperparameter_list in counting_hyperparameters_list:
            yield remove_empty_str([file_name] + [argparser] + single_hyperparameters + index_hyperparameter_list + counting_hyperparameter_list)


task_count = 1
parameter_lib = importlib.import_module(f'.{opt.procedure_name}', package = 'parameter_set')
parameter_retriver = getattr(parameter_lib, 'parameter_retriver')
for hp_list in task_generator(parameter_retriver(opt)):
    if not do_not_use_gpu:
        hp_list.append("--cuda")
    # process = subprocess.Popen([
    #         'python3'
    # ] + hp_list)
    # process.wait()
    logger.warning(f'----> Task {task_count} completed. <----')
    task_count += 1