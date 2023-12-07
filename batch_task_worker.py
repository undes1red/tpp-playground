# You can use this file if you are too lazy to create and modify script files.
# Just pack numerous tasks and run them one by one automatically.

import subprocess, os, argparse, importlib
from batch_task_worker_utils import task_generator_worker, remove_empty_str
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
parser.add_argument('--GPU', type = int, default = None, help='How many GPU you want to use? Set it to a positive number to use all GPUs, \
                                                                 or set it to a negative number or None to go CPU-only.')
parser.add_argument('--dataset', type = str, help = 'The dataset name to select correct parameter collection from the parameter dict.')
parser.add_argument('--model', type = str, help = 'The model name to select correct parameter collection from the parameter dict.')

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
    argparser = opt.procedure_name + '_' + opt.script_type

    single_hyperparameters = hyperparameter_list.get('single')
    single_hyperparameters = single_hyperparameters if single_hyperparameters is not None else ['']
    index_hyperparameters = hyperparameter_list.get('index')
    counting_hyperparameters = hyperparameter_list.get('counting')

    index_hyperparameters_list, _ = task_generator_worker(index_hyperparameters, 'index')
    counting_hyperparameters_list, _ = task_generator_worker(counting_hyperparameters, 'counting')
    
    generated_hyperparameter_list = []
    for index_hyperparameter_list in index_hyperparameters_list:
        for counting_hyperparameter_list in counting_hyperparameters_list:
            generated_hyperparameter_list.append(
                remove_empty_str([file_name] + [argparser] + single_hyperparameters + index_hyperparameter_list + counting_hyperparameter_list)
            )

    logger.info(f'We have planned {len(generated_hyperparameter_list)} tasks!')
    return generated_hyperparameter_list, len(generated_hyperparameter_list)


task_count = 0
parameter_lib = importlib.import_module(f'.{opt.procedure_name}', package = 'parameter_set')
parameter_retriver = getattr(parameter_lib, 'parameter_retriver')
generated_hyperparameter_list, the_number_of_task = task_generator(parameter_retriver(opt))

'''
run all planned tasks via a loop.
'''
failed_tasks = {}
for hp_list in generated_hyperparameter_list:
    task_count += 1

    # Assemble the command string.
    if not do_not_use_gpu:
        hp_list.append("--cuda")
    command = ['python3'] + hp_list
    command_string = " ".join(command)
    logger.warning(f'----> Task {task_count}/{the_number_of_task} started. <----')
    logger.info(f'Command of task {task_count}/{the_number_of_task}: {command_string}')

    # Create and run the task.
    try:
        subprocess.run(command_string, shell = True, check = True, stderr = subprocess.PIPE)
        logger.warning(f'----> Task {task_count}/{the_number_of_task} completed. <----')
    except subprocess.CalledProcessError as e:
        failed_tasks[task_count] = e
        logger.warning(f'----> Task {task_count}/{the_number_of_task} Failed!. <----')

# Report the execution sumamry:
logger.warning(f'==========================================')
logger.warning(f'                Summary                   ')
logger.warning(f'==========================================')
if len(failed_tasks) == 0:
    logger.info(f'All {the_number_of_task} tasks have successfully completed.')
else:
    logger.warning(f'{len(failed_tasks)} tasks have failed. Please check what is wrong and fix them!')
    for index, error_info in failed_tasks.items():
        logger.warning(f'----> Task {index} has failed. <----')
        logger.warning(f'Return Code: {error_info.returncode}.')
        logger.warning(f'Task Command: {error_info.cmd}.')
        logger.warning(f'Exception: {error_info.stderr.decode("UTF-8")}.')