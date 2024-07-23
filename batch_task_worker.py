# You can use this file if you are too lazy to create and modify script files.
# Just pack numerous tasks and run them one by one automatically.

import os, argparse, importlib, copy
from batch_task_worker_utils import task_generator_worker, translate_dict_to_arguments, monitor_and_automaticly_run_tasks
from src.taskhost import get_logger


logger = get_logger(__name__)
root_path = os.path.dirname(os.path.abspath(__file__))
logger.info(f'project root is {root_path}.')
logger.info(f'Please ensure the root_path is correct!')

parser = argparse.ArgumentParser()
parser.add_argument('--script_type', type = str, choices = ['train', 'evaluate', 'previous_failed_tasks'], default = 'train',\
                                     help = 'Use this argument to select worker mode.\n \
                                             train: training mode. Execute training tasks defined in parameter_set/{procedure_name} one by one.\n \
                                             evaluate: evaluation mode. Execute Evaluation tasks defined in parameter_set/{procedure_name} one by one.\n \
                                             previous_failed_tasks: In this mode, this script will read in tasks from parameter_set/{procedure_name}/{model}_previous_failed_tasks.txt and execute these tasks one by one.')
parser.add_argument('--procedure_name', type = str, choices = ['TPP'], \
                                        help = 'You need this argument to select the proper parameter set.')
parser.add_argument('--GPU', nargs='+', default = None, help='How many GPU you want to use? Tell us the ID of available GPUs, \
                                                              or set it to a negative number or None to go CPU-only.')
parser.add_argument('--dataset', type = str, help = 'We use this dataset name to select correct parameter collection from the parameter dict.')
parser.add_argument('--model', type = str, help = 'We use this model name to select correct parameter collection from the parameter dict.')
parser.add_argument('--num_task_parallel', type = int, default = -1, help = 'The number of tasks we should run in parallel. In GPU mode this number should not bigger than the number of available GPUs. \
                                                                             The default value, -1, will automatically use all GPUs, one GPU for one task. \
                                                                             This argument is mandatory when executing tasks on CPU.')

# Preprocess
opt = parser.parse_args()
use_gpu = False
if opt.GPU is not None:
    gpu_pool = [int(gpu_id) for gpu_id in opt.GPU]
    if len([gpu_id for gpu_id in gpu_pool if gpu_id < 0]) == 0:
        assert opt.num_task_parallel <= len(gpu_pool)
        use_gpu = True
        if opt.num_task_parallel == -1:
            opt.num_task_parallel = len(gpu_pool)

if not use_gpu:
    gpu_pool = []


# stdout dir
# where we store printed logs of each task.
stdout_dir = os.path.join(root_path, 'stdout', opt.procedure_name, opt.script_type, opt.dataset, opt.model)
if not os.path.exists(stdout_dir):
    os.makedirs(stdout_dir)


def task_generator(hyperparameter_list):
    '''
    [
        (other single hyperparameters),
        "counting_style": 
        [
            (hyperparameter lists)
        ],
        "zip_style":
        [
            (hyperparameter lists)
        ]
    ]
    '''
    file_name = os.path.join(root_path, hyperparameter_list['file_name'])
    argparser = opt.procedure_name + '_' + opt.script_type

    static_hyperparameters = hyperparameter_list.get('static', {})
    zip_style_hyperparameters = hyperparameter_list.get('zip_style', {})
    counting_style_hyperparameters = hyperparameter_list.get('counting_style', {})

    generated_zip_style_hyperparameters, _ = task_generator_worker(zip_style_hyperparameters, 'zip_style')
    generated_counting_style_hyperparameters, _ = task_generator_worker(counting_style_hyperparameters, 'counting_style')
    
    generated_hyperparameter_list = []
    for zip_style_hyperparameter in generated_zip_style_hyperparameters:
        for counting_hyperparameter in generated_counting_style_hyperparameters:
            parameter_buffer = copy.deepcopy(static_hyperparameters)
            parameter_buffer.update(zip_style_hyperparameter)
            parameter_buffer.update(counting_hyperparameter)
            generated_hyperparameter_list.append(
                [file_name, argparser] + \
                translate_dict_to_arguments(parameter_buffer)
            )

    logger.info(f'We have planned {len(generated_hyperparameter_list)} tasks!')
    return generated_hyperparameter_list, len(generated_hyperparameter_list)


generated_tasks = []
the_number_of_task = 0
if opt.script_type == 'previous_failed_tasks':
    logger.info(f'We are in previous_failed_tasks mode. We will read in and rerun failed commands recorded in {opt.model}_previous_failed_tasks.txt.')
    try:
        f_previous_failed_tasks = open(os.path.join(root_path, 'parameter_set', opt.procedure_name, f'{opt.model}_previous_failed_tasks.txt'), 'r')
    except FileNotFoundError as e:
        logger.exception(f"File {os.path.join('parameter_set', opt.procedure_name, f'{opt.model}_previous_failed_tasks.txt')} not found!")
    except Exception as e:
        raise e
    
    generated_tasks = f_previous_failed_tasks.readlines()
    the_number_of_task = len(generated_tasks)
else:
    parameter_lib = importlib.import_module(f'.{opt.procedure_name}', package = 'parameter_set')
    parameter_retriver = getattr(parameter_lib, 'parameter_retriver')
    generated_hyperparameter_list, the_number_of_task = task_generator(parameter_retriver(opt))
    for hp_list in generated_hyperparameter_list:
        # Assemble the command list into a string.
        task = ['python3'] + hp_list
        generated_tasks.append(' '.join(task))
    

failed_tasks = monitor_and_automaticly_run_tasks(generated_tasks, use_gpu, gpu_pool, opt.num_task_parallel, stdout_dir)

# Report the execution sumamry:
logger.warning('==========================================')
logger.warning('                Summary                   ')
logger.warning('==========================================')
failed_commands = []
if len(failed_tasks) == 0:
    logger.info(f'All {the_number_of_task} tasks have successfully completed.')
else:
    logger.warning(f'{len(failed_tasks)} tasks have failed. Please check what is wrong according to logs in directory stdout/ and fix them!')
    for index, command in failed_tasks.items():
        logger.warning(f'----> Task {index} has failed. <----')
        logger.warning(f'Task Command: {command}.')
        failed_commands.append(command + '\n')

'''
Only in previous_failed_tasks mode we can rewrite the previous_failed_tasks.txt.
By this we can avoid missing failed tasks in the previous task sets if the execution script calls batch_task_worker.py multiple times.
'''
f_previous_failed_tasks = open(os.path.join(root_path, 'parameter_set', opt.procedure_name, f'{opt.model}_previous_failed_tasks.txt'), \
                          'w' if opt.script_type == 'previous_failed_tasks' else 'a')
f_previous_failed_tasks.writelines(failed_commands)
f_previous_failed_tasks.close()