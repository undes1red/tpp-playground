# You can use this file if you are too lazy to create and modify script files.
# This file can pack numerous tasks and run them one by one automatically.

import os, argparse, importlib, copy, sys
from batch_task_worker_utils import task_generator_worker, translate_dict_to_arguments, monitor_and_automaticly_run_tasks, read_yaml
from src.taskhost import get_logger


logger = get_logger(__name__)
root_path = os.path.dirname(os.path.abspath(__file__))
logger.info(f'project root is {root_path}.')
logger.info(f'Please ensure the root_path is correct!')

parser = argparse.ArgumentParser()
parser.add_argument('--script_type', type = str, choices = ['normal', 'previous_failed_tasks'], default = 'normal',\
                                     help = 'Use this argument to select worker mode.\n \
                                             normal: In this mode, the script will pick job dict according to the received job_name. Failed tasks will be recorded in {model_name}_previous_failed_tasks.txt. \n \
                                             previous_failed_tasks: In this mode, this script will read in tasks from parameter_set/{procedure_name}/{model}_previous_failed_tasks.txt and execute these tasks one by one.')
parser.add_argument('--procedure_name', type = str, choices = ['TPP', 'LH', 'OD', 'MDI'], \
                                        help = 'You need this argument to select the proper parameter set.')
parser.add_argument('--model', type = str, help = 'We use this model name to select correct parameter collection.')
parser.add_argument('--job_name', nargs='+', default = None, help='Tell us which job do you want to execute. This argument accepts multiple inputs so you can use "--job_name A B C" to run job A B C one by one. \n \
                                                                   None (Default): no job will be executed. \n \
                                                                   ALL (special): execute all jobs. \n \
                                                                   This argument will be ignored in the previous_failed_tasks mode.')
parser.add_argument('--GPU', nargs='+', default = None, help='How many GPU do you want to use? Tell us the ID of available GPUs, \
                                                              or set it to a negative number or None to go CPU-only.')
parser.add_argument('--num_task_parallel', type = int, default = -1, help = 'The number of tasks we should run in parallel. In GPU mode this number should not bigger than the number of available GPUs. \
                                                                             The default value, -1, will automatically use all GPUs, one GPU for one task. \
                                                                             This argument is mandatory when executing tasks on CPU or submitted through slurm.')
parser.add_argument('--slurm', action  ='store_true', help = 'Submit tasks through slurm.')
parser.add_argument('--slurm_config', type = str, help = 'This argument links to a config file to set up new slurm quota when you have more resources to run your tasks. We will use the default quota if no config is given.')
parser.add_argument('--sleep', type = int, default = 0, help = 'This argument links to a config file to set up new slurm quota when you have more resources to run your tasks. We will use the default quota if no config is given.')
parser.add_argument('--interpreter', type = str, default = 'python3', help = 'This argument links to a config file to set up new slurm quota when you have more resources to run your tasks. We will use the default quota if no config is given.')


# Preprocess
opt = parser.parse_args()

import time
time.sleep(opt.sleep)

use_gpu = False
slurm_arguments = {}
if opt.GPU is not None:
    if not opt.slurm:
        gpu_pool = [int(gpu_id) for gpu_id in opt.GPU]
        if len([gpu_id for gpu_id in gpu_pool if gpu_id < 0]) == 0:
            assert opt.num_task_parallel <= len(gpu_pool)
            use_gpu = True
            if opt.num_task_parallel == -1:
                opt.num_task_parallel = len(gpu_pool)
    else:
        use_gpu = True
        gpu_pool = [int(gpu_id) for gpu_id in opt.GPU] * opt.num_task_parallel
        slurm_arguments = {}
        if opt.slurm_config is not None:
            slurm_arguments = read_yaml(os.path.join(root_path, opt.slurm_config))


if not use_gpu:
    gpu_pool = []


# stdout dir
# where we store printed logs of each task.
stdout_dir = os.path.join(root_path, 'stdout', opt.procedure_name, opt.script_type, opt.model)
if not os.path.exists(stdout_dir):
    os.makedirs(stdout_dir)


def task_generator(worker, hyperparameter_list):
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
    file_name = os.path.join(root_path, worker)
    argparser = opt.procedure_name + '_' + hyperparameter_list['job_type']

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
                [opt.interpreter, file_name, argparser] + \
                translate_dict_to_arguments(parameter_buffer)
            )

    logger.info(f'We have planned {len(generated_hyperparameter_list)} tasks!')
    return generated_hyperparameter_list, len(generated_hyperparameter_list)


if opt.script_type == 'previous_failed_tasks':
    logger.info(f'We are in previous_failed_tasks mode. We will read in and rerun failed commands recorded in {opt.model}_previous_failed_tasks.txt.')
    try:
        f_previous_failed_tasks = open(os.path.join(root_path, 'new_parameter_set', opt.procedure_name, f'{opt.model}_previous_failed_tasks.txt'), 'r')
    except FileNotFoundError as e:
        logger.exception(f"File {os.path.join('new_parameter_set', opt.procedure_name, f'{opt.model}_previous_failed_tasks.txt')} not found!")
    except Exception as e:
        raise e
    
    generated_tasks = []
    for command in f_previous_failed_tasks:
        generated_tasks.append(command.strip())
    the_number_of_task = len(generated_tasks)

    failed_tasks = monitor_and_automaticly_run_tasks(generated_tasks, use_gpu, gpu_pool, opt.num_task_parallel, stdout_dir, opt.slurm, slurm_arguments = slurm_arguments)
    
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
            logger.warning(f'----> Task No.{index} has failed. <----')
            logger.warning(f'Task Command: {command}.')
            failed_commands.append(command + '\n')
    
    '''
    Only in previous_failed_tasks mode we can rewrite the previous_failed_tasks.txt.
    By this we can avoid missing failed tasks in the previous task sets if the execution script calls batch_task_worker.py multiple times.
    '''
    f_previous_failed_tasks = open(os.path.join(root_path, 'new_parameter_set', opt.procedure_name, f'{opt.model}_previous_failed_tasks.txt'), 'w')
    f_previous_failed_tasks.writelines(failed_commands)
    f_previous_failed_tasks.close()
else:
    parameter_lib = importlib.import_module(f'.{opt.procedure_name}', package = 'new_parameter_set')
    parameter_retriver = getattr(parameter_lib, 'parameter_retriver')
    full_job_list = parameter_retriver(opt)
    worker = full_job_list['file_name']
    
    logger.info(f'We call {worker} to run our jobs.')
    if opt.job_name is None:
        logger.warning(f'No job selected! Exiting...')
        sys.exit(0)
    elif opt.job_name == ['ALL',]:
        opt.job_name = full_job_list['jobs'].keys()

    logger.info(f'We will execute the following jobs: {opt.job_name}.')
    logger.info(f'All available jobs: {full_job_list['jobs'].keys()}.')
    
    for job in opt.job_name:
        job_content = full_job_list['jobs'][job]
        
        # condition 1: a single dict.
        if isinstance(job_content, dict):
            job_content = [job_content, ]
        logger.info(f'Current executing the job: {job}. It has {len(job_content)} subjobs.')
        
        # condition 2: a list.
        # Extract the list and run the tasks one by one.
        for idx, sub_job in enumerate(job_content):
            logger.warning(f'============ subjob No. {idx + 1} started ============')
            generated_tasks, the_number_of_task = task_generator(worker, sub_job)
            generated_tasks = [' '.join(sub_task) for sub_task in generated_tasks]
            
            failed_tasks = monitor_and_automaticly_run_tasks(generated_tasks, use_gpu, gpu_pool, opt.num_task_parallel, stdout_dir, opt.slurm, slurm_arguments = slurm_arguments)
            
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
                    logger.warning(f'----> Task No.{index} has failed. <----')
                    logger.warning(f'Task Command: {command}.')
                    failed_commands.append(command + '\n')
            
            '''
            Only in previous_failed_tasks mode we can rewrite the previous_failed_tasks.txt.
            By this we can avoid missing failed tasks in the previous task sets if the execution script calls batch_task_worker.py multiple times.
            '''
            f_previous_failed_tasks = open(os.path.join(root_path, 'new_parameter_set', opt.procedure_name, f'{opt.model}_previous_failed_tasks.txt'), 'a')
            f_previous_failed_tasks.writelines(failed_commands)
            f_previous_failed_tasks.close()

            logger.warning(f'============ subjob No. {idx + 1} ended ============')