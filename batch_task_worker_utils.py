import itertools, math, subprocess, time, os
from src.taskhost import get_logger


logger = get_logger(__name__)

monitor_frequency = 10
def monitor_and_automatic_run_tasks(tasks, use_gpu, available_gpus, num_task_parallel, stdout_dir):        
    if use_gpu:
        return monitor_and_automatic_run_tasks_on_gpu(tasks, available_gpus, num_task_parallel, stdout_dir)
    else:
        return monitor_and_automatic_run_tasks_on_cpu(tasks, num_task_parallel, stdout_dir)
    

def extract_single_multiple_arguments_from_the_list(hyperparameter_list):
    single_parameters = {}
    multiple_parameters = {}

    last_parameter = ''
    for items in hyperparameter_list:
        if last_parameter == '':
            '''
            new arguments:
            '''
            last_parameter = items
        elif last_parameter.startswith('-') and type(items) == list:
            '''
            arguments with multiple choices
            '''
            multiple_parameters[last_parameter] = items
            last_parameter = ''
        elif last_parameter.startswith('-') and items.startswith('--'):
            '''
            store_true arguments
            '''
            single_parameters[last_parameter] = ''
            last_parameter = items
        elif last_parameter.startswith('-') and not items.startswith('--'):
            '''
            arguments with single choice
            '''
            single_parameters[last_parameter] = items
            last_parameter = ''
    if last_parameter != '':
        single_parameters[last_parameter] = ''

    return single_parameters, multiple_parameters


def task_index_generator(hyperparameter_list):
    if hyperparameter_list is None:
        return [[''],]

    # head = os.path.join(root_path, hyperparameter_list[0])
    single_parameters, multiple_parameters = extract_single_multiple_arguments_from_the_list(hyperparameter_list)

    # Now, map all fixed argument into a list.
    # fixed_arguments_part = [head] + [opt.procedure_name + '_' + opt.script_type] \
    fixed_arguments_part = list(itertools.chain.from_iterable(single_parameters.items()))
    
    count_of_each_multiple_hp = len(multiple_parameters)

    if count_of_each_multiple_hp == 0:
        '''
        No multi_hps, just return the fixed parameter set.
        '''
        return [fixed_arguments_part]
    
    '''
    Affirm that all hyperparameter lists have the same length.
    '''
    number_of_parameters = len(list(multiple_parameters.values())[0])
    for hyperparameter in multiple_parameters.values():
        assert len(hyperparameter) == number_of_parameters, "Index mode requires all parameter list must have the same length!"

    # set iterators, the first iterator is always the single directed iterator. We use it to decide when we quit the argument
    # generation loop.
    final_hyperparameter_list = []
    for index in range(number_of_parameters):
        choosed_value = {key: item[index] for (key, item) in multiple_parameters.items()}
        choosed_value_to_list = list(itertools.chain.from_iterable(choosed_value.items()))
        final_list = fixed_arguments_part + choosed_value_to_list
        final_hyperparameter_list.append(final_list)
        
    return final_hyperparameter_list
        

def task_counting_generator(hyperparameter_list):
    if hyperparameter_list is None:
        return [[''],]

    # head = os.path.join(root_path, hyperparameter_list[0])
    single_parameters, multiple_parameters = extract_single_multiple_arguments_from_the_list(hyperparameter_list)
    
    # Now, map all fixed argument into a list.
    # fixed_arguments_part = [head] + [opt.procedure_name + '_' + opt.script_type] \
    fixed_arguments_part = list(itertools.chain.from_iterable(single_parameters.items()))

    # set iterators, the first iterator is always the single directed iterator. We use it to decide when we quit the argument
    # generation loop.
    multi_hp_count = len(multiple_parameters.values())
    count_of_each_multiple_hp = [len(item) for item in multiple_parameters.values()]
    current_index_of_each_list = [0] * multi_hp_count
    the_number_of_task = math.prod(count_of_each_multiple_hp)

    if count_of_each_multiple_hp == []:
        # No multiple hp is present.
        return [fixed_arguments_part]
    else:
        final_hyperparameter_list = []
        for _ in range(the_number_of_task):
            choosed_value = {key: item[index] for (key, item), index in zip(multiple_parameters.items(), current_index_of_each_list)}
            choosed_value_to_list = list(itertools.chain.from_iterable(choosed_value.items()))
            final_list = fixed_arguments_part + choosed_value_to_list
            final_hyperparameter_list.append(final_list)

            current_index_of_each_list[-1] += 1
            add_mark = False
            for idx, (current_index, max_unreachable_index) in enumerate(zip(current_index_of_each_list[::-1], count_of_each_multiple_hp[::-1])):
                if add_mark:
                    current_index_of_each_list[multi_hp_count - idx - 1] += 1
                    if current_index_of_each_list[multi_hp_count - idx - 1]  >= max_unreachable_index:
                        current_index_of_each_list[multi_hp_count - idx - 1] = 0
                        add_mark = True
                    else:
                        add_mark = False
                if current_index >= max_unreachable_index:
                    current_index_of_each_list[multi_hp_count - idx - 1] = 0
                    add_mark = True
        
        return final_hyperparameter_list


def remove_empty_str(x):
    try:
        while 1:
            x.remove('')
    except:
        return x


hyperparameter_parser = {
    'index': task_index_generator,
    'counting': task_counting_generator
}


def task_generator_worker(hyperparameter_list, iterate_style):
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
    if hyperparameter_list is None:
        return [['']], 0


    single_hyperparameters = hyperparameter_list.get('single')
    multiple_hyperparameters = hyperparameter_list.get('multiple')
    index_hyperparameters = hyperparameter_list.get('index')
    counting_hyperparameters = hyperparameter_list.get('counting')

    single_hyperparameters = single_hyperparameters if single_hyperparameters is not None else ['']
    multiple_hyperparameters = hyperparameter_parser[iterate_style](multiple_hyperparameters)
    multiple_the_number_of_task = len(multiple_hyperparameters) if multiple_hyperparameters != [[''],] else 0
    index_hyperparameters_list, index_the_number_of_task = task_generator_worker(index_hyperparameters, iterate_style = 'index')
    counting_hyperparameters_list, counting_the_number_of_task = task_generator_worker(counting_hyperparameters, iterate_style = 'counting')
    
    # specifically used when iterate_style == 'index'
    tmp_length = 0
    if iterate_style == 'index':
        '''
        Ensure that all hyperparameter lists ready for index enumeration have the same length.
        '''
        tmp_length = max(multiple_the_number_of_task, index_the_number_of_task, counting_the_number_of_task)
        multiple_the_number_of_task_for_comp = multiple_the_number_of_task if multiple_the_number_of_task > 0 else tmp_length
        index_the_number_of_task_for_comp = index_the_number_of_task if index_the_number_of_task > 0 else tmp_length
        counting_the_number_of_task_for_comp = counting_the_number_of_task if counting_the_number_of_task > 0 else tmp_length
        assert multiple_the_number_of_task_for_comp == index_the_number_of_task_for_comp == counting_the_number_of_task_for_comp

    generated_hyperparameter_list = []
    if iterate_style == 'index':
        packed_data = zip(multiple_hyperparameters if multiple_the_number_of_task > 0 else [['']] * tmp_length,
                          index_hyperparameters_list if index_the_number_of_task > 0 else [['']] * tmp_length, 
                          counting_hyperparameters_list if counting_the_number_of_task > 0 else [['']] * tmp_length)
        for mh, ih, ch in packed_data:
            generated_hyperparameter_list.append(
                single_hyperparameters + mh + ih + ch
            )
    else:
        for multiple_hyperparameter_list in multiple_hyperparameters:
            for index_hyperparameter_list in index_hyperparameters_list:
                for counting_hyperparameter_list in counting_hyperparameters_list:
                    generated_hyperparameter_list.append(
                        single_hyperparameters + multiple_hyperparameter_list + index_hyperparameter_list + counting_hyperparameter_list
                    )
        
    return generated_hyperparameter_list, len(generated_hyperparameter_list)


def monitor_and_automatic_run_tasks_on_cpu(tasks, num_task_parallel, stdout_dir):
    number_of_tasks = len(tasks)

    def run_task(task, task_id):
        task_list = task.split(' ')

        # Replace this command with your actual task command
        logger.warning(f'----> Task {task_id}/{number_of_tasks} started. <----')
        logger.info(f'Command of task {task_id}/{number_of_tasks}: {task}')
        f_log = open(os.path.join(stdout_dir, f'stdout_log_{task_id}.txt'), 'w')
        process = subprocess.Popen(task_list, stdout = f_log, stderr = f_log, universal_newlines = True)

        return process, f_log

    task_id = 1
    running_tasks = []
    number_of_running_tasks = 0
    completed_tasks = set()
    all_task_executed = False
    failed_tasks = {}

    while True:
        if task_id > number_of_tasks:
            all_task_executed = True

        if number_of_running_tasks < num_task_parallel and not all_task_executed:
            process, log_file = run_task(tasks[task_id - 1], task_id)
            running_tasks.append({'task_id': task_id, 'process': process, 'stdout': log_file})
            task_id += 1
            number_of_running_tasks += 1

        # Check if one task has finished. If so, do some housekeeping 
        # and add the allocated gpu_id back to the gpu_pool, marking this GPU is now free.
        for task in running_tasks:
            if task["task_id"] not in completed_tasks and task['process'].poll() is not None:
                if task['process'].poll() != 0:
                    logger.warning(f'----> Task {task["task_id"]}/{number_of_tasks} failed!. <----')
                    failed_tasks[task["task_id"]] = tasks[task["task_id"] - 1]
                else:
                    logger.warning(f'----> Task {task["task_id"]}/{number_of_tasks} completed!. <----')
                
                completed_tasks.add(task['task_id'])
                task['stdout'].close()
                number_of_running_tasks -= 1
        
        # If the task id is bigger than the the number of tasks, quit the loop.
        if all_task_executed and len(completed_tasks) == number_of_tasks:
            break

        time.sleep(1/monitor_frequency)

    return failed_tasks


def monitor_and_automatic_run_tasks_on_gpu(tasks, available_gpus, num_task_parallel, stdout_dir):        
    gpu_pool = set(available_gpus)
    number_of_gpus = len(gpu_pool)
    number_of_tasks = len(tasks)

    def run_task(task, task_id, gpu_id):
        task_list = task.split(' ') + ['--cuda', '--cuda_device', f'{gpu_id}']

        logger.warning(f'----> Task {task_id}/{number_of_tasks} started. <----')
        logger.info(f'Command of task {task_id}/{number_of_tasks}: {" ".join(task_list)}')
        f_log = open(os.path.join(stdout_dir, f'stdout_log_{task_id}.txt'), 'w')
        process = subprocess.Popen(task_list, stdout = f_log, stderr = f_log, universal_newlines = True)

        return process, f_log

    task_id = 1
    running_tasks = {}
    number_of_running_tasks = 0
    all_task_executed = False
    completed_tasks = set()
    failed_tasks = {}

    while True:
        if task_id > number_of_tasks:
            all_task_executed = True

        if len(gpu_pool) != 0 and number_of_running_tasks < num_task_parallel and not all_task_executed:
            available_gpu = gpu_pool.pop()
            process, log_file = run_task(tasks[task_id - 1], task_id, available_gpu)
            running_tasks[available_gpu] = {'task_id': task_id, 'process': process, 'stdout': log_file}
            task_id += 1
            number_of_running_tasks += 1

        # Check if one task has finished. If so, do some housekeeping 
        # and add the allocated gpu_id back to the gpu_pool, marking this GPU is now free.
        for gpu_id, task in running_tasks.items():
            if task["task_id"] not in completed_tasks and task['process'].poll() is not None:
                if task['process'].poll() != 0:
                    logger.warning(f'----> Task {task["task_id"]}/{number_of_tasks} failed!. <----')
                    failed_tasks[task["task_id"]] = tasks[task["task_id"] - 1]
                else:
                    logger.warning(f'----> Task {task["task_id"]}/{number_of_tasks} completed!. <----')
                
                completed_tasks.add(task["task_id"])
                task['stdout'].close()
                gpu_pool.add(gpu_id)
                number_of_running_tasks -= 1
        
        # If all GPUs are free again and the task id is bigger than the the number of tasks, quit the loop.
        if len(gpu_pool) == number_of_gpus and all_task_executed:
            break

        time.sleep(1/monitor_frequency)
    
    return failed_tasks