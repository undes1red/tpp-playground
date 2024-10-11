import copy, math, subprocess, time, os
from src.taskhost import get_logger


logger = get_logger(__name__)

slurm_kwargs = {
    'slurm_partition': 'SCT',
    'slurm_job_name': 'slurm_task',
    'slurm_cpus_per_task': 8,
    'slurm_time': 1400,
    'slurm_mem': '32GB',
    'slurm_gres': 'gpu:1',
    'slurm_qos': 'normal'
}

monitor_frequency = 10
def monitor_and_automaticly_run_tasks(tasks, use_gpu, available_gpus, num_task_parallel, stdout_dir, use_slurm):  
    if use_slurm:
        if use_gpu:
            return monitor_and_automaticly_run_tasks_on_slurm_gpu_node(tasks, available_gpus, num_task_parallel, stdout_dir)
        else:
            return monitor_and_automaticly_run_tasks_on_slurm_cpu_node(tasks, num_task_parallel, stdout_dir)
    else:
        if use_gpu:
            return monitor_and_automaticly_run_tasks_on_gpu(tasks, available_gpus, num_task_parallel, stdout_dir)
        else:
            return monitor_and_automaticly_run_tasks_on_cpu(tasks, num_task_parallel, stdout_dir)


def task_generator_worker(hyperparameter_list, iterate_style):
    '''
    [
        (other single hyperparameters),
        "counting_style": 
        {
            (hyperparameter lists)
        },
        "zip_style":
        {
            (hyperparameter lists)
        }
    ]
    '''
    if hyperparameter_list == {}:
        return [{}], 0

    static_hyperparameters = hyperparameter_list.get('static', {})
    tasks_hyperparameters = hyperparameter_list.get('tasks', {})
    zip_style_hyperparameters = hyperparameter_list.get('zip_style', {})
    counting_style_hyperparameters = hyperparameter_list.get('counting_style', {})

    tasks_hyperparameters = hyperparameter_parser[iterate_style](tasks_hyperparameters)
    number_of_tasks = len(tasks_hyperparameters)
    zip_style_hyperparameters_list, zip_style_the_number_of_task \
        = task_generator_worker(zip_style_hyperparameters, iterate_style = 'zip_style')
    counting_style_hyperparameters_list, counting_style_the_number_of_task \
        = task_generator_worker(counting_style_hyperparameters, iterate_style = 'counting_style')
    
    '''
    Ensure that all hyperparameter lists ready for index enumeration have the same length.
    Specifically used when iterate_style == 'index'
    '''
    tmp_length = 0
    if iterate_style == 'zip_style':
        tmp_length = max(number_of_tasks, zip_style_the_number_of_task, counting_style_the_number_of_task)
        number_of_tasks_for_comp = number_of_tasks if number_of_tasks > 0 else tmp_length
        zip_style_the_number_of_task_for_comp = zip_style_the_number_of_task if zip_style_the_number_of_task > 0 else tmp_length
        counting_style_the_number_of_task_for_comp = counting_style_the_number_of_task if counting_style_the_number_of_task > 0 else tmp_length
        assert number_of_tasks_for_comp == zip_style_the_number_of_task_for_comp == counting_style_the_number_of_task_for_comp

    generated_hyperparameters = []
    if iterate_style == 'zip_style':
        packed_data = zip(tasks_hyperparameters if number_of_tasks > 0 else [{} for _ in range(tmp_length)],
                          zip_style_hyperparameters_list if zip_style_the_number_of_task > 0 else [{} for _ in range(tmp_length)], 
                          counting_style_hyperparameters_list if counting_style_the_number_of_task > 0 else [{} for _ in range(tmp_length)])
        for packed_parameter_dicts in packed_data:
            parameter_buffer = copy.deepcopy(static_hyperparameters)
            for parameter_dict in packed_parameter_dicts:
                parameter_buffer.update(parameter_dict)
            generated_hyperparameters.append(parameter_buffer)
    elif iterate_style == 'counting_style':
        for tasks_hyperparameter in tasks_hyperparameters:
            for zip_style_hyperparameter in zip_style_hyperparameters_list:
                for counting_style_hyperparameter in counting_style_hyperparameters_list:
                    parameter_buffer = copy.deepcopy(static_hyperparameters)
                    parameter_buffer.update(tasks_hyperparameter)
                    parameter_buffer.update(zip_style_hyperparameter)
                    parameter_buffer.update(counting_style_hyperparameter)

                    generated_hyperparameters.append(parameter_buffer)
    else:
        raise KeyError('Unknown iterate style.')
        
    return generated_hyperparameters, len(generated_hyperparameters)


def task_zip_style_generator(hyperparameter_list):
    '''
    Affirm that all hyperparameter lists have the same length.
    '''
    expected_task_num = len(next(iter(hyperparameter_list.values())))
    for hyperparameters in hyperparameter_list.values():
        assert len(hyperparameters) == expected_task_num, "Zip style requires that all parameter lists must have the same length!"
    
    '''
    Generate the hyperparameter list.
    '''
    generated_hyperparameter_combinations = []
    for index in range(expected_task_num):
        generated_hyperparameter_combination = {key: item[index] for (key, item) in hyperparameter_list.items()}
        generated_hyperparameter_combinations.append(generated_hyperparameter_combination)
        
    return generated_hyperparameter_combinations
        

def task_counting_style_generator(hyperparameter_list):
    number_of_digits = len(hyperparameter_list.values())
    max_val_for_each_digit = [len(item) for item in hyperparameter_list.values()]
    current_index_of_each_list = [0] * number_of_digits
    number_of_combinations = math.prod(max_val_for_each_digit)

    '''
    No tasks defined
    '''
    if max_val_for_each_digit == []:
        return [{}]
    else:
        generated_hyperparameter_combinations = []
        for _ in range(number_of_combinations):
            generated_hyperparameter_combination \
                = {key: item[index] for (key, item), index in zip(hyperparameter_list.items(), current_index_of_each_list)}
            generated_hyperparameter_combinations.append(generated_hyperparameter_combination)

            current_index_of_each_list[0] += 1
            add_mark = False
            for idx, (current_index, max_unreachable_index) in enumerate(zip(current_index_of_each_list, max_val_for_each_digit)):
                if add_mark:
                    current_index_of_each_list[idx] += 1
                    if current_index_of_each_list[idx]  >= max_unreachable_index:
                        current_index_of_each_list[idx] = 0
                        add_mark = True
                    else:
                        add_mark = False
                if current_index >= max_unreachable_index:
                    current_index_of_each_list[idx] = 0
                    add_mark = True
        
        return generated_hyperparameter_combinations


hyperparameter_parser = {
    'zip_style': task_zip_style_generator,
    'counting_style': task_counting_style_generator
}


def translate_dict_to_arguments(input_dict):
    output_arguments = []
    for key, value in input_dict.items():
        if isinstance(value, bool):
            if value:
                output_arguments.append('--' + str(key))
        else:
            output_arguments.extend(['--' + str(key), str(value)])
    
    return output_arguments


def monitor_and_automaticly_run_tasks_on_cpu(tasks, num_task_parallel, stdout_dir):
    number_of_tasks = len(tasks)

    def run_task(task, task_id):
        task_list = task.split(' ')

        # Replace this command with your actual task command
        logger.warning(f'----> Task No.{task_id}/{number_of_tasks} started. <----')
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
            command = tasks[task_id - 1]
            process, log_file = run_task(command, task_id)
            running_tasks.append({'task_id': task_id, 'command': command, 'process': process, 'stdout': log_file})
            task_id += 1
            number_of_running_tasks += 1

        # Check if one task has finished. If so, do some housekeeping 
        # and add the allocated gpu_id back to the gpu_pool, marking this GPU is now free.
        for task in running_tasks:
            if task["task_id"] not in completed_tasks and task['process'].poll() is not None:
                if task['process'].poll() != 0:
                    logger.warning(f'----> Task No.{task["task_id"]}/{number_of_tasks} failed!. <----')
                    failed_tasks[task["task_id"]] = task["command"]
                else:
                    logger.warning(f'----> Task No.{task["task_id"]}/{number_of_tasks} completed!. <----')
                
                completed_tasks.add(task['task_id'])
                task['stdout'].close()
                number_of_running_tasks -= 1
        
        # If the task id is bigger than the the number of tasks, quit the loop.
        if all_task_executed and len(completed_tasks) == number_of_tasks:
            break

        time.sleep(1/monitor_frequency)

    return failed_tasks


def monitor_and_automaticly_run_tasks_on_gpu(tasks, available_gpus, num_task_parallel, stdout_dir):        
    gpu_pool = set(available_gpus)
    ticket_pool = set(range(num_task_parallel))
    number_of_gpus = len(gpu_pool)
    number_of_tasks = len(tasks)

    def run_task(task, task_id, gpu_id):
        task_list = task.split(' ') + ['--cuda', '--cuda_device', f'{gpu_id}']

        logger.warning(f'----> Task No.{task_id}/{number_of_tasks} started. <----')
        logger.info(f'Command of task {task_id}/{number_of_tasks}: {" ".join(task_list)}')
        f_log = open(os.path.join(stdout_dir, f'stdout_log_{task_id}.txt'), 'w')
        process = subprocess.Popen(task_list, stdout = f_log, stderr = f_log, universal_newlines = True)

        return process, f_log

    unique_task_id = 1
    running_tasks = {}
    all_task_executed = False
    failed_tasks = {}

    while True:
        if unique_task_id > number_of_tasks:
            all_task_executed = True

        if len(gpu_pool) != 0 and len(ticket_pool) != 0 and not all_task_executed:
            available_gpu = gpu_pool.pop()
            ticket = ticket_pool.pop()
            command = tasks[unique_task_id - 1]
            process, log_file = run_task(command, unique_task_id, available_gpu)
            running_tasks[ticket] = {'task_id': unique_task_id, 'gpu_id': available_gpu, 'command': command, 'process': process, 'stdout': log_file}
            unique_task_id += 1

        # Check if one task has finished. If so, get the result and do some housekeeping 
        # Add the allocated gpu_id and ticket back to the gpu_pool and ticket pool, saying we can start a new task if the GPU resources are sufficient.
        for ticket, task in running_tasks.items():
            if task != {} and task['process'].poll() is not None:
                if task['process'].poll() != 0:
                    logger.warning(f'----> Task No.{task["task_id"]}/{number_of_tasks} failed!. <----')
                    failed_tasks[task["task_id"]] = task['command']
                else:
                    logger.warning(f'----> Task No.{task["task_id"]}/{number_of_tasks} completed!. <----')
                
                task['stdout'].close()
                gpu_pool.add(task["gpu_id"])
                ticket_pool.add(ticket)
                running_tasks[ticket] = {}
                
        # If all GPUs are free again and the task id is bigger than the the number of tasks, quit the loop.
        if len(gpu_pool) == number_of_gpus and all_task_executed:
            break

        time.sleep(1/monitor_frequency)
    
    return failed_tasks


def monitor_and_automaticly_run_tasks_on_slurm_cpu_node(tasks, num_task_parallel, stdout_dir):
    import submitit
    
    number_of_tasks = len(tasks)

    def run_task(task, task_id):
        task_list = task.split(' ')

        # Replace this command with your actual task command
        logger.warning(f'----> Task No.{task_id}/{number_of_tasks} started. <----')
        logger.info(f'Command of task {task_id}/{number_of_tasks}: {task}')

        executor = submitit.AutoExecutor(folder = os.path.join(stdout_dir, str(task_id)))
        executor.update_parameters(**slurm_kwargs)
        function = submitit.helpers.CommandFunction(task_list)
        job = executor.submit(function)

        return job


    task_id = 1
    running_tasks = []
    number_of_running_tasks = 0
    completed_tasks = set()
    all_task_executed = False
    failed_tasks = {}

    logger.warning(f'Tasks submitted to slurm are out of our control. We can check if one job has finished, while automatically detecting if a task errored out is unreliable.')
    logger.warning(f'You have to check the result and affirm failed tasks by yourself.')

    while True:
        if task_id > number_of_tasks:
            all_task_executed = True

        if number_of_running_tasks < num_task_parallel and not all_task_executed:
            command = tasks[task_id - 1]
            job = run_task(command, task_id)
            running_tasks.append({'task_id': task_id, 'command': command, 'job': job, 'slurm_id': job.job_id})
            task_id += 1
            number_of_running_tasks += 1

        # Check if one task has finished. If so, do some housekeeping 
        # and add the allocated gpu_id back to the gpu_pool, marking this GPU is now free.
        for task in running_tasks:
            if task["task_id"] not in completed_tasks and task['job'].done() == True:
                if 'Submitted job triggered an exception' in task['job'].stderr():
                    logger.warning(f'----> Task No.{task["task_id"]}/{number_of_tasks} failed!. <----')
                    failed_tasks[task["task_id"]] = task["command"]
                else:
                    logger.warning(f'----> Task No.{task["task_id"]}/{number_of_tasks} completed!. <----')
                
                completed_tasks.add(task['task_id'])
                number_of_running_tasks -= 1
        
        # If the task id is bigger than the the number of tasks, quit the loop.
        if all_task_executed and len(completed_tasks) == number_of_tasks:
            break

        time.sleep(1/monitor_frequency)

    return failed_tasks



def monitor_and_automaticly_run_tasks_on_slurm_gpu_node(tasks, available_gpus, num_task_parallel, stdout_dir):        
    # I don't quite know how the GPU allocation works in slurm.
    # Due to this, we temporarily disable gpu_pool in this function.
    gpu_pool = list(available_gpus)
    ticket_pool = set(range(num_task_parallel))
    number_of_gpus = len(gpu_pool)
    number_of_tasks = len(tasks)

    def run_task(task, task_id, gpu_id):
        task_list = task.split(' ') + ['--cuda', '--cuda_device', f'{gpu_id}']

        logger.warning(f'----> Task No.{task_id}/{number_of_tasks} started. <----')
        logger.info(f'Command of task {task_id}/{number_of_tasks}: {" ".join(task_list)}')
        executor = submitit.AutoExecutor(folder = os.path.join(stdout_dir, str(task_id)))
        executor.update_parameters(**slurm_kwargs)
        function = submitit.helpers.CommandFunction(task_list)
        job = executor.submit(function)

        return job

    unique_task_id = 1
    running_tasks = {}
    all_task_executed = False
    failed_tasks = {}
    logger.warning(f'Tasks submitted to slurm are out of our control. We can check if one job has finished, while automatically detecting if a task errored out is unreliable.')
    logger.warning(f'You have to check the result and affirm failed tasks by yourself.')

    while True:
        if unique_task_id > number_of_tasks:
            all_task_executed = True

        if len(gpu_pool) != 0 and len(ticket_pool) != 0 and not all_task_executed:
            available_gpu = gpu_pool.pop()
            ticket = ticket_pool.pop()
            command = tasks[unique_task_id - 1]
            job = run_task(command, unique_task_id, available_gpu)
            running_tasks[ticket] = {'task_id': unique_task_id, 'gpu_id': available_gpu, 'command': command, 'job': job, 'slurm_id': job.job_id}
            unique_task_id += 1

        # Check if one task has finished. If so, get the result and do some housekeeping 
        # Add the allocated gpu_id and ticket back to the gpu_pool and ticket pool, saying we can start a new task if the GPU resources are sufficient.
        for ticket, task in running_tasks.items():
            if task != {} and task['job'].done() == True:
                if 'Submitted job triggered an exception' in task['job'].stderr():
                    logger.warning(f'----> Task No.{task["task_id"]}/{number_of_tasks} failed!. <----')
                    failed_tasks[task["task_id"]] = task['command']
                else:
                    logger.warning(f'----> Task No.{task["task_id"]}/{number_of_tasks} completed!. <----')
                
                gpu_pool.append(task["gpu_id"])
                # gpu_pool.add(task["gpu_id"])
                ticket_pool.add(ticket)
                running_tasks[ticket] = {}
                
        # If all GPUs are free again and the task id is bigger than the the number of tasks, quit the loop.
        if len(gpu_pool) == number_of_gpus and all_task_executed:
            break

        time.sleep(1/monitor_frequency)
    
    return failed_tasks