# You can use this file if you are too lazy to create and modify script files.
# Just pack numerous tasks and run them one by one automatically.

import subprocess, os, argparse, itertools, math
from src.traininghost import getLogger

logger = getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--script_type', type = str, choices = ['train', 'plot'], default = 'train',\
                                     help = 'You can use this only argument to select what you want to do.')
parser.add_argument('--GPU', type = int, default = None, help='How many GPU you want to use? Set it to None to use all GPUs, \
                                                                 or set it to negative number or None for CPU learning.')
opt = parser.parse_args()
# Environment variables
do_not_use_gpu = False
if opt.GPU is not None and opt.GPU >= 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.GPU)
else:
    do_not_use_gpu = True


training_hyperparameter_list = [
    "train.py", \
    "--no_seed", \
    "--dataloader_name", "syn", \
    "--dataloader_config", "retweet/no_norm.json", \
    "--dataset_name", ["mimic"], \
    "--n_training_steps", "50000", \
    "--n_evaluation_steps", "500", \
    "--n_report_steps", "500", \
    "--b", "16", \
    "--n_warmup_steps", "5000", \
    "--model_name", "fullynn", \
    "--model_config", ["normal/fullynn.json", "normal/fullynn_no_act.json", "normal/fullynn_no_act_expand.json",
                       "normal/fullynn_no_all.json", "normal/fullynn_no_expand.json", "normal/fullynn_no_inv.json", 
                       "normal/fullynn_no_norm_act.json", "normal/fullynn_no_norm_expand.json", "normal/fullynn_no_norm.json"],
    "--lr", "0.002", \
    "--save_mode", "best", \
    "--lr_sched", \
    "--op_name", "AdamW", \
    "--optim_json", "optimizer.json", \
    "" if do_not_use_gpu else "--cuda", \
    "--n_cycles", "0.5",\
    "--wandb"
]

plot_hyperparameter_list = [
    "graph.py", \
    "--seed", "32", \
    "--model_name", "fullynn", \
    "--model_config", ["normal/fullynn_no_norm_act.json", "normal/fullynn_no_norm.json", "normal/fullynn_no_norm_expend.json", \
                       "normal/fullynn_no_act.json", "normal/fullynn_no_act_expend.json", "normal/fullynn_no_all.json", \
                       "normal/fullynn_no_inv.json", "normal/fullynn_no_expand.json"], \
    "--lr", "0.002", \
    "--batch_size", "128", \
    "--n_training_steps", "50000", \
    "--dataset_name", ["hawkes_1_v2", "hawkes_2_v2", "hawkes_1_new_v1", "hawkes_2_new_v1", "self_correct_v2",\
                       "self_correct_new_v1", "poisson_new_v1", "poisson_v2"], \
    "--dataloader_name", "syn", \
    "--figure_count", "10", \
    "--train", \
    "--test", \
    "--evaluation", \
    # "--used_dataloader_config", "shift.json", \
    "--plot_type", ["intensity", "probability"], \
    "--dataloader_config", "normal/plot.json", \
    "--resolution", "100", \
    "" if do_not_use_gpu else "--cuda", \
]

hyperparameters_dict = {
    'train': training_hyperparameter_list,
    'plot': plot_hyperparameter_list
}

def list_generator(hyperparameter_list):
    '''
    Special used only
    '''
    head = hyperparameter_list[0]
    single_parameters = {}
    multiple_parameters = {}

    last_parameter = ''
    for items in hyperparameter_list[1:]:
        if last_parameter == '':
            '''
            new arguments:
            '''
            last_parameter = items
        elif last_parameter.startswith('--') and type(items) == list:
            '''
            arguments with multiple choices
            '''
            multiple_parameters[last_parameter] = items
            last_parameter = ''
        elif last_parameter.startswith('--') and items.startswith('--'):
            '''
            store_true arguments
            '''
            single_parameters[last_parameter] = ''
            last_parameter = items
        elif last_parameter.startswith('--') and not items.startswith('--'):
            '''
            arguments with single choice
            '''
            single_parameters[last_parameter] = items
            last_parameter = ''
    if last_parameter != '':
        single_parameters[last_parameter] = ''
    
    # Now, map all fixed argument into a list.
    fixed_arguments_part = [head] + list(itertools.chain.from_iterable(single_parameters.items()))

    # set iterators, the first iterator is always the single directed iterator. We use it to decide when we quit the argument
    # generation loop.
    multi_hp_count = len(multiple_parameters.values())
    count_of_each_multiple_hp = [len(item) for item in multiple_parameters.values()]
    current_index_of_each_list = [0] * multi_hp_count
    the_number_of_task = math.prod(count_of_each_multiple_hp)
    logger.info(f'Totally, {the_number_of_task} tasks are planned.')

    def remove_all(x, bad_item = ''):
        try:
            while 1:
                x.remove(bad_item)
        except:
            return x

    if count_of_each_multiple_hp == []:
        # No multiple hp is present.
        yield remove_all(fixed_arguments_part)
    else:
        for _ in range(the_number_of_task):
            choosed_value = {key: item[index] for (key, item), index in zip(multiple_parameters.items(), current_index_of_each_list)}
            choosed_value_to_list = list(itertools.chain.from_iterable(choosed_value.items()))
            final_list = fixed_arguments_part + choosed_value_to_list
            yield remove_all(final_list)

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

task_count = 1
for hp_list in list_generator(hyperparameters_dict[opt.script_type]):
    process = subprocess.Popen([
            'python3'
    ] + hp_list)
    process.wait()
    logger.info(f'Task {task_count} completed.')
    task_count += 1