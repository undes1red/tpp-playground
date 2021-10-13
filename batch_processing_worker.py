# You can use this file if you are too lazy to create and modify script files.
# Just pack numerous tasks and run them one by one automatically.

import subprocess, os, argparse, itertools, math

parser = argparse.ArgumentParser()
parser.add_argument('--script_type', type = 'str', choices = ['train', 'plot'], default = 'train',\
                                     help = 'You can use this only argument to select what you want to do.')
parser.add_argument('--GPU', type = 'int', default = None, help='How many GPU you want to use? Set it to None to use all GPUs, \
                                                                 or set it to negative number for CPU learning.')
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
    "--dataset_name", "hawkes_1", \
    "--n_training_steps", "100000", \
    "--n_evaluation_steps", "500", \
    "--n_report_steps", "500" \
    "--b", "128", \
    "--n_warmup_steps", "5000", \
    "--model_name", "dwg", \
    "--model_json", "dwg_master_restrict.json", \
    "--lr", "0.001", \
    "--save_mode", "best", \
    "--lr_sched", \
    "--op_name", "AdamW", \
    "--optim_json", "optimizer.json", \
    "" if do_not_use_gpu else "--cuda", \
    "--n_cycles", "2", \
    "--wandb"
]

plot_hyperparameter_list = [
    "graph.py", \
    "--seed", "42", \
    "--model_name", "ctlstm", \
    "--model_config", "ctlstm.json", \
    "--lr", "0.02", \
    "--batch_size", "16", \
    "--n_training_steps", "10000", \
    "--dataset_name", "hawkes_1", \
    "--dataloader_name", "ctlstm", \
    "--figure_count", "10", \
    "--train", \
    "--test", \
    "--evaluation", \
    "--plot_type", "intensity", \
    "--dataloader_config", "ctlstm_dl.json", \
    "--used_dataloader_config", "ctlstm_dl.json", \
    "--resolution", "16"
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
        elif last_parameter.startswith('--') and items.startswith('--'):
            '''
            store_true arguments
            '''
            single_parameters[last_parameter] = ''
            last_prarmeter = items
        elif last_parameter.startswith('--') and not items.startswith('--'):
            '''
            arguments with single choice
            '''
            single_parameters[last_parameter] = items
    
    # Now, generate all argument patterns.
    fixed_arguments_part = [head] + list(itertools.chain.from_iterable(single_parameters.items()))
    count_each_choices = [len(argument_choices) for argument_choices in multiple_parameters.values()]
    task_number = math.prod(count_each_choices)

    # set iterators, the first iterator is always the single directed iterator. We use it to decide when we quit the argument
    # generation loop.
    


task_count = 0

process = subprocess.Popen([
        "python3", hyperparameters_dict[opt.script_type]
        
])
process.wait()
print(f'Task {1} completed.')