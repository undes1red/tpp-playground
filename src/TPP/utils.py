# Several extensive operations for python list.
import math, yaml, os
from tqdm import tqdm
from functools import reduce
import torch


suffix_shortcut_dict = {
    'model_name': '',
    'lr': 'lr',
    'training_batch_size': 'bs',
    'used_batch_size': 'bs',
    'n_training_steps': 'nts',
    'dataloader_config': '',
    'used_dataloader_config': '',
    'model_config': ''
}


def add(a, b):
    return a + b


def mean(iter):
    return reduce(add, iter)/len(iter)


def lst_add_lst(list1, list2):
    return [sum(x) for x in zip(list1, list2)]


def lst_divide(lst, denominator):
    if isinstance(denominator, list):
        assert len(lst) == len(denominator)
        return [x/y for x, y in zip(lst, denominator)]
    return [x/denominator for x in lst]


# How to print formated logs via logger and format definitions.
def print_performances(logger, procedure, data_dict):
    info_string = ''
    for key, value in data_dict.items():
        sub_info = ' ,' + key + ': {' + value['num_format'] + '}' + value['suffix']
        info_string += sub_info.format(value['data'])
    
    info_string = f'{procedure:12}' + info_string
    logger.info(info_string)


def pack_one_value_to_dict(data, num_format = '6.5f', suffix = ''):
    return {'data': data, 'num_format': ':' + num_format, 'suffix': suffix}


def only_keep_data(dict_input):
    plain_results = {}
    for key, value in dict_input.items():
        plain_results[key] = value['data']

    return plain_results


# Read and convert a YAML file into a dict object.
def read_yaml(yaml_path):
    a = None
    with open(yaml_path, 'r') as f:
        try:
            a = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)

    return a


# Help construct the output dir name using model hyperparameters.
def suffix(opt, *args):
    output = []
    for item in args:
        hyperparameter = getattr(opt, item)
        translated_suffix = suffix_shortcut_dict[item] + str(hyperparameter)
        output.append(translated_suffix)
    
    output = "_".join(output)
    
    return output


# General evaluation procedure.
def get_evaluation_results(data, model, model_class, device, output_length, desc):
    sum_ = [0] * output_length
    dataset_size = len(data)
    
    for minibatch in tqdm(data, desc):
        batch_sum = model_class.evaluation_step(model, minibatch, device)
        sum_ = lst_add_lst(sum_, lst_divide(batch_sum, dataset_size))

    return {'results': sum_}


# extract dataset name from the input string
# eg: 'dataset_name_new_v2'
def restore_dataset_name(name):
    name = name.strip('v123456789')
    name = name[:-1]
    if name.endswith('_new'):
        name = name[:-4]
    if name.endswith('_continuous'):
        name = name[:-11]
    return name


class Metric():
    '''
    A Metric handler.
    1. metric_number: How many metric do you have?
    2. smaller_is_better: If model performance is better with lower metric value, you should set it to true. Otherwise, it is false.
    If smaller_is_better is set, its length must match argument 'metric_number'.
    '''
    def __init__(self, metric_number, smaller_is_better = None):
        self.metric_number = metric_number
        self.map = {True: 1, False: -1}
        self.best_metric = [math.inf] * self.metric_number
        if smaller_is_better is None:
            self.mask = [1] * self.metric_number
        else:
            assert len(smaller_is_better) == self.metric_number
            self.mask = [self.map[item] for item in smaller_is_better]
    

    def compare(self, input_metric):
        assert len(input_metric) == len(self.mask)
        tmp = lst_divide(input_metric, self.mask)
        output = True

        for input_number, recorded in zip(tmp, self.best_metric):
            if input_number > recorded:
                output = False
                break
        
        if output:
            self.best_metric = input_metric
        
        return output
    
    
    def show(self):
        return self.best_metric


# A more neat way to print hyperparameters:
def print_args(opt):
    output = '\nAll hyperparameters:\n'
    for key, value in opt.__dict__.items():
        output += str(key) + ': ' + str(value) + '\n'

    return output


def replace_check(opt, root_path, *subdirs):
    '''
    This function must ensure we use the same index in all subdirs.
    Throw an exception if indexes calculated in each dirs do not match.
    '''
    calculated_indexes = []
    folder_suffix = suffix(opt, 'model_name', 'lr', 'training_batch_size', 'n_training_steps', 'dataloader_config', 'model_config')
    
    for subdir in subdirs:
        leaf_dir_name = f'{subdir}_' + folder_suffix
        tmp_path = os.path.join(root_path, subdir, opt.procedure)
        if not os.path.exists(tmp_path):
            # Return 1 if the tmp_path does not exist.
            calculated_indexes.append(1)
            continue

        files = os.scandir(tmp_path)
        valid_dir_names = [int(dir_item.name) for dir_item in filter(lambda x: not x.is_file() and x.name.isdigit(), files)]
        valid_dir_names = sorted(valid_dir_names)
        
        index = 1
        for vaild_dir_name in valid_dir_names:
            vaild_dir = os.path.join(tmp_path, str(vaild_dir_name), opt.dataset_name, leaf_dir_name)
            if os.path.exists(vaild_dir):
                index += 1
            else:
                break
        
        calculated_indexes.append(index)
    
    baseline = calculated_indexes[0]
    for index in calculated_indexes:
        assert index == baseline, f'We get different task indexes in {subdirs}!'
    
    return str(baseline)


def possible_checkpoint_detect(opt, root_path):
    # We check if the checkpoint and related checkpoint.csv exist.
    # If checkpoint.csv exists, the training process should successfully complete, so checkpoint should exist.
    # If only the checkpoint, we might meet a runtime error during training.
    # We count runs that leaves a legit checkpoint.
    model_hyperparameters = suffix(opt, 'model_name', 'lr', 'used_batch_size', 'n_training_steps', 'used_dataloader_config', 'model_config')
    folder_name = 'model_' + model_hyperparameters

    # Scan valid folders.
    tmp_path = os.path.join(root_path, 'model', opt.procedure)
    files = os.scandir(tmp_path)
    possible_valid_dir_names = [int(dir_item.name) for dir_item in filter(lambda x: not x.is_file() and x.name.isdigit(), files)]
    valid_dir_indexes = []

    for possible_valid_dir_name in possible_valid_dir_names:
        possible_checkpoint = os.path.join(tmp_path, str(possible_valid_dir_name), opt.dataset_name, folder_name, 'checkpoint.chkpt')
        # possible_checkpoint_log = os.path.join(tmp_path, str(possible_valid_dir_name), opt.dataset_name, folder_name, 'checkpoint.csv')
        if os.path.exists(possible_checkpoint):
            valid_dir_indexes.append(possible_valid_dir_name)
    
    return valid_dir_indexes


def load_checkpoint(logger, checkpoint_dir, model, device, evaluation = True):
    '''
    Here, we need to 1. restore the model weights from the checkpoint, 2. convert it into a DP if possible.
    '''
    model_raw = torch.load(checkpoint_dir, weights_only = True, map_location = device)
    model_state_dict = model_raw['model']
    model.load_state_dict(model_state_dict)
    if evaluation:
        model.requires_grad_(requires_grad = False)
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Model restore completed. The number of trainable parameters in this model: {trainable_parameters} out of {total_params}.')

    return model