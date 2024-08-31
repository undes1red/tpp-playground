# Several extensive operations for python list.
import os
from tqdm import tqdm
import torch

from src.toolbox.list_operation import list_add, list_div


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


# General evaluation procedure.
def get_evaluation_results(data, model, model_class, device, output_length, desc):
    sum_ = [0] * output_length
    dataset_size = len(data)
    
    for minibatch in tqdm(data, desc):
        batch_sum = model_class.evaluation_step(model, minibatch, device)
        sum_ = list_add(sum_, batch_sum)

    sum_ = list_div(sum_, dataset_size)
    return {'results': sum_}


# A more neat way to print hyperparameters:
def print_args(opt):
    output = '\nAll hyperparameters:\n'
    for key, value in opt.__dict__.items():
        output += str(key) + ': ' + str(value) + '\n'

    return output


def replace_check(opt, id, *subdirs):
    '''
    This function must ensure we use the same index in all subdirs.
    Throw an exception if indexes calculated in each dirs do not match.
    '''
    calculated_indexes = []
    
    for subdir in subdirs:
        leaf_dir_name = f'{subdir}_' + id
        tmp_path = os.path.join(opt.root_path, subdir, opt.procedure)
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


def possible_checkpoint_detect(opt, id):
    # We check if the checkpoint and related checkpoint.csv exist.
    # If checkpoint.csv exists, the training process should successfully complete, so checkpoint should exist.
    # If only the checkpoint, we might meet a runtime error during training.
    # We count runs that leaves a legit checkpoint.
    folder_name = 'model_' + id

    # Scan valid folders.
    tmp_path = os.path.join(opt.root_path, 'model', opt.procedure)
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
    model_raw = torch.load(checkpoint_dir, weights_only = False, map_location = device)
    model_state_dict = model_raw['model']
    model.load_state_dict(model_state_dict)
    if evaluation:
        model.requires_grad_(requires_grad = False)
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Model restore completed. The number of trainable parameters in this model: {trainable_parameters} out of {total_params}.')

    return model