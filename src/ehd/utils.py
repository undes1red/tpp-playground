
# Help construct the output dir name using model hyperparameters.
def suffix(opt, *args):
    shortcut_dict = {
        'model_name': '',
        'used_model_name': '',
        'lr': 'lr',
        'used_lr': 'lr',
        'training_batch_size': 'bs',
        'used_batch_size': 'bs',
        'used_training_batch_size': 'bs',
        'n_training_steps': 'nts',
        'used_n_training_steps': 'nts',
        'dataloader_config': '',
        'used_dataloader_config': '',
        'model_config': '',
        'used_model_config': ''
    }
    
    output = []
    for item in args:
        hyperparameter = getattr(opt, item)
        translated_suffix = shortcut_dict[item] + str(hyperparameter)
        output.append(translated_suffix)
    
    output = "_".join(output)
    
    return output


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