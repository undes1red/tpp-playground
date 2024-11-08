
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


# Help construct the output dir name using model hyperparameters.
def suffix(opt, *args):
    output = []
    for item in args:
        hyperparameter = getattr(opt, item)
        translated_suffix = suffix_shortcut_dict[item] + str(hyperparameter)
        output.append(translated_suffix)
    
    output = "_".join(output)
    
    return output