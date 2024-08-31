import os

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