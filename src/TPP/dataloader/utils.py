import torch, random
import numpy as np


# Referring to https://pytorch.org/docs/stable/notes/randomness.html#reproducibility
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def check_exist(file_list, file_type, *file_names):
    existing_files = []
    for file_name in file_names:
        if file_name is None:
            continue

        complete_file_name = f'{file_name}.{file_type}'
        if complete_file_name not in file_list:
            return FileExistsError(f'{file_name} is not found!')
        
        existing_files.append(complete_file_name)
    
    return existing_files