import os


def mkdir_if_not_exist(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    return 0