import os

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