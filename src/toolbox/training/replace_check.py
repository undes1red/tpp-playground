import os
from pathlib import Path


def replace_check(opt, idx, **subdirs_and_marker):
    '''
    This function must ensure we use the same index in all subdirs.
    Throw an exception if indexes calculated in each dirs do not match.
    '''
    calculated_indexes = []
    continue_running = True

    for (subdir, marker_file) in subdirs_and_marker.items():
        leaf_dir_name = f'{subdir}_' + idx
        tmp_path = Path(opt.root_path, subdir, opt.procedure)
        if not tmp_path.exists():
            # Return 1 if the tmp_path does not exist.
            calculated_indexes.append(1)
            continue

        files = os.scandir(tmp_path)
        valid_dir_names = [int(dir_item.name) for dir_item in filter(lambda x: not x.is_file() and x.name.isdigit(), files)]
        valid_dir_names = sorted(valid_dir_names)

        index = 1
        for vaild_dir_name in valid_dir_names:
            vaild_dir = tmp_path / str(vaild_dir_name) / opt.dataset_name / leaf_dir_name / marker_file
            if vaild_dir.exists():
                index += 1
            else:
                break

        calculated_indexes.append(index)

    baseline = calculated_indexes[0]
    for index in calculated_indexes:
        if index != baseline:
            raise ValueError(f'We get different task indexes in {list(subdirs_and_marker.keys())}!')

    if baseline > opt.maximum_retrain:
        continue_running = False

    return continue_running, str(baseline)
