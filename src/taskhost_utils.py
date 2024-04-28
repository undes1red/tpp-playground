import logging
import sys
import os
import operator
import re
import importlib
import pickle as pkl

from packaging.version import Version


def get_logger(name = None, root = True):
    '''
    Get normal loggers or file loggers.

    Args:
    name: The name of a generated logger
    file: print all logs into the file if set.
    '''

    logger = logging.getLogger(name)
    if root:
        logger.parent = None
        logger.root = logger

    logger.setLevel(logging.INFO)
    if (logger.hasHandlers()):
        logger.handlers.clear()
    # create console handler and set level to debug
    ch = logging.StreamHandler(sys.stdout)
    # create formatter
    formatter = logging.Formatter('%(asctime)s [%(filename)s:%(lineno)d]: %(message)s', datefmt = '%Y-%m-%d %H:%M:%S')
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)

    return logger


def mkdir_if_not_exist(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    return 0


operator_dict = {
    '>': operator.gt,
    '<': operator.lt,
    '<=': operator.le,
    '==': operator.eq,
    '>=': operator.ge,
    '!=': operator.ne
}


def version_check(version, criteria):
    version = Version(version)
    criteria = criteria.split(',')
    print_warning = True

    for sub_criterion in criteria:
        sub_criterion = sub_criterion.strip()
        compare_label = re.match(r'^[!<>=]=?', sub_criterion).group(0)
        target_pytorch_version = Version(sub_criterion[len(compare_label):].strip())

        if not operator_dict[compare_label](version, target_pytorch_version):
            print_warning = False
        
    return print_warning


def dump_to_pkl(data, filepath, compression = None):
    dict_compression_algorithms = {
        None: open,
        # Is it a good choice?
        'lzma': importlib.import_module('lzma').open,
        'bz2': importlib.import_module('bz2').open,
        'gz': importlib.import_module('gzip').open
    }
    '''
    Add proper suffix to the base file name if compression is not None.
    '''
    head, tail = os.path.split(filepath)
    tail = tail + f'{"." + compression if compression is not None else ""}'
    filepath = os.path.join(head, tail)

    selected_open_function = dict_compression_algorithms[compression]
    f = selected_open_function(filepath, 'wb')
    pkl.dump(data, f)
    f.close()

    return 0


def write_to_txt(strings, filepath):
    f = open(filepath, 'w')
    
    if isinstance(strings, list):
        f.writelines(strings)
    else:
        f.write(strings)

    f.close()

    return 0