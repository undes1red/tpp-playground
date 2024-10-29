def load_from_pkl(filepath, compression = None):
    import importlib
    import pickle as pkl
    
    dict_compression_algorithms = {
        None: open,
        # Is it a good choice?
        'lzma': importlib.import_module('lzma').open,
        'bz2': importlib.import_module('bz2').open,
        'gz': importlib.import_module('gzip').open
    }

    selected_open_function = dict_compression_algorithms[compression]
    f = selected_open_function(filepath, 'rb')
    data = pkl.load(f)
    f.close()

    return data


def load_from_npz(filepath, **kwargs):
    import numpy as np

    '''
    Add proper suffix to the base file name if compression is not None.
    '''
    np.savez(filepath, **kwargs)

    return 0