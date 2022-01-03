from torch.utils.data import DataLoader
from ..utils import getLogger, read_json
import torch, importlib, random, os
import numpy as np


logger = getLogger(__name__)

def find_dataset(name, rank):
    try:
        dataloader_combo = dataloader_zoo(name)
        if rank == 0:
            logger.info(f"Dataloader named {name} is retrieved.")
        return dataloader_combo
    except:
        if rank == 0:
            logger.exception(f"Dataloader named {name} is not found! Please register your dataset in src/data/__init__.py and try again.")

# Referring to https://pytorch.org/docs/stable/notes/randomness.html#reproducibility
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class move_data_to_the_correct_device:
    def __init__(self, device):
        self.device = device
    
    '''
    These following two functions try to mimic the official collate_fn().
    '''
    def __call__(self, data):
        from torch.utils.data._utils.collate import default_collate
        data_in_cpu = default_collate(data)

        data_in_correct_location = []
        for item in data_in_cpu:
            data_in_correct_location.append(item.to(self.device))
        return data_in_correct_location

def prepare_dataloaders(opt, rank = 0, train = True, test = True, evaluate = True, preload = False):
    file_names = os.listdir(opt.data_path)
    dataloader_config_dict = read_json(opt.abs_dataloader_config) if opt.abs_dataloader_config else {}

    logger.info(f"Additional dataloader settings from config files: {dataloader_config_dict}")

    dataset, read_data = find_dataset(opt.dataloader_name, rank)
    data_raw = read_data(opt.data_path, file_names)

    #========= Preparing dataloaders =========#
    train = dataset(data_raw['train'], device = opt.device, **dataloader_config_dict)
    evaluate = dataset(data_raw['evaluate'], device = opt.device, **dataloader_config_dict)
    test = dataset(data_raw['test'], device = opt.device, **dataloader_config_dict)
    device_locator = move_data_to_the_correct_device(device = opt.device)

    train_iterator, evaluation_iterator, test_iterator = None, None, None
    g = torch.Generator()
    g.manual_seed(opt.seed + rank)

    if train:
        train_iterator = DataLoader(train, shuffle = True, batch_size=opt.batch_size, \
            collate_fn = device_locator if preload else None, \
            num_workers=opt.n_worker, worker_init_fn = seed_worker, generator = g, pin_memory = False)
    if evaluate:
        evaluation_iterator = DataLoader(evaluate, batch_size=opt.batch_size, \
            collate_fn = device_locator if preload else None, \
            num_workers=opt.n_worker, worker_init_fn = seed_worker, generator = g, pin_memory = False)
    if test:
        test_iterator = DataLoader(test, batch_size=opt.batch_size, \
            collate_fn = device_locator if preload else None, \
            num_workers=opt.n_worker, worker_init_fn = seed_worker, generator = g, pin_memory = False)

    return train_iterator, evaluation_iterator, test_iterator


dataloader_modulepath = {
    # These following dataloaders fits all other models.
    'syn': ['synthetic.synthetic_dataset', 'syn_dataloader'],
    'ctlstm': ['ctlstm.ctlstm_dataset', 'ctlstm_dataloader'],
    'cnf': ['cnf.cnf_dataset', 'cnf_dataloader'],
    'ifl': ['ifl.ifl_dataset', 'ifl_dataloader']

    # 2021-10-14 update: legacy dataloaders are deprecated and are removed from the dataloader zoo.
    # Take your own risk to readd and use them.
}

def dataloader_zoo(name):
    path, function_name = dataloader_modulepath[name]
    module = importlib.import_module('.' + path, package = 'src.TPP.dataloader')
    return getattr(module, function_name)()