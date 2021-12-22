from torch.utils.data import DataLoader
from ..utils import getLogger, read_json
import torch
import numpy as np
import random
import os

logger = getLogger(__name__)

def find_dataset(name, rank):
    try:
        dataloader_combo = dataloader_zoo[name]()
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

def prepare_dataloaders(opt, rank = 0, train = True, test = True, evaluate = True):
    file_names = os.listdir(opt.data_path)
    dataloader_config_dict = read_json(opt.abs_dataloader_config) if opt.abs_dataloader_config else {}

    logger.info(f"Additional dataloader settings from config files: {dataloader_config_dict}")

    dataset, read_data = find_dataset(opt.dataloader_name, rank)
    data_raw = read_data(opt.data_path, file_names)

    #========= Preparing dataloaders =========#
    train = dataset(data_raw['train'], device = opt.device, **dataloader_config_dict)
    evaluate = dataset(data_raw['evaluate'], device = opt.device, **dataloader_config_dict)
    test = dataset(data_raw['test'], device = opt.device, **dataloader_config_dict)
    train_iterator, evaluation_iterator, test_iterator = None, None, None
    g = torch.Generator()
    g.manual_seed(opt.seed + rank)

    if train:
        train_iterator = DataLoader(train, shuffle = True, batch_size=opt.batch_size, \
            num_workers=opt.n_worker, worker_init_fn = seed_worker, generator = g, pin_memory = False)
    if evaluate:
        evaluation_iterator = DataLoader(evaluate, batch_size=opt.batch_size, \
            num_workers=opt.n_worker, worker_init_fn = seed_worker, generator = g, pin_memory = False)
    if test:
        test_iterator = DataLoader(test, batch_size=opt.batch_size, \
            num_workers=opt.n_worker, worker_init_fn = seed_worker, generator = g, pin_memory = False)

    return train_iterator, evaluation_iterator, test_iterator

def syn_dataloader():
    '''
    Synthetic dataloader for all synthetic datasets.
    '''
    from .synthetic import synthetic_dataset
    return [synthetic_dataset.SynDataset, synthetic_dataset.read_data]

def ctlstm_dataloader():
    '''
    Dataloader for ctlstm mainly. Perhaps, it can be used against another suitable models.
    '''
    from .ctlstm import ctlstm_dataset as ctdata
    return [ctdata.CTLSTMDataset, ctdata.read_data]

def cnf_dataloader():
    '''
    Dataloader for cnf-based models.
    '''
    from .cnf import cnf_dataset as cnf
    return [cnf.CNFDataset, cnf.read_data]

def ifl_dataloader():
    '''
    Dataloader for IFL model.
    '''
    from .ifl import ifl_dataset as ifl
    return [ifl.IflDataset, ifl.read_data]
    

dataloader_zoo = {
    # These following dataloaders fits all other models.
    'syn': syn_dataloader,
    'ctlstm': ctlstm_dataloader,
    'cnf': cnf_dataloader,
    'ifl': ifl_dataloader

    # This dataloader fits all legacy models.
    # 2021-10-14 update: The same as legacy models, legacy dataloaders are deprecated now.
    # Take your own risk to readd and use them.
}