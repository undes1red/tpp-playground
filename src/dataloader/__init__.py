from torch.utils.data import DataLoader
from ..utils import getLogger, read_json
import torch
import numpy as np
import random
import os

# Dataset registration
from .synthetic import synthetic_dataset
from .ctlstm import ctlstm_dataset as ctdata
from .cnf import cnf_dataset as cnf
from .ifl import ifl_dataset as ifl
from .synthetic_legacy import synthetic_dataset_legacy

logger = getLogger(__name__)

dataloader_zoo = {
    # This dataloader fits all legacy models.
    'syn_legacy': [synthetic_dataset_legacy.SynDataset_legacy, synthetic_dataset_legacy.read_data],

    # These following dataloaders fits all other models.
    'syn': [synthetic_dataset.SynDataset, synthetic_dataset.read_data],
    'ctlstm': [ctdata.CTLSTMDataset, ctdata.read_data],
    'cnf': [cnf.CNFDataset, cnf.read_data],
    'ifl': [ifl.IflDataset, ifl.read_data]
}

def find_dataset(name, rank):
    try:
        dataloader_combo = dataloader_zoo[name]
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
    dataloader_config_dict = read_json(opt.dataloader_config) if opt.dataloader_config else {}

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
            num_workers=opt.n_worker, worker_init_fn = seed_worker, generator = g, pin_memory = True)
    if evaluate:
        evaluation_iterator = DataLoader(evaluate, batch_size=opt.batch_size, \
            num_workers=opt.n_worker, worker_init_fn = seed_worker, generator = g, pin_memory = True)
    if test:
        test_iterator = DataLoader(test, batch_size=opt.batch_size, \
            num_workers=opt.n_worker, worker_init_fn = seed_worker, generator = g, pin_memory = True)

    return train_iterator, evaluation_iterator, test_iterator