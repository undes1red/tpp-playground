from torch.utils.data import DataLoader
from ..utils import getLogger
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
        dataset_combo = dataloader_zoo[name]
        if rank == 0:
            logger.info(f"Dataloader named {name} is retrieved.")
        return dataset_combo
    except:
        if rank == 0:
            logger.exception(f"Dataloader named {name} is not found! Please register your dataset in src/data/__init__.py and try again.")


def prepare_dataloaders(opt, rank = 0, train = True, test = True, evaluate = True):
    batch_size = opt.batch_size
    file_names = os.listdir(os.path.expanduser(opt.data_path))

    dataset, read_data = find_dataset(opt.dataloader_name, rank)
    data_raw = read_data(opt.data_path, file_names)

    #========= Preparing Model =========#
    train = dataset(data_raw['train'], device = opt.device)
    evaluate = dataset(data_raw['evaluate'], device = opt.device)
    test = dataset(data_raw['test'], device = opt.device)
    train_iterator, evaluation_iterator, test_iterator = None, None, None

    if train:
        train_iterator = DataLoader(train, shuffle = True, batch_size=batch_size, num_workers=opt.n_worker)
    if evaluate:
        evaluation_iterator = DataLoader(evaluate, batch_size=batch_size, num_workers=opt.n_worker)
    if test:
        test_iterator = DataLoader(test, batch_size=batch_size, num_workers=opt.n_worker)

    return train_iterator, evaluation_iterator, test_iterator