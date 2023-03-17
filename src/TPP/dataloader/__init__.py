import torch, os, importlib, glob

from torch.utils.data import DataLoader

from src.TPP.dataloader.utils import *
from src.TPP.utils import getLogger, read_json


logger = getLogger(__name__)


def dataloader_zoo(name):
    module = importlib.import_module('.' + name, package = 'src.TPP.dataloader')
    return module.get_dataloader()


def find_dataset(name, rank):
    try:
        dataloader_combo = dataloader_zoo(name)
        if rank == 0:
            logger.info(f"Dataloader named {name} is retrieved.")
        return dataloader_combo
    except:
        if rank == 0:
            logger.exception(f"Dataloader named {name} is not found! Please register your dataset in src/data/__init__.py and try again.")


def prepare_dataloaders(opt, rank = 0):
    file_names = [os.path.basename(item) for item in glob.glob(opt.data_path + f'/*.{opt.dataset_type}')]
    if len(file_names) == 0:
        logger.exception(f'No appropriate data file in {opt.data_path}! Please check the training script.')
    else:
        logger.info(f'There are {len(file_names)} appropriate files in {opt.data_path}. Format: {opt.dataset_type}. Is that expected?')
    
    dataloader_config_dict = read_json(opt.abs_dataloader_config) if opt.abs_dataloader_config else {}
    logger.info(f"Additional dataloader settings from config files: {dataloader_config_dict}")

    dataset, read_data = find_dataset(opt.dataloader_name, rank)
    data_raw = read_data(opt.data_path, file_names)
    try:
        with open(os.path.join(opt.data_path, 'num_events.txt'), 'r') as f:
            opt.num_events = int(f.read())
    except:
        '''
        Assume that no event information is available.
        '''
        opt.num_events = 1

    #========= Preparing dataloaders =========#
    train_dataset = dataset(data_raw['train'], device = opt.device, num_events = opt.num_events, **dataloader_config_dict)
    evaluate_dataset = dataset(data_raw['evaluate'], num_events = opt.num_events, device = opt.device, **dataloader_config_dict)
    test_dataset = dataset(data_raw['test'], num_events = opt.num_events, device = opt.device, **dataloader_config_dict)

    try:
        data_collator = getattr(train_dataset, '__call__')
    except:
        '''
        This data collator is for data evaluation.
        '''
        data_collator = move_data_to_the_correct_device(device = opt.device)


    train_iterator, evaluation_iterator, test_iterator = None, None, None
    g = torch.Generator()
    g.manual_seed(opt.seed + rank)

    if not hasattr(opt, 'train') or (hasattr(opt, 'train') and opt.train):
        train_iterator = DataLoader(train_dataset, shuffle = True, batch_size=opt.batch_size, \
            collate_fn = data_collator, num_workers=opt.n_worker, worker_init_fn = seed_worker,\
            generator = g, pin_memory = False)
    if not hasattr(opt, 'evaluation') or (hasattr(opt, 'evaluation') and opt.evaluation):
        evaluation_iterator = DataLoader(evaluate_dataset, batch_size=opt.batch_size, \
            collate_fn = data_collator, num_workers=opt.n_worker, worker_init_fn = seed_worker,\
            generator = g, pin_memory = False)
    if not hasattr(opt, 'test') or (hasattr(opt, 'test') and opt.test):
        test_iterator = DataLoader(test_dataset, batch_size=opt.batch_size, \
            collate_fn = data_collator, num_workers=opt.n_worker, worker_init_fn = seed_worker,\
            generator = g, pin_memory = False)

    return train_iterator, evaluation_iterator, test_iterator

