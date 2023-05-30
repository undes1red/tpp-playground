import torch, os, importlib, glob

from torch.utils.data import DataLoader
from src.taskhost_utils import getLogger
from src.TPP.dataloader.utils import seed_worker
from src.TPP.utils import read_yaml


logger = getLogger(__name__)


def dataloader_zoo(name):
    module = importlib.import_module('.' + name, package = 'src.TPP.dataloader')
    return module.get_dataloader()


def find_dataset(name, rank):
    try:
        dataloader_combo = dataloader_zoo(name)
        if rank == 0:
            logger.info(f"Dataloader name: {name}")
        return dataloader_combo
    except:
        if rank == 0:
            logger.exception(f"Dataloader named {name} is not found! Please try again.")


def prepare_dataloaders(opt, rank = 0):
    '''
    Creates the required dataloader against custom dataloader settings.

    Args:
    * opt:  namespace
            This namespace stores all parsed arguments.
    * rank: int
            Says which GPU we should use.
    '''
    file_names = [os.path.basename(item) for item in glob.glob(opt.data_path + f'/*.{opt.dataset_type}')]

    if rank == 0:
        if len(file_names) == 0:
            logger.exception(f'No available dataset file in {opt.data_path}!')
        else:
            logger.info(f'We are going to read {len(file_names)} files in {opt.data_path}. They are all {opt.dataset_type} files. Is that right?')
    
    dataloader_config_dict = read_yaml(opt.abs_dataloader_config) if opt.abs_dataloader_config else {}

    if rank == 0:
        if opt.abs_dataloader_config is None:
            logger.info(f"No custom dataloader settings! We will use the default dataloader settings.")
        else:
            logger.info(f"Custom dataloader settings are loaded from this config file {opt.abs_dataloader_config}.")
            logger.info(f"Custom dataloader settings are: {dataloader_config_dict}.")

    dataset, read_data = find_dataset(opt.dataloader_name, rank)
    data_raw = read_data(opt.data_path, file_names)

    '''
    Now, dataset_card.yml is mandatory for every dataset.
    This YAML file should contain useful information about this dataset, like the number of classes it has.
    '''
    opt.info_dict = read_yaml(os.path.join(opt.data_path, 'dataset_card.yml'))

    #========= Preparing dataloaders =========#
    train_dataset = dataset(data_raw['train'], property_dict = opt.info_dict, device = opt.device, **dataloader_config_dict)
    evaluate_dataset = dataset(data_raw['evaluate'], property_dict = opt.info_dict, device = opt.device, **dataloader_config_dict)
    test_dataset = dataset(data_raw['test'], property_dict = opt.info_dict, device = opt.device, **dataloader_config_dict)

    train_data_collator = getattr(train_dataset, '__call__')
    evaluate_data_collator = getattr(evaluate_dataset, '__call__')
    test_data_collator = getattr(test_dataset, '__call__')

    train_iterator, evaluation_iterator, test_iterator = None, None, None
    g = torch.Generator()
    g.manual_seed(opt.seed + rank)

    if not hasattr(opt, 'train') or (hasattr(opt, 'train') and opt.train):
        train_iterator = DataLoader(train_dataset, shuffle = True, batch_size=opt.batch_size, \
            collate_fn = train_data_collator, num_workers=opt.n_worker, worker_init_fn = seed_worker,\
            generator = g, pin_memory = False)
    if not hasattr(opt, 'evaluation') or (hasattr(opt, 'evaluation') and opt.evaluation):
        evaluation_iterator = DataLoader(evaluate_dataset, batch_size=opt.batch_size, \
            collate_fn = evaluate_data_collator, num_workers=opt.n_worker, worker_init_fn = seed_worker,\
            generator = g, pin_memory = False)
    if not hasattr(opt, 'test') or (hasattr(opt, 'test') and opt.test):
        test_iterator = DataLoader(test_dataset, batch_size=opt.batch_size, \
            collate_fn = test_data_collator, num_workers=opt.n_worker, worker_init_fn = seed_worker,\
            generator = g, pin_memory = False)

    return train_iterator, evaluation_iterator, test_iterator

