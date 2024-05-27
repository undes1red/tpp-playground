import torch, os, importlib, glob

from torch.utils.data import DataLoader
from src.taskhost_utils import get_logger
from src.TPP.dataloader.utils import seed_worker, check_exist
from src.TPP.utils import read_yaml


logger = get_logger(__name__)


def dataloader_zoo(opt):
    module = importlib.import_module('.' + opt.dataloader_name, package = f'src.{opt.procedure}.dataloader')
    return module.get_dataloader()


def find_dataset(opt):
    try:
        dataloader_combo = dataloader_zoo(opt)
    except Exception as e:
        logger.exception(f'{e}.')
        logger.exception(f"Dataloader named {opt.dataloader_name} is not found! Please try again.")
    
    logger.info(f"Dataloader name: {opt.dataloader_name}")
    return dataloader_combo


def prepare_dataloaders(opt):
    '''
    Creates the required dataloader against custom dataloader settings.

    Args:
    * opt:  namespace
            This namespace stores all parsed arguments.
    '''
    available_file_names = [os.path.basename(item) for item in glob.glob(opt.data_path + f'/*.{opt.dataset_type}')]

    # find if required dataset files exists.
    file_names = check_exist(available_file_names, opt.dataset_type, opt.training_data_name, 
                                                                     opt.evaluate_data_name, 
                                                                     opt.test_data_name)

    if len(file_names) == 0:
        logger.exception(f'No available dataset file in {opt.data_path}!')
    else:
        logger.info(f'We are going to read {len(file_names)} files in {opt.data_path}. They are {file_names}. Is that right?')
    
    dataloader_config_dict = read_yaml(opt.abs_dataloader_config) if opt.abs_dataloader_config else {}

    # Read in the used_dataloader_config
    used_dataloader_config_dict = {}
    try:
        if opt.combine_used_and_current_dataloader_config:
            used_dataloader_config_dict = read_yaml(opt.abs_used_dataloader_config) if opt.abs_used_dataloader_config else {}
    except AttributeError as e:
        logger.warning('combine_used_and_current_dataloader_config unset! Possibly we are training a model. We will ignore it.')
    # apply used_dataloader_config to current dataloader config if opt.combine_used_and_current_dataloader_config is True
    dataloader_config_dict.update(used_dataloader_config_dict)

    if opt.abs_dataloader_config is None:
        logger.info(f"No custom dataloader settings! We will use the default dataloader settings.")
    else:
        logger.info(f"Custom dataloader settings are loaded from this config file {opt.abs_dataloader_config}.")
        logger.info(f"Custom dataloader settings are: {dataloader_config_dict}.")

    dataset, read_data = find_dataset(opt)
    data_raw = read_data(opt.data_path, file_names)

    '''
    Now, dataset_card.yml is mandatory for every dataset.
    This YAML file should contain useful information about this dataset, like the number of classes it has.
    '''
    opt.dataloader_config_dict = dataloader_config_dict
    opt.info_dict = read_yaml(os.path.join(opt.data_path, 'dataset_card.yml'))

    #========= Preparing dataloaders =========#
    train_iterator, evaluation_iterator, test_iterator = None, None, None
    g = torch.Generator()
    g.manual_seed(opt.seed)

    if getattr(opt, 'training_data_name') is not None:
        train_dataset = dataset(data_raw[opt.training_data_name], property_dict = opt.info_dict, device = opt.device, **dataloader_config_dict)
        train_iterator = DataLoader(train_dataset, shuffle = True, batch_size = opt.training_batch_size, \
            collate_fn = train_dataset.data_collator, num_workers = opt.n_worker, worker_init_fn = seed_worker,\
            generator = g, pin_memory = True)
    if getattr(opt, 'evaluate_data_name', True) is not None:
        evaluate_dataset = dataset(data_raw[opt.evaluate_data_name], property_dict = opt.info_dict, device = opt.device, **dataloader_config_dict)
        evaluation_iterator = DataLoader(evaluate_dataset, batch_size = opt.evaluation_batch_size, \
            collate_fn = evaluate_dataset.data_collator, num_workers = opt.n_worker, worker_init_fn = seed_worker,\
            generator = g, pin_memory = True)
    if getattr(opt, 'test_data_name', True) is not None:
        test_dataset = dataset(data_raw[opt.test_data_name], property_dict = opt.info_dict, device = opt.device, **dataloader_config_dict)
        test_iterator = DataLoader(test_dataset, batch_size = opt.evaluation_batch_size, \
            collate_fn = test_dataset.data_collator, num_workers = opt.n_worker, worker_init_fn = seed_worker,\
            generator = g, pin_memory = True)

    return {'Training': train_iterator, 
            'Evaluation': evaluation_iterator, 
            'Test': test_iterator}