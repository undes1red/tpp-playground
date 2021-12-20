# Intensity function plotter.
# Should work for both synthetic datasets and real-word datasets.
# Conduct comparisons between learned distributions and real distributions to show the fidelity of learned models.

from ..utils import suffix, read_json, getLogger, print_args
from ..model import get_model
from ..dataloader import prepare_dataloaders
from .plotter_utils import draw
import os, argparse, torch

# It is nasty
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42, help='Set global random seed.')
    parser.add_argument('--model_name', type=str, help='The model name of the required checkpoint.')
    parser.add_argument('--model_config', type=str, help='The config file containing hyperparameters corresponding to the required checkpoint.')
    parser.add_argument('--lr', type=float, help='The learning rate used for training the required model.')
    parser.add_argument('--batch_size', type=int, help='The batch size used for training the required model.')
    parser.add_argument('--n_training_steps', type=int, help='The total training step used for training the required model.')
    parser.add_argument('--resolution', type=int, default=100, help='How many interpolating points may each time interval have?')

    parser.add_argument('--dataset_name', type=str, help='The name of used dataset related to the required checkpoint.')
    parser.add_argument('--dataloader_name', type=str, help='The name of used dataset related to the required checkpoint.')
    parser.add_argument('--used_dataloader_config', type=str, default = None, help='The name of used dataset related to the required checkpoint.')
    parser.add_argument('--dataloader_config', type=str, default = None, \
                        help='Choose the dataloader config file in the corresponding model config folder for plot drawing.')
    parser.add_argument('--figure_count', type = int, help='We will select \{figure_count\} records from training set(if set),\
                                                      test set(if set), and evaluation set(if set), respectively. So there will be\
                                                      \{enabled_dataset\} * figure_count plots when the plotter finish running.')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--evaluation', action='store_true')
    parser.add_argument('--plot_type', type=str, choices=['intensity', 'probability', 'debug'], default = 'intensity', help='Temporal point process only.')

    parser.add_argument('--cuda', action='store_true', help='Use GPUs to accelerate model evaluation speed.')
    logger = getLogger(name = 'Plotter')

    opt = parser.parse_args()
    # Read in model hyperparameters
    opt.device = 'cuda' if opt.cuda and torch.cuda.is_available() else 'cpu'
    opt.data_path = os.path.join(root, 'data', 'inputs', opt.dataset_name)
    model_param = read_json(os.path.join(root, 'config', opt.model_name, opt.model_config))
    param_names = list(model_param.keys())
    opt.__dict__.update(model_param)
    logger.info(print_args(opt))

    # Find the checkpoint file.
    model_hyperparameters = suffix(opt, 'model_name', 'lr', 'batch_size', 'n_training_steps', 'used_dataloader_config', *param_names)
    folder_suffix = 'output_' + model_hyperparameters
    checkpoint_folder = os.path.join(root, 'data', 'outputs', opt.dataset_name, folder_suffix)
    logger.info(f'Choosed model checkpoint file is in directory {checkpoint_folder}.')

    # where these figures output.
    opt.store_dir = os.path.join(root, 'output', opt.dataset_name, '_'.join([opt.model_name, str(opt.model_config) \
                                               , opt.dataloader_name, str(opt.used_dataloader_config),\
                                               suffix(opt, 'lr', 'batch_size', 'n_training_steps')]))
    opt.abs_dataloader_config = os.path.join(root, 'config', opt.model_name, opt.dataloader_config) if opt.dataloader_config else None
    if not os.path.exists(opt.store_dir):
        os.makedirs(opt.store_dir)

    # Create model object.
    model_class = get_model(name = opt.model_name)
    model = model_class(device = opt.device, **model_param)
    model.eval()

    # Load model checkpoint.
    model_raw = torch.load(os.path.join(checkpoint_folder, 'checkpoint.chkpt'), map_location=torch.device(opt.device))
    model_state_dict = model_raw['model']
    model_setting = model_raw['settings']
    model.load_state_dict(model_state_dict)
    opt.n_worker = model_setting.n_worker
    logger.info('Model restore completed.')

    # we don't need large batch for figure evaluation, so we minimize the batch size to 1.
    opt.batch_size = 1

    # Read in original dataset and create corresponding dataset loader.
    torch.manual_seed(model_setting.seed)
    opt.n_worker = 0
    train, evaluation, test = prepare_dataloaders(opt)
    iter_train = iter(train)
    iter_test = iter(test)
    iter_eva = iter(evaluation)

    # We will get three records from the training set, test set, and evaluation set, respectively.
    for figure_index in range(opt.figure_count):
        if opt.train:
            train_data = next(iter_train)
            draw(model, train_data, 'train', figure_index, opt = opt)
            logger.info(f'Figure train_{figure_index} finished drawing.')

        if opt.evaluation:
            evaluation_data = next(iter_eva)
            draw(model, evaluation_data, 'evaluation', figure_index, opt = opt)
            logger.info(f'Figure evaluation_{figure_index} finished drawing.')

        if opt.test:
            test_data = next(iter_test)
            draw(model, test_data, 'test', figure_index, opt = opt)
            logger.info(f'Figure test_{figure_index} finished drawing.')
    
    logger.info('Task finished')
