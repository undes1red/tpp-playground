# Intensity function plotter.
# Should work for both synthetic datasets and real-word datasets.
# Conduct comparisons between learned distributions and real distributions to show the fidelity of learned models.

from src.utils import suffix, read_json, getLogger
from src.model import get_model
from src.dataloader import prepare_dataloaders
import os, argparse, torch
import matplotlib.pyplot as plt
import seaborn as sns

root = os.path.dirname(os.path.abspath(__file__))

def draw_intensity(model, data, desc, plot_count):
    '''
    Now you should investigate your own model implementations and modify this function.
    Because of the design of training_step() and evaluation_step(), intensity and integral outputs can not be handled automatically.

    data: [time_diff, score, target_intensity]
    '''
    # Intensity probe
    intensity_integral, intensity = model(data[0])
    print(intensity.squeeze())
    print(data[2].squeeze())
    mse = torch.nn.functional.mse_loss(intensity.squeeze(), data[2].squeeze())
    logger.info(f'The MSE loss between model intensity and target intensity at event points is {mse.item()}.')

    _, expand_intensity, timestamp = model.function_prober(data[0], resolution = 100)
    intensity = expand_intensity.squeeze().detach().cpu().numpy()
    timestamp = timestamp.detach().cpu().numpy().cumsum()

    # Draw plot.
    fig = plt.figure()
    sns.lineplot(x = timestamp[1:], y = intensity[1:])
    plt.savefig(os.path.join(root, 'intensity_' + desc + '_' + str(plot_count) + '.png'), dpi = 1000)
    plt.close(fig = fig)


def draw_probability(model, data, desc, plot_count):
    pass

def draw(model, data, desc, plot_count, type):
    if type == 'intensity':
        draw_intensity(model, data, desc, plot_count)
    elif type == 'probability':
        draw_probability(model, data, desc, plot_count)
    else:
        raise Exception('Unknown plot type detected!')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, help='The model name of the required checkpoint.')
    parser.add_argument('--model_config', type=str, help='The config file containing hyperparameters corresponding to the required checkpoint.')
    parser.add_argument('--lr', type=float, help='The learning rate used for training the required model.')
    parser.add_argument('--batch_size', type=int, help='The batch size used for training the required model.')
    parser.add_argument('--n_training_steps', type=int, help='The total training step used for training the required model.')

    parser.add_argument('--dataset_name', type=str, help='The name of used dataset related to the required checkpoint.')
    parser.add_argument('--dataloader_name', type=str, help='The name of used dataset related to the required checkpoint.')
    parser.add_argument('--dataloader_json', type=str, default = None, help='The name of used dataset related to the required checkpoint.')
    parser.add_argument('--figure_count', type = int, help='We will select \{figure_count\} records from training set(if set),\
                                                      test set(if set), and evaluation set(if set), respectively. So there will be\
                                                      \{enabled_dataset\} * figure_count plots when the plotter finish running.')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--evaluation', action='store_true')
    parser.add_argument('--plot_type', type=str, choices=['intensity', 'probability'], default = 'intensity', help='Temporal point process only.')

    parser.add_argument('--cuda', action='store_true', help='Use GPUs to accelerate model evaluation speed.')
    logger = getLogger(name = 'Plotter')

    opt = parser.parse_args()
    # Read in model hyperparameters
    opt.device = 'cuda' if opt.cuda and torch.cuda.is_available() else 'cpu'
    opt.data_path = os.path.join(root, 'data', 'inputs', opt.dataset_name)
    model_param = read_json(os.path.join(root, 'config', opt.model_name, opt.model_config))
    param_names = list(model_param.keys())
    opt.__dict__.update(model_param)
    logger.info(opt)
    if not os.path.exists(os.path.join(root, 'output')):
        os.mkdir(os.path.join(root, 'output'))

    # Find the checkpoint file.
    model_hyperparameters = suffix(opt, 'model_name', 'lr', 'batch_size', 'n_training_steps', *param_names)
    folder_suffix = 'output_' + '_'.join(map(str, model_hyperparameters.values()))
    checkpoint_folder = os.path.join(root, 'data', 'outputs', opt.dataset_name, folder_suffix)
    logger.info(f'Choosed model checkpoint file is in directory {checkpoint_folder}.')

    # Create model object
    model_class = get_model(name = opt.model_name)
    model = model_class(device = opt.device, **model_param)
    model.eval()

    # load model checkpoint
    model_raw = torch.load(os.path.join(checkpoint_folder, 'checkpoint.chkpt'))
    model_state_dict = model_raw['model']
    model_setting = model_raw['settings']
    model.load_state_dict(model_state_dict)
    opt.seed = model_setting.seed
    opt.n_worker = model_setting.n_worker
    logger.info('Model restore completed.')

    # we don't need large batch of data, so we minimize the batch size.
    opt.batch_size = 1

    # Read in original dataset and create corresponding dataset loader.
    torch.manual_seed(model_setting.seed)
    train, evaluation, test = prepare_dataloaders(opt)

    # We will get three records from the training set, test set, and evaluation set, respectively.
    for figure_index in range(opt.figure_count):
        if opt.train:
            train_data = next(iter(train))
            draw(model, train_data, 'train', figure_index, type = 'intensity')
            logger.info(f'Figure train_{figure_index} finished drawing.')

        if opt.evaluation:
            evaluation_data = next(iter(evaluation))
            draw(model, evaluation_data, 'evaluation', figure_index, type = 'intensity')
            logger.info(f'Figure evaluation_{figure_index} finished drawing.')

        if opt.test:
            test_data = next(iter(test))
            draw(model, test_data, 'test', figure_index, type = 'intensity')
            logger.info(f'Figure test_{figure_index} finished drawing.')
    
    logger.info('Task finished')
