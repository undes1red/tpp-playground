# Intensity function plotter.
# Should work for both synthetic datasets and real-word datasets.
# Conduct comparisons between learned distributions and real distributions to show the fidelity of learned models.

from src.TPP.utils import suffix, read_json, getLogger, print_args
from src.TPP.model import get_model
from src.TPP.dataloader import prepare_dataloaders
from src.TPP.plotter_utils import draw, spearman_and_l1
import os, argparse, torch
from tqdm import tqdm
from itertools import tee

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=32, help='Set global random seed.')
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
    parser.add_argument('--plot_type', type=str, choices=['intensity', 'probability', 'debug', 'debug_addition_only'], default = 'intensity', help='Temporal point process only.')
    parser.add_argument('--custom_collator', action='store_true',\
                help='If your datasets are special, and the default collator doesn\'t meet your requirements, you can write your own collate_fn() as a method in the dataset class and use it by toggling this argument to True.')

    parser.add_argument('--cuda', action='store_true', help='Use GPUs to accelerate model evaluation speed.')
    parser.add_argument('--synthetic_evaluation', action='store_true', help='Use this argument to switch to synthetic evaluation')
    logger = getLogger(name = 'Plotter')

    # It is nasty
    root = os.path.dirname(os.path.abspath(__file__))
    logger.info(f'Root path is {root}')

    opt = parser.parse_args()
    # Read in model hyperparameters
    opt.device = 'cuda' if opt.cuda and torch.cuda.is_available() else 'cpu'
    opt.data_path = os.path.join(root, 'data', 'inputs', opt.dataset_name)
    model_param = read_json(os.path.join(root, 'config', opt.model_name, opt.model_config)) if opt.model_config else {}
    opt.model_config = os.path.basename(os.path.join(root, 'config', opt.model_name, opt.model_config)) if opt.model_config else None
    param_names = list(model_param.keys())
    opt.__dict__.update(model_param)

    # Find the checkpoint file.
    model_hyperparameters = suffix(opt, 'model_name', 'lr', 'batch_size', 'n_training_steps', 'used_dataloader_config', 'model_config')
    folder_suffix = 'output_' + model_hyperparameters
    checkpoint_folder = os.path.join(root, 'model', opt.dataset_name, folder_suffix)
    logger.info(f'Choosed model checkpoint file is in directory {checkpoint_folder}.')

    # where these figures output.
    opt.store_dir = os.path.join(root, 'output', opt.dataset_name, '_'.join([opt.model_name, str(opt.model_config) \
                                               , opt.dataloader_name, str(opt.used_dataloader_config),\
                                               suffix(opt, 'lr', 'batch_size', 'n_training_steps')]))
    opt.abs_dataloader_config = os.path.join(root, 'config', opt.model_name, opt.dataloader_config) if opt.dataloader_config else None
    if not os.path.exists(opt.store_dir):
        os.makedirs(opt.store_dir)

    # Load the model training setting.
    model_raw = torch.load(os.path.join(checkpoint_folder, 'checkpoint.chkpt'), map_location=torch.device(opt.device))
    model_state_dict = model_raw['model']
    model_setting = model_raw['settings']

    # we don't need large batch for figure evaluation, so we minimize the batch size to 1.
    opt.batch_size = 1

    # Read in original dataset and create corresponding dataset loader.
    torch.manual_seed(model_setting.seed)
    opt.n_worker = 0
    train, evaluation, test = prepare_dataloaders(opt)

    train = iter(train)
    test = iter(test)
    evaluation = iter(evaluation)

    train_size, test_size, evaluation_size = len(train), len(test), len(evaluation)

    # Copy iterators
    train_mae, train_graph = tee(train)
    test_mae, test_graph = tee(test)
    evaluation_mae, evaluation_graph = tee(evaluation)

    iterator_dict_mae = {
        'train': [train_mae, train_size],
        'test': [test_mae, test_size],
        'evaluation': [evaluation_mae, evaluation_size]
    }

    iterator_dict_graph = {
        'train': train_graph,
        'test': test_graph,
        'evaluation': evaluation_graph       
    }

    # Create model object.
    model_class = get_model(name = opt.model_name)
    model = model_class(device = opt.device, num_events = opt.num_events, **model_param)
    model.eval()

    # Load the model checkpoint.
    model.load_state_dict(model_state_dict)
    opt.n_worker = model_setting.n_worker
    logger.info('Model restore completed.')
    logger.info(print_args(opt))

    graph = False

    if not graph:
        for key, (value, value_size) in iterator_dict_mae.items():
            if key != 'test':
                continue
            print(f'The length of the {key} dataset is {value_size}')

            if opt.synthetic_evaluation:
                rho, r, L1 = 0, 0, 0
            else:
                f1 = 0
                mae_per_event_pred = 0
                mae_per_event_real = 0

            for data in tqdm(value, desc = f'{key}', leave = False, total = value_size):
                input_time = data[0][0]
                input_events = data[0][1]
                mask = data[0][-2]
                mean = data[1][0]
                var = data[1][1]
                # filter out the event sequences with single event.
                if input_time.shape[-1] == 1:
                    continue

                if opt.synthetic_evaluation:
                    rho_, r_, L1_ = spearman_and_l1(model = model, data = data, opt = opt)
                    rho += rho_
                    r += r_
                    L1 += L1_
                else:
                    f1_, _, _, (mae_per_event_predict, mae_per_event_avg), for_debug = \
                        model.mean_absolute_error_per_event(input_time = input_time, input_events = input_events, 
                                                            mask = mask, mean = mean, var = var, fast = True)
                    f1 += f1_
                    mae_per_event_pred += mae_per_event_predict
                    mae_per_event_real += mae_per_event_avg
            
            if opt.synthetic_evaluation:
                rho = rho / value_size
                r = r / value_size
                L1 = L1 / value_size
                report = f'For dataset {key}, the average pearson coefficient is {rho}. The average spearman coefficient is {r}, and the mean of L1 distance is {L1}.'
            else:
                f1 = f1 / value_size
                mae_per_event_pred = mae_per_event_pred / value_size
                mae_per_event_real = mae_per_event_real / value_size
                report = f'For dataset {key}, the average f1 is {f1}. The average of mae_per_event against predictions is {mae_per_event_pred}, while the mean of mae_per_event against real events is {mae_per_event_real}.'
        

            print(report)
            with open(os.path.join(opt.store_dir, f'result_{key}.log'), 'w') as f:
                f.write(report)

    if graph:
        # We will get three records from the training set, test set, and evaluation set, respectively.
        for figure_index in range(opt.figure_count):
            if opt.train:
                train_data = next(iterator_dict_graph['train'])
                draw(model, train_data, 'train', figure_index, opt = opt)
                logger.info(f'Figure train_{figure_index} finished drawing.')

            if opt.evaluation:
                evaluation_data = next(iterator_dict_graph['evaluation'])
                draw(model, evaluation_data, 'evaluation', figure_index, opt = opt)
                logger.info(f'Figure evaluation_{figure_index} finished drawing.')

            if opt.test:
                test_data = next(iterator_dict_graph['test'])
                draw(model, test_data, 'test', figure_index, opt = opt)
                logger.info(f'Figure test_{figure_index} finished drawing.')

    logger.info('Task finished')
