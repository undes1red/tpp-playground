# Intensity function plotter.
# Should work for both synthetic datasets and real-word datasets.
# Conduct comparisons between learned distributions and real distributions to show the fidelity of learned models.

from src.TPP.utils import suffix, read_json, getLogger, print_args
from src.TPP.model import get_model
from src.TPP.dataloader import prepare_dataloaders
from src.TPP.tpp_plotter import draw, spearman_and_l1
import os, argparse, torch
from tqdm import tqdm

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

    train = list(train) if opt.train else []
    test = list(test) if opt.test else []
    evaluation = list(evaluation) if opt.evaluation else []

    train_size, test_size, evaluation_size = len(train), len(test), len(evaluation)

    data_dict = {
        'train': [train, train_size],
        'test': [test, test_size],
        'evaluation': [evaluation, evaluation_size]
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
    MAE_E = True

    if not graph:
        for key, (value, value_size) in data_dict.items():
            if key != 'test':
                continue
            print(f'The length of the {key} dataset is {value_size}')

            if opt.synthetic_evaluation:
                rho, r, L1 = 0, 0, 0
            else:
                if MAE_E:
                    f1 = 0
                    mae_per_event_pred = 0
                    mae_per_event_real = 0
                    probability_integral_sum = 0
                    all_event_pred_mean = 0
                    all_event_pred_var = 0
                else:
                    mae = 0

            for data in tqdm(value, desc = f'{key}', leave = False, total = value_size):
                if opt.dataloader_name == 'syn':
                    input_time = data[0][0]
                    input_events = data[0][1]
                    mask = data[0][-2]
                    mean = data[1][0]
                    var = data[1][1]
                elif opt.dataloader_name == 'ifl':
                    input_events, input_time, mask = data[0]
                    if data[2] == None:
                        mean, var = 0, 1
                    else:
                        mean, var = data[2]

                # filter out the event sequences with single event.
                if input_time.shape[-1] == 1:
                    continue

                if opt.synthetic_evaluation:
                    rho_, r_, L1_ = spearman_and_l1(model = model, data = data, opt = opt)
                    rho += rho_
                    r += r_
                    L1 += L1_
                else:
                    if MAE_E:
                        # MAE-E
                        f1_, _, probability_integral_sum_, all_event_pred_, \
                            (mae_per_event_predict_, mae_per_event_avg_), \
                            (mae_perdict_each_event, mae_event_next_each_event) = \
                            model.mean_absolute_error_per_event(input_time = input_time, input_events = input_events, 
                                                                mask = mask, mean = mean, var = var, fast = True)
                        probability_integral_sum_ = probability_integral_sum_.detach().mean(dim = -1)

                        # For RMTPP
                        if opt.model_name == 'rmtpp':
                            f1 += f1_
                            mae_per_event_pred += mae_per_event_predict_
                            mae_per_event_real += mae_per_event_avg_
                            probability_integral_sum += probability_integral_sum_
                        else:
                        # For MTPP models except RMTPP
                            results = zip(f1_, mae_per_event_predict_, mae_per_event_avg_, probability_integral_sum_, all_event_pred_)
                            for f1_per_seq, mae_per_event_predict_per_seq, mae_per_event_avg_per_seq, probability_integral_sum_per_seq, all_event_pred_per_seq in results:
                                f1 += f1_per_seq
                                mae_per_event_pred += mae_per_event_predict_per_seq
                                mae_per_event_real += mae_per_event_avg_per_seq
                                probability_integral_sum += probability_integral_sum_per_seq
                                all_event_pred_mean += all_event_pred_per_seq.mean()
                                all_event_pred_var += all_event_pred_per_seq.var()

                        
                            f1 /= len(f1_)
                            mae_per_event_pred /= len(mae_per_event_predict_)
                            mae_per_event_real /= len(mae_per_event_avg_)
                            probability_integral_sum /= len(mae_per_event_avg_)
                            all_event_pred_mean /= len(mae_per_event_avg_)
                            all_event_pred_var /= len(mae_per_event_avg_)
                    else:
                        # For RMTPP
                        if opt.model_name == 'rmtpp':
                            events_history, events_next = model.divide_history_and_next(input_events, unsqueeze = False)
                            time_history, time_next = model.divide_history_and_next(input_time, unsqueeze = True)
                            mask_history, mask_next = model.divide_history_and_next(mask, unsqueeze = False)
                        else:
                        # For MTPP models except RMTPP
                            events_history, events_next = model.divide_history_and_next(input_events)
                            time_history, time_next = model.divide_history_and_next(input_time)
                            mask_history, mask_next = model.divide_history_and_next(mask)
                        
                        mae_ = model.mean_absolute_error_static(events_history, time_history, time_next, mask_history, mask_next, mean, var)
                        mae += mae_
            
            if opt.synthetic_evaluation:
                rho = rho / value_size
                r = r / value_size
                L1 = L1 / value_size
                report = f'For dataset {key}, the average pearson coefficient is {rho}. The average spearman coefficient is {r}, and the mean of L1 distance is {L1}.'
            else:
                if MAE_E:
                    f1 = f1 / value_size
                    mae_per_event_pred = mae_per_event_pred / value_size
                    mae_per_event_real = mae_per_event_real / value_size
                    probability_integral_sum = probability_integral_sum / value_size
                    all_event_pred_mean = all_event_pred_mean / value_size
                    all_event_pred_var = all_event_pred_var / value_size
                    report = f'For dataset {key}, the average f1 is {f1}. The average of mae_per_event against predictions is {mae_per_event_pred}, while the mean of mae_per_event against real events is {mae_per_event_real}, the sum of p(m|H) is {probability_integral_sum}, the mean of all predicted times is {all_event_pred_mean}, and the variance is {all_event_pred_var}.'
                else:
                    mae = mae / value_size
                    report = f'For dataset {key}, the average mae is {mae}.'

            print(report)
            with open(os.path.join(opt.store_dir, f'result_{key}.log'), 'w') as f:
                f.write(report)

    if graph:
        # We will get three records from the training set, test set, and evaluation set, respectively.
        if opt.train:
            for idx, train_data in enumerate(data_dict['train'][0]):
                draw(model, train_data, 'train', idx = idx, opt = opt)
                if idx >= opt.figure_count - 1:
                    break

        if opt.evaluation:
            for idx, evaluation_data in enumerate(data_dict['evaluation'][0]):
                draw(model, evaluation_data, 'evaluation', idx = idx, opt = opt)
                if idx >= opt.figure_count - 1:
                    break

        if opt.test:
            for idx, test_data in enumerate(data_dict['test'][0]):
                draw(model, test_data, 'test', idx = idx, opt = opt)
                if idx >= opt.figure_count - 1:
                    break

    logger.info('Task finished')
