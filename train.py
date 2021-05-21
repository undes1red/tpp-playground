# The model training script

import argparse, time, os, math, random
import torch

from src.utils import getLogger, path, print_performances, FileLogger, read_json, suffix

from src.model import get_model
from src.optimizer.optim import ScheduledOptim
from src.optimizer.loss import train_epoch, eval_epoch
from src.data.dataloader import prepare_dataloaders

logger = getLogger(__name__)

def train(model, training_data, evaluation_data, test_data, optimizer, device, opt):
    ''' Start training '''

    log_train_file, log_eva_file, log_test_file = None, None, None
    folder_suffix = suffix(opt, 'model_name', 'lr', 'batch_size', 'epoch', 'd_history', 'd_intensity', 'rnn_layers', 'mlp_layers')
    if not os.path.exists(os.path.join(opt.save_model, 'output' + folder_suffix)):
        os.mkdir(os.path.join(opt.save_model, 'output' + folder_suffix))

    if opt.log:
        log_folder = 'log' + folder_suffix
        if not os.path.exists(os.path.join(opt.log, log_folder)):
            os.mkdir(os.path.join(opt.log, log_folder))
        log_train_file = os.path.join(opt.log, log_folder, 'train.log')
        log_eva_file = os.path.join(opt.log, log_folder, 'evaluate.log')
        log_test_file = os.path.join(opt.log, log_folder, 'test.log')

        logger.info(f'Training performance will be written to file: \n{log_train_file},\n{log_eva_file},\n{log_test_file}')
        # These log_items defined here should match corresponding logger's print() method.
        log_item = ['Epoch', 'Loss', 'Gap']
        log_format = ['', ':8.5f', ':8.5f']

        file_logger = FileLogger(log_item, training_log = log_train_file, evaluation_log = log_eva_file, test_log = log_test_file)

    warmup_epoches = opt.n_warmup_steps / len(training_data)
    num_format = [':8.5f', ':8.5f', ':2.5f']
    eva_losses_min = math.inf

    for epoch_i in range(opt.epoch):
        logger.info('[ Epoch ' + str(epoch_i + 1) + ' ]')

        train_loss, fact_tr = train_epoch(model, training_data, optimizer, device)
        print_performances(logger = logger, procedure='Training', absolute_loss=train_loss, relative_loss=train_loss-fact_tr, 
                           average_lr=optimizer.get_lr(), num_format = num_format)

        eva_loss, fact_ev = eval_epoch(model, evaluation_data, device)
        print_performances(logger = logger, procedure='Evaluation', absolute_loss=eva_loss, relative_loss=eva_loss-fact_ev,
                           average_lr=optimizer.get_lr(), num_format = num_format)

        test_loss, fact_test = eval_epoch(model, test_data, device)
        print_performances(logger = logger, procedure='Test', absolute_loss=test_loss, relative_loss=test_loss-fact_test, 
                           average_lr=optimizer.get_lr(), num_format = num_format)

        checkpoint = {'epoch': epoch_i, 'settings': opt,
                      'model': model.state_dict()}

        if opt.save_model and epoch_i > warmup_epoches:
            if opt.save_mode == 'all':
                model_name = os.path.join(
                    opt.save_model, ('_tr_loss_{tr_loss:3.3f}' + '_ev_loss_{eva_loss:3.3f}' + '.chkpt').format(tr_loss=train_loss, eva_loss=eva_loss))
                torch.save(checkpoint, model_name)
            elif opt.save_mode == 'best':
                model_name = os.path.join(opt.save_model, 'output' + folder_suffix, 'checkpoint.chkpt')
                if eva_loss < eva_losses_min and epoch_i > warmup_epoches:
                    eva_losses_min = eva_loss
                    torch.save(checkpoint, model_name)
                    logger.info('    - The checkpoint file has been updated.')

        if file_logger:
            file_logger.print(logger_name = 'training_log', num_format = log_format, 
                              Epoch = epoch_i, Loss = train_loss, Gap = train_loss - fact_tr)
            file_logger.print(logger_name = 'evaluation_log', num_format = log_format, 
                              Epoch = epoch_i, Loss = eva_loss, Gap = eva_loss - fact_ev)
            file_logger.print(logger_name = 'test_log', num_format = log_format, 
                              Epoch = epoch_i, Loss = test_loss, Gap = test_loss - fact_test)


def main():
    ''' 
    Usage:
    See scripts files in directory scripts/ for examples. Check all arguments via 'python3 train.py --help'
    '''

    parser = argparse.ArgumentParser()
    # The Ultimate
    parser.add_argument('--no_seed', action='store_true',
                        help='Do not freeze random seed. Use this option if you want to explore your model robustness.')
    parser.add_argument('--seed', type=int, default=42,
                        help='The global random seed.')
    parser.add_argument('--cuda', action='store_true', 
                        help="Set it to true if you want to use GPU.")

    # Model save
    parser.add_argument('--data_path', action=path, default=None, help='Input dataset file path.')
    parser.add_argument('--log', action=path, default=None, help='Log file path.')
    parser.add_argument('--save_model', action=path, default=None, help='Saved checkpoint file path.')
    parser.add_argument('--save_mode', type=str, choices=['all', 'best'], default='best', help='Store all model checkpoints or only store the best one.')

    # Training procedure related hyperparameters
    parser.add_argument('--epoch', type=int, default=10, help='Epoch number')
    parser.add_argument('-b', '--batch_size', type=int, default=2048, help='Batch size')

    # Model-related hyperparameters
    parser.add_argument('--model_name', default=None,
                        help="The model name.")
    parser.add_argument('--model_json', action=path, default=None,
                        help="The path of json file that contains model hyperparameters.")

    # Optimizer-related hyperparameters
    parser.add_argument('--optim_json', action=path, default=None,
                        help='The path of json file that contains optimizer and scheduler settings.')
    parser.add_argument('--custom_op', action='store_true', 
                        help='Set it to true if you want to use your own optimizer or that from third-party packages.')
    parser.add_argument('--op_name', type=str, default='AdamW', 
                        help='The name of optimizer. All optimizer hyperparameters are set as default.')
    parser.add_argument('--lr_sched', action='store_true', 
                        help='Do you want to use learning rate scheduler? If scheduler is disabled, the warmup settings won\'t come into effect.')
    parser.add_argument('--n_warmup_steps', type=int, default=2000, 
                        help='The number of warmup steps. Models during warmup won\'t be stored.')
    parser.add_argument('--lr', type=float, default=0.1, 
                        help='Input learning rate. The real learning rate could change due to the lr scheduler.')
    parser.add_argument('--n_cycles', type=float, default=0.5)
    parser.add_argument('--last_epoch', type=int, default=-1)

    opt = parser.parse_args()

    if torch.__version__ == '1.4.0':
        raise logger.exception('Due to the pytorch issue #36313(https://github.com/pytorch/pytorch/issues/36313), several learning rate schedulers including LambdaLR fail to run. Please update PyTorch to 1.5.0 or above.')

    # Reproducibility
    if opt.no_seed:
        opt.seed = random.randint(0, 2**16)
        logger.warning(f'Random seed is not given explicitly. This time the used random seed is {opt.seed}.')
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Debug
    torch.autograd.set_detect_anomaly(True)

    if not opt.log and not opt.save_model:
        logger.warning('No experiment result will be saved.')

    # Cuda
    opt.device = torch.device(
        'cuda' if opt.cuda and torch.cuda.is_available() else 'cpu')

    # Create there dirs if they don't exist.
    if not os.path.isdir(opt.log):
        os.makedirs(opt.log)
    if not os.path.isdir(opt.save_model):
        os.makedirs(opt.save_model)

    #========= Loading Dataset =========#

    if opt.data_path:
        training_data, evaluation_data, test_data, opt.training_size = prepare_dataloaders(opt)
    else:
        raise logger.exception("Wrong input data path.")

    logger.info(opt)

    model_param = read_json(opt.model_json)
    logger.info(f'The input model hyperparameters are {model_param}')
    # Load model
    model = get_model(opt.model_name)(
        **model_param
    ).to(opt.device)
    opt.__dict__.update(model_param)

    # Due to the complexity of learning rate scheduler, the scheduler is fixed. 
    # If you want to use another learning rate scheduler, plz modify it in src.optim.
    sched_optimizer = ScheduledOptim(opt, model)

    train(model, training_data, evaluation_data,
          test_data, sched_optimizer, opt.device, opt)


if __name__ == '__main__':
    main()
