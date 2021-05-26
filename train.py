# The model training script

import argparse, os, math, random
import torch
from tqdm import tqdm
from itertools import cycle

from src.utils import getLogger, path, print_performances, FileLogger, read_json, suffix

from src.model import get_model
from src.optimizer.optim import ScheduledOptim
from src.data import prepare_dataloaders

logger = getLogger(__name__)

def evaluation(data, model, model_class, desc, device):
    r = range(1, len(data) + 1)
    data_itr = iter(data)
    sum_loss, sum_fact = 0, 0
    
    for _ in tqdm(r, desc=desc, leave=False):
        minibatch = next(data_itr)
        eva_loss, fact_ev = model_class.evaluation_step(model, minibatch, device)
        sum_loss += eva_loss
        sum_fact += fact_ev

    return sum_loss/len(data), sum_fact/len(data)


def train(model, model_class, training_data, evaluation_data, test_data, optimizer, opt, model_suffix):
    ''' Start training '''

    log_train_file, log_eva_file, log_test_file = None, None, None
    folder_suffix = suffix(opt, 'model_name', 'lr', 'batch_size', 'n_training_steps', *model_suffix)
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
        log_item = ['Step', 'Loss', 'Gap']
        log_format = ['', ':8.5f', ':8.5f']

        file_logger = FileLogger(log_item, training_log = log_train_file, evaluation_log = log_eva_file, test_log = log_test_file)

    num_format = [':8.5f', ':8.5f', ':2.5f']
    eva_losses_min = math.inf
    report_loss_sum, report_fact_sum = 0, 0
    desc = '  - (Training)   '
    step_range = range(1, opt.n_training_steps + 1)
    training = cycle(iter(training_data))

    for current_step in tqdm(step_range, desc=desc, leave=False):
        data = next(training)
        train_loss, train_fact = model_class.train_step(model, data, optimizer, device = opt.device)
        report_loss_sum += train_loss
        report_fact_sum += train_fact
    
        if current_step % opt.n_report_steps == 0:
            logger.warning(f'Brief training status report at step {current_step}.')
            print_performances(logger = logger, procedure='Training', absolute_loss=report_loss_sum/opt.n_report_steps,
                               relative_loss=(report_loss_sum-report_fact_sum)/opt.n_report_steps,
                               average_lr=optimizer.get_lr(), num_format=num_format)
            if file_logger:
                file_logger.print(logger_name = 'training_log', num_format = log_format, 
                                  Step = current_step, Loss = report_loss_sum/opt.n_report_steps, Gap = (report_loss_sum - report_fact_sum)/opt.n_report_steps)
            report_loss_sum, report_fact_sum = 0, 0
            
        if current_step % opt.n_evaluation_steps == 0:
            logger.warning(f'Model evaluation and checkpoint saving at step {current_step}.')
            eva_loss, eva_fact = evaluation(evaluation_data, model, model_class, '  - (Evaluating)   ', device = opt.device)
            print_performances(logger = logger, procedure='Evaluation', absolute_loss=eva_loss,
                               relative_loss=eva_loss-eva_fact,
                               average_lr=optimizer.get_lr(), num_format=num_format)

            test_loss, test_fact = evaluation(test_data, model, model_class, '  - (Testing)   ', device = opt.device)
            print_performances(logger = logger, procedure='Test', absolute_loss=test_loss,
                               relative_loss=test_loss-test_fact, average_lr=optimizer.get_lr(), num_format=num_format)
    
            # We will store the checkpoint after model evaluation.
            checkpoint = {'step': current_step, 'settings': opt, 'model': model.state_dict()}
        
            if opt.save_model and current_step > opt.n_warmup_steps:
                if opt.save_mode == 'all':
                    model_name = os.path.join(
                            opt.save_model, ('_tr_loss_{tr_loss:3.3f}' + '_ev_loss_{eva_loss:3.3f}' + '.chkpt').format(tr_loss=train_loss, eva_loss=eva_loss))
                    torch.save(checkpoint, model_name)
                elif opt.save_mode == 'best':
                    model_name = os.path.join(opt.save_model, 'output' + folder_suffix, 'checkpoint.chkpt')
                    if eva_loss < eva_losses_min and current_step > opt.n_warmup_steps:
                            eva_losses_min = eva_loss
                            torch.save(checkpoint, model_name)
                            logger.info('    - The checkpoint file has been updated.')
            if file_logger:
                file_logger.print(logger_name = 'evaluation_log', num_format = log_format, 
                                  Step = current_step, Loss = eva_loss, Gap = eva_loss - eva_fact)
                file_logger.print(logger_name = 'test_log', num_format = log_format, 
                                  Step = current_step, Loss = test_loss, Gap = test_loss - test_fact)
    
    logger.warning('Training finished!')


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

    # Input data
    parser.add_argument('--data_path', action=path, default=None, help='Input dataset file path.')
    parser.add_argument('--dataset_name', default=None, help='Input dataset class name.')

    # Model save
    parser.add_argument('--log', action=path, default=None, help='Log file path.')
    parser.add_argument('--save_model', action=path, default=None, help='Saved checkpoint file path.')
    parser.add_argument('--save_mode', type=str, choices=['all', 'best'], default='best', help='Store all model checkpoints or only store the best one.')

    # Training procedure related hyperparameters
    parser.add_argument('--n_training_steps', type=int, default=10000, help='The number of training steps.')
    parser.add_argument('--n_evaluation_steps', type=int, default=200, help='The number of steps that follows a model evaluation.')
    parser.add_argument('--n_report_steps', type = int, default=200, help='After a given number of steps, report the current model training status.')
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
        training_data, evaluation_data, test_data = prepare_dataloaders(opt)
        opt.training_size = len(training_data)
    else:
        raise logger.exception("Wrong input data path.")

    model_param = read_json(opt.model_json)
    param_names = list(model_param.keys())
    logger.info(f'The input model hyperparameters are {model_param}')
    # Load model
    model_class = get_model(opt.model_name)
    model = model_class(device = opt.device,
        **model_param
    )

    logger.info(opt)
    logger.info(f'For someone who needs the number of training epoches, the number is {opt.n_training_steps/len(training_data):5.5f}')
    logger.info(f'The number of trainable model parameters is {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    opt.__dict__.update(model_param)

    # Due to the complexity of learning rate scheduler, the scheduler is fixed. 
    # If you want to use another learning rate scheduler, plz modify it in src.optim.
    sched_optimizer = ScheduledOptim(opt, model)

    train(model = model, model_class = model_class, training_data = training_data, evaluation_data = evaluation_data,
          test_data = test_data, optimizer = sched_optimizer, opt = opt, model_suffix = param_names)


if __name__ == '__main__':
    main()
