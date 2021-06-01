# The model training script

import argparse, os, random
import torch
from tqdm import tqdm
from itertools import cycle

from src.utils import getLogger, path, print_performances, FileLogger, read_json, suffix, lst_add_lst, lst_divide, evaluation, Metric

from src.model import get_model
from src.optimizer.optim import ScheduledOptim
from src.data import prepare_dataloaders

logger = getLogger(__name__)


def postprocess(input):
    return [input[0], input[0] - input[1]]

def log_print_format(input):
    format_dict = {}
    format_dict['absolute_loss'] = input[0]
    format_dict['relative_loss'] = input[1]
    return format_dict

def logfile_print_format(input):
    format_dict = {}
    format_dict['absolute loss'] = input[0]
    format_dict['relative loss'] = input[1]
    return format_dict

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
        logfile_format = {'step': '', 'absolute loss': ':8.5f', 'relative loss': ':8.5f'}

        file_logger = FileLogger(logfile_format, training_log = log_train_file, evaluation_log = log_eva_file, test_log = log_test_file)

    # What you should modify
    log_format = {'absolute_loss': ':8.5f', 'relative_loss': ':8.5f'}
    output_length = 2
    metric_checker = Metric(2)
    report_sum = [0] * output_length # [absolute loss sum, relative loss sum]

    desc = '  - (Training)   '
    step_range = range(1, opt.n_training_steps + 1)
    training = cycle(iter(training_data))
    optimizer.zero_grad()
    do_update = True

    for current_step in tqdm(step_range, desc=desc, leave=False):
        data = next(training)
        do_update = current_step % opt.agg_update_step == 0
        step_result = model_class.train_step(model, data, optimizer, device = opt.device, update_or_not = do_update)
        report_sum = lst_add_lst(report_sum, step_result)
    
        if current_step % opt.n_report_steps == 0:
            logger.warning(f'Brief training status report at step {current_step}.')
            report_sum = postprocess(lst_divide(report_sum, opt.n_report_steps))
            print_performances(logger = logger, procedure='Training', num_format = log_format, **log_print_format(report_sum))
            if file_logger:
                file_logger.print(logger_name = 'training_log', step = current_step, **logfile_print_format(report_sum))
            report_sum = [0] * output_length
            
        if current_step % opt.n_evaluation_steps == 0:
            logger.warning(f'Model evaluation and checkpoint saving at step {current_step}.')
            eva_report = postprocess(
                evaluation(evaluation_data, model, model_class, '  - (Evaluating)   ', device = opt.device, output_length = output_length)
            )
            print_performances(logger = logger, procedure='Evaluation', num_format = log_format, **log_print_format(eva_report))

            test_report = postprocess(
                evaluation(test_data, model, model_class, '  - (Testing)   ', device = opt.device, output_length = output_length)
            )
            print_performances(logger = logger, procedure='Test', num_format = log_format, **log_print_format(test_report))
            
            # We will store the checkpoint after model evaluation.
            checkpoint = {'step': current_step, 'settings': opt, 'model': model.state_dict()}
        
            if opt.save_model and current_step > opt.n_warmup_steps:
                if opt.save_mode == 'all':
                    model_name = os.path.join(
                            opt.save_model, (f'_training_step_{current_step}' + '.chkpt'))
                    torch.save(checkpoint, model_name)
                elif opt.save_mode == 'best':
                    model_name = os.path.join(opt.save_model, 'output' + folder_suffix, 'checkpoint.chkpt')
                    if metric_checker.compare(eva_report) and current_step > opt.n_warmup_steps:
                            torch.save(checkpoint, model_name)
                            logger.info('    - The checkpoint file has been updated.')
            if file_logger:
                file_logger.print(logger_name = 'evaluation_log', step = current_step, **logfile_print_format(eva_report))
                file_logger.print(logger_name = 'test_log', step = current_step, **logfile_print_format(test_report))
    
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
    parser.add_argument('--n_worker', default=0, type=int,
              help='The number of dataloader workers. For most datasets, multiprocessing can speed up the training procedure. But you should set it to lower value, even 0 \
                  if you meet \'received 0 items of ancdata\' exception.')

    # Model save
    parser.add_argument('--log', action=path, default=None, help='Log file path.')
    parser.add_argument('--save_model', action=path, default=None, help='Saved checkpoint file path.')
    parser.add_argument('--save_mode', type=str, choices=['all', 'best'], default='best', help='Store all model checkpoints or only store the best one.')

    # Training procedure related hyperparameters
    parser.add_argument('--n_training_steps', type=int, default=10000, help='The number of training steps.')
    parser.add_argument('--n_evaluation_steps', type=int, default=200, help='The number of steps that follows a model evaluation.')
    parser.add_argument('--n_report_steps', type = int, default=200, help='After a given number of steps, report the current model training status.')
    parser.add_argument('-b', '--batch_size', type=int, default=2048, help='Batch size')
    parser.add_argument('--agg_update_step', type=int, default=1, help='The number of minibatches to do a optimizer step. The number of practical training steps is \
                                                                        agg_update_step * n_training_steps')

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

    if opt.agg_update_step > 1:
        logger.warning(f'Gradient aggregation is detected! The number of practical training steps is multiplied by {opt.agg_update_step}!')
        opt.n_training_steps *= opt.agg_update_step
        opt.n_evaluation_steps *= opt.agg_update_step
        opt.n_report_steps *= opt.agg_update_step
        opt.n_warmup_steps *= opt.agg_update_step

    if torch.__version__ == '1.4.0':
        raise logger.exception('Due to the pytorch issue #36313(https://github.com/pytorch/pytorch/issues/36313), several learning rate schedulers including LambdaLR fail to run. Please update PyTorch to 1.5.0 or above.')
    if torch.__version__ == '1.7.1':
        raise logger.exception('Due to a possible GPU memory leak, we do not suggest to use pytorch 1.7.1 against this framework.')

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
