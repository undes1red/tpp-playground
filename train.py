# The model training script
import argparse, os, random
from tqdm import tqdm
from itertools import cycle

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import sys
import numpy as np
import datetime

from src.utils import getLogger, print_performances, FileLogger, read_json, suffix, lst_add_lst, lst_divide, evaluation, Metric
from src.model import get_model
from src.optimizer.optim import ScheduledOptim
from src.data import prepare_dataloaders

logger = getLogger(__name__)
# Hope we can get rid of absolute path in training scripts.
root_path = os.path.dirname(os.path.abspath(__file__))

def train(model, model_class, training_data, evaluation_data, test_data, optimizer, opt, model_suffix, rank):
    '''
    Main training procedure. Mostly, one should not modify it.
    If you do everything in the right way, it won't complain about anything and will commence the model training.
    '''

    log_train_file, log_eva_file, log_test_file = None, None, None
    model_hyperparameters = suffix(opt, 'model_name', 'lr', 'batch_size', 'n_training_steps', *model_suffix)
    folder_suffix = "_".join(map(str, model_hyperparameters.values()))
    if not os.path.exists(os.path.join(opt.save_model, 'output_' + folder_suffix)) and rank == 0:
        os.mkdir(os.path.join(opt.save_model, 'output_' + folder_suffix))

    writer, file_logger = None, None
    if opt.log and rank == 0:
        log_folder = 'log_' + folder_suffix
        if not os.path.exists(os.path.join(opt.log, log_folder)):
            os.mkdir(os.path.join(opt.log, log_folder))
        log_train_file = os.path.join(opt.log, log_folder, 'train.log')
        log_eva_file = os.path.join(opt.log, log_folder, 'evaluate.log')
        log_test_file = os.path.join(opt.log, log_folder, 'test.log')

        logger.info(f'Training performance will be written to file: \n{log_train_file},\n{log_eva_file},\n{log_test_file}')
        # These log_items defined here should match corresponding logger's print() method.
        file_logger = FileLogger(model_class.logfile_format, training_log = log_train_file, evaluation_log = log_eva_file, test_log = log_test_file)
        model_logger = getLogger(name = 'best_model', file = os.path.join(opt.save_model, 'output_' + folder_suffix, 'checkpoint.log'))

        if opt.tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir = os.path.join(opt.log, log_folder))

    metric_checker = Metric(model_class.metric_number)
    report_sum = [0] * opt.report_result_length # [absolute loss sum, relative loss sum]

    desc = '  - (Training)   '
    step_range = range(1, opt.n_training_steps + 1)
    training = cycle(iter(training_data))
    optimizer.zero_grad()

    if opt.profiler:
        profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                          schedule=torch.profiler.schedule(wait=1,warmup=1,active=2),
                                          on_trace_ready=torch.profiler.tensorboard_trace_handler(dir_name = os.path.join(opt.log, 'tensorboard')),with_stack=True)
    for current_step in tqdm(step_range, desc=desc, leave=False):
        data = next(training)
        step_result = model_class.train_step(model, data, device = opt.device)
        if current_step % opt.agg_update_step == 0:
            optimizer.step_and_update_lr()
            optimizer.zero_grad()

        report_sum = lst_add_lst(report_sum, step_result)
        if opt.profiler:
            profiler.step()
    
        if current_step % opt.n_report_steps == 0 and rank == 0:
            logger.warning(f'Brief training status report at step {current_step}.')
            report_sum = model_class.postprocess(lst_divide(report_sum, opt.n_report_steps))
            print_performances(logger = logger, procedure='Training', **(model_class.log_print_format(report_sum)))
            if rank == 0 and file_logger:
                report = model_class.logfile_print_format(report_sum)
                file_logger.print(logger_name = 'training_log', step = current_step, **report)
                if writer:
                    for key, value in report.items():
                        writer.add_scalar(tag = 'Train/' + key, scalar_value = value, global_step = current_step)
            report_sum = [0] * opt.report_result_length
            
        if current_step % opt.n_evaluation_steps == 0:
            if rank == 0:
                logger.warning(f'Model evaluation and checkpoint saving at step {current_step}.')
            eva_report = model_class.postprocess(
                evaluation(evaluation_data, model, model_class, device = opt.device, output_length = opt.report_result_length)
            )
            test_report = model_class.postprocess(
                evaluation(test_data, model, model_class, device = opt.device, output_length = opt.report_result_length)
            )
            if rank == 0:
                print_performances(logger = logger, procedure='Evaluation', **(model_class.log_print_format(eva_report)))
                print_performances(logger = logger, procedure='Test', **(model_class.log_print_format(test_report)))
            
                # We will store the checkpoint after model evaluation.
                checkpoint = {'step': current_step, 'settings': opt, 'model': model.module.state_dict()}
            
                if opt.save_model and current_step > opt.n_warmup_steps:
                    if opt.save_mode == 'all':
                        model_name = os.path.join(
                                opt.save_model, (f'_training_step_{current_step}' + '.chkpt'))
                        torch.save(checkpoint, model_name)
                    elif opt.save_mode == 'best':
                        model_name = os.path.join(opt.save_model, 'output_' + folder_suffix, 'checkpoint.chkpt')
                        if metric_checker.compare(model_class.choose_metric(eva_report, test_report)) and current_step > opt.n_warmup_steps:
                            torch.save(checkpoint, model_name)
                            logger.info('  The checkpoint file has been updated.')
                            model_logger.info(f'{metric_checker.show()}')
                if file_logger:
                    eva = model_class.logfile_print_format(eva_report)
                    test = model_class.logfile_print_format(test_report)
                    file_logger.print(logger_name = 'evaluation_log', step = current_step, **eva)
                    file_logger.print(logger_name = 'test_log', step = current_step, **test)
                    if writer:
                        for key in eva.keys():
                            writer.add_scalar(tag = 'Evaluation/' + key, scalar_value = eva[key], global_step = current_step)
                            writer.add_scalar(tag = 'Test/' + key, scalar_value = test[key], global_step = current_step)
                    
    if rank == 0:
        if writer:
            writer.add_hparams(model_hyperparameters, model_class.logfile_print_format(metric_checker.show()))
            writer.flush()
            writer.close()
        logger.warning('Training finished!')
        logger.info(f'The best metric value is {metric_checker.show()}.')

def _main(rank, logger, opt):
    '''
    DistributedDataParallel model preparation and another stuff.
    '''
    if opt.agg_update_step > 1 and rank == 0:
        logger.warning(f'Gradient aggregation is detected! The number of practical training steps is multiplied by {opt.agg_update_step}!')
        opt.n_training_steps *= opt.agg_update_step
        opt.n_evaluation_steps *= opt.agg_update_step
        opt.n_report_steps *= opt.agg_update_step
        opt.n_warmup_steps *= opt.agg_update_step

    if torch.__version__ == '1.4.0' and rank == 0:
        raise logger.exception('Due to the pytorch issue #36313(https://github.com/pytorch/pytorch/issues/36313), several learning rate schedulers including LambdaLR fail to run. Please update PyTorch to 1.5.0 or above.')

    if not opt.log and not opt.save_model and rank == 0:
        logger.warning('No experiment result will be saved.')

    # cuda
    opt.device = torch.device(
        f'cuda:{rank:d}' if opt.cuda and torch.cuda.is_available() else 'cpu')

    if rank == 0:
        if opt.device.type == 'cuda':
            logger.info('Found {} CUDA devices.'.format(torch.cuda.device_count()))
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                logger.info('{} \t Memory: {:.2f}GB'.format(props.name, props.total_memory / (1024**3)))
        else:
            logger.info('WARNING: Using device {}'.format(opt.device))

    # Create there dirs if they don't exist.
    if not os.path.isdir(opt.log):
        os.makedirs(opt.log)
    if not os.path.isdir(opt.save_model):
        os.makedirs(opt.save_model)

    #========= Loading Dataset =========#

    if opt.data_path:
        training_data, evaluation_data, test_data = prepare_dataloaders(opt, rank = rank)
        opt.training_size = len(training_data)
    else:
        raise logger.exception("Wrong input data path.")

    model_param = read_json(opt.model_json)
    param_names = list(model_param.keys())
    if rank == 0:
        logger.info(f'The input model hyperparameters are {model_param}')
    # Load model
    model_class = get_model(opt.model_name, rank = rank)
    model = model_class(device = opt.device,
        **model_param
    )
    model = DDP(model, device_ids = [rank], find_unused_parameters = True)

    if rank == 0:
        logger.info(opt)
        logger.info(f'For someone who needs the number of training epoches, the number is {opt.n_training_steps/len(training_data):5.5f}')
        logger.info(f'The number of trainable model parameters is {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    opt.__dict__.update(model_param)

    # Due to the complexity of learning rate scheduler, the scheduler is fixed. 
    # If you want to use another learning rate scheduler, plz modify it in src.optim.
    sched_optimizer = ScheduledOptim(opt, model, rank)

    train(rank = rank, model = model, model_class = model_class, training_data = training_data, evaluation_data = evaluation_data,
          test_data = test_data, optimizer = sched_optimizer, opt = opt, model_suffix = param_names)


def main(rank, ngpus, opt):
    '''
    Multiprocessing training controller.
    '''
    dist.init_process_group("nccl", rank=rank, world_size=ngpus, timeout=datetime.timedelta(minutes=30))

    logger = getLogger('__Trainer__')

    try:
        _main(rank = rank, logger = logger, opt = opt)
    except:
        import traceback
        logger.error(traceback.format_exc())
        raise

    dist.destroy_process_group()


if __name__ == '__main__':
    ''' 
    Usage:
    Take scripts files in directory scripts/ as examples. One can also check all available arguments and corresponding help via 'python3 train.py --help'
    '''

    parser = argparse.ArgumentParser()
    # The Ultimate
    parser.add_argument('--no_seed', action='store_true',
                        help='Do not freeze random seed. Use this option if you want to explore your model robustness.')
    parser.add_argument('--seed', type=int, default=42,
                        help='The global random seed.')
    parser.add_argument('--cuda', action='store_true', 
                        help="Set it to true if you want to use GPU.")
    parser.add_argument('--profiler', action='store_true', 
                        help="Use a profiler to probe the bottleneck of your model when your model is slow. (Because of pytorch issue #56008, profiler support is now disabled.)")
    parser.add_argument("--ngpus", type=int, default=1,
                        help="If you want to train your model on multiple GPUs, please set this parameter with integer bigger than 1.")


    # Input data
    parser.add_argument('--dataset_name', type=str, default=None, help='Feeding in dataset name. All datasets should be placed in root/data/input')
    parser.add_argument('--dataloader_name', default=None, help='Input dataloader class name.')
    parser.add_argument('--n_worker', default=0, type=int,
              help='The number of dataloader workers. For most datasets, multiprocessing can speed up the training procedure. But you should set it to lower value, even 0 \
                  if you meet \'received 0 items of ancdata\' exception.')

    # Model save and log management
    parser.add_argument('--save_mode', type=str, choices=['all', 'best'], default='best', help='Store all model checkpoints or only store the best one.')
    parser.add_argument('--tensorboard', action='store_true', help='Use tensorboard to visualize the training result.')
    parser.add_argument('--report_result_length', type=int, default=2, help='The number of metric numbers each running step returns.')

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
    parser.add_argument('--model_json', type=str, default=None,
                        help="The path of json file that contains model hyperparameters.")

    # Optimizer-related hyperparameters
    parser.add_argument('--optim_json', type=str, default=None,
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
    logger = getLogger("MP_parser")

    # Reproducibility
    if opt.no_seed:
        opt.seed = random.randint(0, 2**16)
        logger.warning(f'Random seed is not given explicitly. This time the used random seed is {opt.seed}.')
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    random.seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Relative path to absolute path
    opt.data_path = os.path.join(root_path, 'data', 'inputs', opt.dataset_name)
    opt.log = os.path.join(root_path, 'log', opt.dataset_name)
    opt.save_model = os.path.join(root_path, 'data', 'outputs', opt.dataset_name)
    opt.model_json = os.path.join(root_path, 'config', opt.model_name, opt.model_json)
    opt.optim_json = os.path.join(root_path, 'config', opt.optim_json)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(int(np.random.randint(10000, 20000)))
    try:
        mp.set_start_method("forkserver")
        mp.spawn(main, args = (opt.ngpus, opt), nprocs=opt.ngpus, join=True)
    except Exception:
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
