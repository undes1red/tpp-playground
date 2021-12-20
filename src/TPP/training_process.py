import os, torch
from tqdm import tqdm
from itertools import cycle
from torch.nn.parallel import DistributedDataParallel as DDP

from .utils import print_performances, suffix, lst_add_lst, read_json, \
                   lst_divide, evaluation, Metric, add_prefix_to_keys, print_args, getLogger, FileLogger
from .model import get_model
from .optimizer.optim import ScheduledOptim
from .dataloader import prepare_dataloaders

logger = getLogger(__name__)

'''
Detailed training procedure after all required data are ready.
'''
def train(rank, logger, opt):
    '''
    ========= Loading Dataset =========
    '''
    if opt.data_path:
        training_data, evaluation_data, test_data = prepare_dataloaders(opt, rank = rank)
        opt.training_size = len(training_data)
    else:
        raise logger.exception("Wrong input data path.")

    model_param = read_json(opt.model_json)
    param_names = list(model_param.keys())
    if rank == 0:
        logger.info(f'The input model hyperparameters are {model_param}')
    
    '''
    Load model
    '''
    model_class = get_model(opt.model_name, rank = rank)
    model = model_class(device = opt.device,
        **model_param
    )

    if rank == 0:
        logger.info(print_args(opt))
        logger.info(f'For someone who needs the number of training epoches, the number is {opt.n_training_steps/len(training_data):5.5f}')
        logger.info(f'The number of trainable model parameters is {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    opt.__dict__.update(model_param)

    '''
    Due to the complexity of learning rate scheduler, the scheduler is fixed. 
    If you want to use another learning rate scheduler, plz modify it in src.optim.
    '''
    sched_optimizer = ScheduledOptim(opt, model, rank)
    
    model = DDP(model, device_ids = [rank] if opt.cuda else None, find_unused_parameters = True)

    task(rank = rank, model = model, model_class = model_class, training_data = training_data, evaluation_data = evaluation_data,
          test_data = test_data, optimizer = sched_optimizer, opt = opt, model_suffix = param_names)


def task(model, model_class, training_data, evaluation_data, test_data, optimizer, opt, model_suffix, rank):
    log_train_file, log_eva_file, log_test_file = None, None, None
    folder_suffix = suffix(opt, 'model_name', 'lr', 'batch_size', 'n_training_steps', 'dataloader_config', *model_suffix)
    if not os.path.exists(os.path.join(opt.save_model, 'output_' + folder_suffix)) and rank == 0:
        os.mkdir(os.path.join(opt.save_model, 'output_' + folder_suffix))

    file_logger = None
    if opt.log and rank == 0:
        log_folder = 'log_' + folder_suffix
        if not os.path.exists(os.path.join(opt.log, log_folder)):
            os.mkdir(os.path.join(opt.log, log_folder))
        log_train_file = os.path.join(opt.log, log_folder, 'train.log')
        log_eva_file = os.path.join(opt.log, log_folder, 'evaluate.log')
        log_test_file = os.path.join(opt.log, log_folder, 'test.log')
        log_best_model_file = os.path.join(opt.save_model, 'output_' + folder_suffix, 'checkpoint.log')

        logger.info(f'Training performance will be written to file: \n{log_train_file},\n{log_eva_file},\n{log_test_file}')
        # These log_items defined here should match corresponding logger's print() method.
        file_logger = FileLogger(model_class.logfile_format, training_log = log_train_file, evaluation_log = log_eva_file, test_log = log_test_file)
        metric_format_dict = {
            **{'step': ''},
            **dict(zip([f'metric_{metric_count}' for metric_count in range(1, model_class.metric_number + 1)], [':8.5f'] * model_class.metric_number))
        }
        best_model_logger = FileLogger(metric_format_dict, best_model = log_best_model_file)

        if opt.wandb:
            import wandb
            wandb.init(project = 'Temporal point process', config = vars(opt), group = opt.dataset_name, \
                       name = '-'.join([opt.model_name, os.path.basename(str(opt.model_json)), \
                                        opt.dataset_name, str(opt.dataloader_config)]), \
                       dir = os.path.join(opt.log, log_folder), \
                       resume = 'never'
                       )
            wandb.watch(model, log = 'all', log_freq = opt.n_report_steps)

    metric_checker = Metric(model_class.metric_number)
    format_dict_length = model_class.format_dict_length
    report_sum = [0] * format_dict_length

    desc = '  - (Training)   '
    step_range = range(1, opt.n_training_steps + 1)
    training = cycle(iter(training_data))
    optimizer.zero_grad()

    # Start training
    for current_step in tqdm(step_range, desc=desc, leave=False):
        data = next(training)
        step_result = model_class.train_step(model, data, device = opt.device)
        if current_step % opt.agg_update_step == 0:
            if opt.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
            optimizer.step_and_update_lr()
            optimizer.zero_grad()

        report_sum = lst_add_lst(report_sum, step_result)
    
        if current_step % opt.n_report_steps == 0 and rank == 0:
            logger.warning(f'Brief training status report at step {current_step}.')
            report_sum = model_class.postprocess(lst_divide(report_sum, opt.n_report_steps))
            print_performances(logger = logger, procedure='Training', lr = optimizer.get_lr(), **(model_class.log_print_format(report_sum)))
            if opt.wandb:
                wandb.log(
                    add_prefix_to_keys(model_class.log_print_format(report_sum), temp = 'train_'), commit = False, step = current_step)
                wandb.log({'lr': optimizer.get_lr()}, step = current_step)
            if rank == 0 and file_logger:
                report = model_class.logfile_print_format(report_sum)
                file_logger.print(logger_name = 'training_log', step = current_step, **report)
            report_sum = [0] * format_dict_length
            
        if current_step % opt.n_evaluation_steps == 0:
            if rank == 0:
                logger.warning(f'Model evaluation and checkpoint saving at step {current_step}.')
            eva_report = model_class.postprocess(
                evaluation(evaluation_data, model, model_class, device = opt.device, output_length = format_dict_length, desc = '  - (Evaluation)   ')
            )
            test_report = model_class.postprocess(
                evaluation(test_data, model, model_class, device = opt.device, output_length = format_dict_length, desc = '  - (Test)   ')
            )
            if rank == 0:
                print_performances(logger = logger, procedure='Evaluation', lr = optimizer.get_lr(), **(model_class.log_print_format(eva_report)))
                print_performances(logger = logger, procedure='Test', lr = optimizer.get_lr(), **(model_class.log_print_format(test_report)))
                if opt.wandb:
                    wandb.log(add_prefix_to_keys(model_class.log_print_format(eva_report), temp = 'evaluation_'), commit = False, step = current_step)
                    wandb.log(add_prefix_to_keys(model_class.log_print_format(test_report), temp = 'test_'), step = current_step)
            
                # We will store the checkpoint after model evaluation.
                checkpoint = {'step': current_step, 'settings': opt, 'model': model.module.state_dict(),
                              'optimizer': optimizer.state_dict()}
            
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
                            best_model_dict = dict(zip(
                                [f'metric_{metric_count}' for metric_count in range(1, model_class.metric_number + 1)], \
                                model_class.choose_metric(eva_report, test_report)
                                ))
                            best_model_logger.print(logger_name = 'best_model', step = current_step, **best_model_dict)
                if file_logger:
                    eva = model_class.logfile_print_format(eva_report)
                    test = model_class.logfile_print_format(test_report)
                    file_logger.print(logger_name = 'evaluation_log', step = current_step, **eva)
                    file_logger.print(logger_name = 'test_log', step = current_step, **test)
                    
    if rank == 0:
        logger.warning('Training finished!')
        if opt.wandb:
            wandb.finish()
