import os, torch
from tqdm import tqdm
from itertools import cycle
from torch.nn.parallel import DistributedDataParallel as DDP

from src.TPP.utils import print_performances, suffix, lst_add_lst, read_json, \
                   lst_divide, evaluation, Metric, add_prefix_to_keys, print_args, getLogger, FileLogger
from src.TPP.model import get_model
from src.TPP.optimizer.optim import ScheduledOptim
from src.TPP.dataloader import prepare_dataloaders


'''
Detailed training procedure after all required data are ready.
Define the logger.
'''
logger = getLogger(__name__)

class TPPTrainer:
    def __init__(self):
        pass

    def train(self, rank, opt):
        '''
        Store required initial information
        '''
        self.opt = opt
        self.rank = rank

        '''
        ========= Loading Dataset =========
        '''
        if opt.data_path:
            self.training_data, self.evaluation_data, self.test_data = prepare_dataloaders(opt, rank = rank)
            opt.training_size = len(self.training_data)
        else:
            raise logger.exception("Wrong input data path.")
    
        model_param = read_json(opt.abs_model_config) if opt.abs_model_config else {}
        self.param_names = list(model_param.keys())
        if rank == 0:
            logger.info(f'The input model hyperparameters are {model_param}')
        
        '''
        Load model
        '''
        self.model_class = get_model(opt.model_name, rank = rank)
        model = self.model_class(device = opt.device, num_events = opt.num_events,
            **model_param
        )
    
        if rank == 0:
            logger.info(print_args(opt))
            logger.info(f'For someone who needs the number of training epoches, the number is {opt.n_training_steps/len(self.training_data):5.5f}')
            logger.info(f'The number of trainable model parameters is {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    
        self.opt.__dict__.update(model_param)
    
        '''
        Due to the complexity of learning rate scheduler, the scheduler is fixed. 
        If you want to use another learning rate scheduler, plz modify it in src.optim.
        '''
        self.sched_optimizer = ScheduledOptim(opt, model, rank)
        
        self.model = DDP(model, device_ids = [rank] if opt.cuda else None, find_unused_parameters = True)
    
        self.task()
    
    
    def task(self):
        '''
        Directory preparation
        '''

        '''
        Create log and model-saving dirs if they are not present.
        '''
        if not os.path.isdir(self.opt.log):
            os.makedirs(self.opt.log)
        if not os.path.isdir(self.opt.save_model):
            os.makedirs(self.opt.save_model)

        self.folder_suffix = suffix(self.opt, 'model_name', 'lr', 'batch_size', 'n_training_steps', 'dataloader_config', 'model_config')
        if not os.path.exists(os.path.join(self.opt.save_model, 'output_' + self.folder_suffix)) and self.rank == 0:
            os.mkdir(os.path.join(self.opt.save_model, 'output_' + self.folder_suffix))

        '''
        Setting up file loggers and a wandb online logger.
        '''
        if self.opt.log and self.rank == 0:
            self.file_logger, self.best_model_logger = self.create_file_logger()
    
            if self.opt.wandb:
                import wandb
                wandb.init(project = 'Temporal point process', config = vars(self.opt), group = self.opt.dataset_name, \
                           name = '-'.join([self.opt.model_name, str(self.opt.model_config), \
                                            self.opt.dataset_name, str(self.opt.dataloader_config)]), \
                           dir = os.path.join(self.opt.log, self.log_folder), \
                           resume = 'never', settings = wandb.Settings(start_method="fork")
                           )
                wandb.watch(self.model, log = 'all', log_freq = self.opt.n_report_steps)
    
        '''
        Metric checker for choosing the best model during training.
        '''
        self.metric_checker = Metric(self.model_class.metric_number)
        self.format_dict_length = self.model_class.format_dict_length
        self.report_sum = [0] * self.format_dict_length
    
        desc = '  - (Training)   '
        step_range = range(1, self.opt.n_training_steps + 1)
        training = cycle(iter(self.training_data))
        self.sched_optimizer.zero_grad()

        '''
        Start training.
        '''
        for current_step in tqdm(step_range, desc=desc, leave=False):
            data = next(training)
            step_result = self.model_class.train_step(self.model, data, device = self.opt.device)
            if current_step % self.opt.agg_update_step == 0:
                if self.opt.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt.grad_clip)
                self.sched_optimizer.step_and_update_lr()
                self.sched_optimizer.zero_grad()
    
            self.report_sum = lst_add_lst(self.report_sum, lst_divide(step_result, self.opt.n_report_steps))

            '''
            A short report about training.
            '''
            if current_step % self.opt.n_report_steps == 0 and self.rank == 0:
                self.train_report(current_step)
            
            '''
            A short report about evaluation and testing.
            '''
            if current_step % self.opt.n_evaluation_steps == 0:
                self.evaluation_report(current_step)
                        
        if self.rank == 0:
            logger.warning('Training finished!')
            if self.opt.wandb:
                wandb.finish()

    def create_file_logger(self):
        self.log_folder = 'log_' + self.folder_suffix
        if not os.path.exists(os.path.join(self.opt.log, self.log_folder)):
            os.mkdir(os.path.join(self.opt.log, self.log_folder))
        log_train_file = os.path.join(self.opt.log, self.log_folder, 'train.log')
        log_eva_file = os.path.join(self.opt.log, self.log_folder, 'evaluate.log')
        log_test_file = os.path.join(self.opt.log, self.log_folder, 'test.log')
        log_best_model_file = os.path.join(self.opt.save_model, 'output_' + self.folder_suffix, 'checkpoint.log')

        logger.info(f'Training performance will be written to file: \n{log_train_file},\n{log_eva_file},\n{log_test_file}')
        # These log_items defined here should match corresponding logger's print() method.
        file_logger = FileLogger(self.model_class.logfile_format, training_log = log_train_file, evaluation_log = log_eva_file, test_log = log_test_file)
        metric_format_dict = {
            **{'step': ''},
            **dict(zip([f'metric_{metric_count}' for metric_count in range(1, self.model_class.metric_number + 1)], \
                         [':8.5f'] * self.model_class.metric_number))
        }
        best_model_logger = FileLogger(metric_format_dict, best_model = log_best_model_file)

        return file_logger, best_model_logger

    def train_report(self, current_step):
        logger.warning(f'Brief training status report at step {current_step}.')
        report_sum = self.model_class.postprocess(self.report_sum, procedure = 'Training')
        print_performances(logger = logger, procedure='Training', lr = self.sched_optimizer.get_lr(), \
                           **(self.model_class.log_print_format(report_sum, procedure = 'Training')))
        if self.opt.wandb:
            import wandb
            wandb.log(
                add_prefix_to_keys(self.model_class.log_print_format(report_sum, \
                    procedure = 'Training'), temp = 'train_'), commit = False, step = current_step)
            wandb.log({'lr': self.sched_optimizer.get_lr()}, step = current_step)
        if self.rank == 0 and self.file_logger:
            report = self.model_class.logfile_print_format(report_sum)
            self.file_logger.print(logger_name = 'training_log', step = current_step, **report)
        self.report_sum = [0] * self.format_dict_length

    def evaluation_report(self, current_step):
        if self.rank == 0:
            logger.warning(f'Model evaluation and checkpoint saving at step {current_step}.')

        eva_report = self.model_class.postprocess(
            evaluation(self.evaluation_data, self.model, self.model_class, device = self.opt.device, \
                       output_length = self.format_dict_length, desc = '  - (Evaluation)   '), procedure = 'Evaluation'
        )
        test_report = self.model_class.postprocess(
            evaluation(self.test_data, self.model, self.model_class, device = self.opt.device, \
                       output_length = self.format_dict_length, desc = '  - (Test)   '), procedure = 'Test'
        )

        if self.rank == 0:
            print_performances(logger = logger, procedure='Evaluation', lr = self.sched_optimizer.get_lr(), \
                               **(self.model_class.log_print_format(eva_report, procedure = 'Evaluation')))
            print_performances(logger = logger, procedure='Test', lr = self.sched_optimizer.get_lr(), \
                               **(self.model_class.log_print_format(test_report, procedure = 'Test')))
            if self.opt.wandb:
                import wandb
                wandb.log(add_prefix_to_keys(self.model_class.log_print_format(eva_report, \
                    procedure = 'Evaluation'), temp = 'evaluation_'), commit = False, step = current_step)
                wandb.log(add_prefix_to_keys(self.model_class.log_print_format(test_report, \
                    procedure = 'Test'), temp = 'test_'), step = current_step)
        
            self.save(current_step, eva_report, test_report)

            if self.file_logger:
                eva = self.model_class.logfile_print_format(eva_report)
                test = self.model_class.logfile_print_format(test_report)
                self.file_logger.print(logger_name = 'evaluation_log', step = current_step, **eva)
                self.file_logger.print(logger_name = 'test_log', step = current_step, **test)

    def save(self, current_step, eva_report, test_report):
        # We will store the checkpoint after model evaluation.

        checkpoint = {'step': current_step, 'settings': self.opt, 'model': self.model.module.state_dict(),
                      'optimizer': self.sched_optimizer.state_dict()}

        # if self.opt.save_model and current_step > self.opt.n_warmup_steps:
        if self.opt.save_model:
            if self.opt.save_mode == 'all':
                model_name = os.path.join(
                        self.opt.save_model, 'output_' + self.folder_suffix, (f'checkpoint_training_step_{current_step}' + '.chkpt'))
                torch.save(checkpoint, model_name)
            elif self.opt.save_mode == 'best':
                model_name = os.path.join(self.opt.save_model, 'output_' + self.folder_suffix, 'checkpoint.chkpt')
                if current_step > self.opt.n_warmup_steps and self.metric_checker.compare(self.model_class.choose_metric(eva_report, test_report)):
                    torch.save(checkpoint, model_name)
                    logger.info('  The checkpoint file has been updated.')
                    best_model_dict = dict(zip(
                        [f'metric_{metric_count}' for metric_count in range(1, self.model_class.metric_number + 1)], \
                        self.model_class.choose_metric(eva_report, test_report)
                        ))
                    self.best_model_logger.print(logger_name = 'best_model', step = current_step, **best_model_dict)