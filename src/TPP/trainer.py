import os, torch, yaml, io, copy
from tqdm import tqdm
import pandas as pd
from itertools import cycle
from torch.nn import DataParallel as DP

from src.taskhost_utils import getLogger
from src.TPP.utils import print_performances, suffix, lst_add_lst, read_yaml, \
                          lst_divide, evaluation, Metric, add_prefix_to_keys, \
                          print_args
from src.TPP.model import get_model
from src.TPP.optimizer.optim import ScheduledOptim
from src.TPP.dataloader import prepare_dataloaders


logger = getLogger(__name__)


class TPPTrainer:
    def __init__(self):
        '''
        Now, we use pd.DataFrame to record training records.
        '''
        self.df_records = {
            'Training': None,
            'Evaluation': None,
            'Test': None,
            'Best': None
        }


    def get_procedure_monitor_dict(self):
        return {
            'num_format': {'lr': ':8.5f', 'tensor_memory_consumption': ':5f', 'reserved_memory': ':5f'},
            'suffix': {'lr': '', 'tensor_memory_consumption': 'MiB', 'reserved_memory': 'MiB'},
            'lr': self.sched_optimizer.get_lr(),
            'tensor_memory_consumption': torch.cuda.memory_allocated(self.opt.device) / 1024 / 1024,
            'reserved_memory': torch.cuda.memory_reserved(self.opt.device) / 1024 / 1024,
        }


    def work(self, opt):
        '''
        The entry function for TaskHost to start the task.
        
        Args:
        * opt : namespace
                This namespace stores all parsed arguments.
        '''

        # Store required initial information.
        self.opt = opt

        '''
        We try to check if models and logs are saved and give some hints if you don't store any models or logs(most time you should store them).
        '''
        if not self.opt.log and not self.opt.save_model:
            logger.warning('No experiment result will be saved. If it is not intended, please check your training script.')


        '''
        ========= Load Dataset =========
        '''
        if self.opt.data_path:
            self.training_data, self.evaluation_data, self.test_data = prepare_dataloaders(opt)
            self.opt.training_size = len(self.training_data)
        else:
            raise logger.exception("Wrong input data path.")
    
        model_param = read_yaml(self.opt.abs_model_config) if self.opt.abs_model_config else {}
        self.param_names = list(model_param.keys())
        opt.model_params = model_param
        logger.info(f'The input model hyperparameters are {model_param}')
        
        '''
        Load model
        '''
        self.model_class = get_model(self.opt)
        model = self.model_class(device = self.opt.device, info_dict = self.opt.info_dict,
            **model_param
        )
    
        self.opt.__dict__.update(model_param)

        trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_parameters = sum(p.numel() for p in model.parameters())
        self.opt.trainable_parameters = trainable_parameters
        self.opt.epoch = opt.n_training_steps/opt.training_size
        logger.info(print_args(self.opt))
        logger.info(f'For someone who needs the number of training epoches, the number is {self.opt.epoch:5.5f}')
        logger.info(f'The number of trainable model parameters is {self.opt.trainable_parameters} out of {total_parameters}.')
    
        '''
        Due to the complexity of learning rate scheduler, the scheduler is fixed. 
        If you want to use another learning rate scheduler, plz modify it in src.optim.
        '''
        self.sched_optimizer = ScheduledOptim(opt, model)
        self.model = DP(model)
        self.task()
    
    
    def task(self):
        '''
        Directory preparation
        
        Create log and model-saving dirs if they are not present.
        '''
        self.folder_suffix = suffix(self.opt, 'model_name', 'lr', 'training_batch_size', 'n_training_steps', 'dataloader_config', 'model_config')
        self.output_checkpoint_folder = 'model_' + self.folder_suffix
        self.log_folder = 'log_' + self.folder_suffix
        if not os.path.exists(os.path.join(self.opt.save_model, self.output_checkpoint_folder)):
            os.makedirs(os.path.join(self.opt.save_model, self.output_checkpoint_folder))
        if not os.path.exists(os.path.join(self.opt.log, self.log_folder)):
            os.makedirs(os.path.join(self.opt.log, self.log_folder))

        '''
        Write hyperparameters into the model dir.
        '''
        with io.open(os.path.join(self.opt.save_model, self.output_checkpoint_folder, 'model_card.yml'), 'w', encoding = 'utf8') as f_hyperparameters:
            hyperparameters = copy.deepcopy(vars(self.opt))
            del hyperparameters['device']
            logger.debug(hyperparameters)
            yaml.safe_dump(hyperparameters, f_hyperparameters, default_flow_style = False, allow_unicode = True)

        '''
        Setting up file loggers and a wandb online logger.
        '''
        if self.opt.log:
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
        self.metric_checker = Metric(self.model_class.metric_number, getattr(self.model_class, 'smaller_is_better', None))
        self.format_dict_length = self.model_class.format_dict_length
        self.report_sum = [0] * self.format_dict_length
    
        desc = '  - (Training)   '
        step_range = range(1, self.opt.n_training_steps + 1)
        training = cycle(iter(self.training_data))
        self.sched_optimizer.zero_grad()

        '''
        Start training.
        '''
        self.evaluation_report(0)
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
            if current_step % self.opt.n_report_steps == 0:
                self.train_report(current_step)
            
            '''
            A short report about evaluation and testing.
            '''
            if current_step % self.opt.n_evaluation_steps == 0:
                self.evaluation_report(current_step)
                        
        if self.opt.log:
            for key, value in self.df_records.items():
                if value is None:
                    logger.warning(f'You require us to track the {key} process, but nothing is recorded!')
                    continue

                if key == 'Best':
                    log_filepath = os.path.join(self.opt.save_model, self.output_checkpoint_folder, 'checkpoint.csv')
                else:
                    log_filepath = os.path.join(self.opt.log, self.log_folder, f'{key}_record.csv')
                logger.info(f'Logs of {key} process are stored in {log_filepath}.')
                df_value = pd.DataFrame.from_dict(value)
                df_value.to_csv(log_filepath, index = False)

            logger.warning('Training finished!')
            if self.opt.wandb:
                wandb.finish()


    def train_report(self, current_step):
        logger.warning(f'Brief training status report at step {current_step}.')
        report_sum = self.model_class.postprocess(self.report_sum, procedure = 'Training')
        log_print_format_dict = self.model_class.log_print_format(report_sum, procedure = 'Training')
        if self.opt.log:
            self.transform_report_sum_into_recording_df(**log_print_format_dict, procedure = 'Training', current_step = current_step)
        procedure_monitor_dict = self.get_procedure_monitor_dict()
        print_performances(logger = logger, procedure = 'Training', \
                           model_performance_dict = log_print_format_dict,
                           procedure_monitor_dict = procedure_monitor_dict)

        if self.opt.wandb:
            import wandb
            wandb.log(
                add_prefix_to_keys(self.model_class.log_print_format(report_sum, \
                    procedure = 'Training'), temp = 'train_'), commit = False, step = current_step)
            wandb.log({'lr': self.sched_optimizer.get_lr()}, step = current_step)
        self.report_sum = [0] * self.format_dict_length


    def evaluation_report(self, current_step):
        logger.warning(f'Model evaluation and checkpoint saving at step {current_step}.')

        '''
        Evaluation on the dev dataset.
        '''
        eva_report = self.model_class.postprocess(
            evaluation(self.evaluation_data, self.model, self.model_class, device = self.opt.device, \
                       output_length = self.format_dict_length, desc = '  - (Evaluation)   '), procedure = 'Evaluation'
        )
        log_print_format_dict_eva = self.model_class.log_print_format(eva_report, procedure = 'Evaluation')
        procedure_monitor_dict = self.get_procedure_monitor_dict()
        print_performances(logger = logger, procedure = 'Evaluation', \
                           model_performance_dict = log_print_format_dict_eva,
                           procedure_monitor_dict = procedure_monitor_dict)

        '''
        Evaluation on the test dataset.
        '''
        test_report = self.model_class.postprocess(
            evaluation(self.test_data, self.model, self.model_class, device = self.opt.device, \
                       output_length = self.format_dict_length, desc = '  - (Test)   '), procedure = 'Test'
        )
        log_print_format_dict_test = self.model_class.log_print_format(test_report, procedure = 'Test')
        procedure_monitor_dict = self.get_procedure_monitor_dict()
        print_performances(logger = logger, procedure = 'Test', \
                           model_performance_dict = log_print_format_dict_test,
                           procedure_monitor_dict = procedure_monitor_dict)

        if self.opt.log:
            self.transform_report_sum_into_recording_df(**log_print_format_dict_eva, procedure = 'Evaluation', current_step = current_step)
            self.transform_report_sum_into_recording_df(**log_print_format_dict_test, procedure = 'Test', current_step = current_step)
        if self.opt.wandb:
            import wandb
            wandb.log(add_prefix_to_keys(self.model_class.log_print_format(eva_report, \
                procedure = 'Evaluation'), temp = 'evaluation_'), commit = False, step = current_step)
            wandb.log(add_prefix_to_keys(self.model_class.log_print_format(test_report, \
                procedure = 'Test'), temp = 'test_'), step = current_step)
        
        self.save(current_step, log_print_format_dict_eva, log_print_format_dict_test)


    def save(self, current_step, eva_report_format_dict, test_report_format_dict):
        # We will store the checkpoint after model evaluation.
        checkpoint = {'step': current_step, 'settings': self.opt, 'model': self.model.module.state_dict(),
                      'optimizer': self.sched_optimizer.state_dict()}

        if self.opt.save_mode == 'all':
            model_name = os.path.join(self.opt.save_model, 'model_' + self.folder_suffix, \
                                      (f'checkpoint_at_step_{current_step}' + '.chkpt'))
            torch.save(checkpoint, model_name)
            logger.warning(f'----> The checkpoint file at step {current_step} has been stored. <----')
            metric_values, metric_names = self.model_class.choose_metric(eva_report_format_dict, test_report_format_dict)
            self.transform_report_sum_into_recording_df(num_format = {}, procedure = 'Best', current_step = current_step,\
                                                        **dict(zip(metric_names, metric_values)))
        elif self.opt.save_mode == 'best':
            model_name = os.path.join(self.opt.save_model, 'model_' + self.folder_suffix, 'checkpoint.chkpt')
            metric_values, metric_names = self.model_class.choose_metric(eva_report_format_dict, test_report_format_dict)
            assert len(metric_values) == len(metric_names), "metric_values mismatches metric_names!"
            if current_step >= self.opt.n_warmup_steps and self.metric_checker.compare(metric_values):
                torch.save(checkpoint, model_name)
                logger.warning(f'----> We have updated the model checkpoint at step {current_step}. <----')
                self.transform_report_sum_into_recording_df(num_format = {}, procedure = 'Best', current_step = current_step,\
                                                            **dict(zip(metric_names, metric_values)))


    def transform_report_sum_into_recording_df(self, num_format, procedure, current_step, **kwargs):
        df_perline = kwargs
        new_df_perline_dict = {'current_step': current_step}
        new_df_perline_dict.update(df_perline)

        if self.df_records[procedure] is None:
            empty_execution_log_dict = {}
            for key in new_df_perline_dict.keys():
                empty_execution_log_dict[key] = []
            self.df_records[procedure] = empty_execution_log_dict
        
        for key in self.df_records[procedure].keys():
            self.df_records[procedure][key].append(new_df_perline_dict[key])