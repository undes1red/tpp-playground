import os, torch, yaml, io, copy
from tqdm import tqdm
import pandas as pd
from itertools import cycle
from torch.nn import DataParallel as DP
from torch.utils.flop_counter import FlopCounterMode

from src.toolbox.misc import get_logger, mkdir_if_not_exist
from src.toolbox.optimizer import ScheduledOptim

from src.TPP.utils import print_performances, suffix, lst_add_lst, read_yaml, only_keep_data, \
                          lst_divide, get_evaluation_results, Metric, print_args, pack_one_value_to_dict
from src.TPP.model import get_model
from src.TPP.dataloader import prepare_dataloaders


logger = get_logger(__name__)


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


    def get_procedure_monitor_dict(self, additional_info = {}):
        monitored_info = {'lr': pack_one_value_to_dict(self.sched_optimizer.get_lr(), '8.5f'),
                          'tensor_memory_consumption': pack_one_value_to_dict(torch.cuda.memory_allocated(self.opt.device) / 1024 / 1024 if self.opt.cuda else 0, '5f', 'MiB'),
                          'reserved_memory': pack_one_value_to_dict(torch.cuda.memory_reserved(self.opt.device) / 1024 / 1024 if self.opt.cuda else 0, '5f', 'MiB')
                         }
        for key, value in additional_info.items():
            monitored_info[key] = value
        
        return monitored_info


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
            self.raw_data = prepare_dataloaders(opt)
            self.opt.training_size = len(self.raw_data['Training'])
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
        self.model = self.model_class(device = self.opt.device, info_dict = self.opt.info_dict,
            **model_param
        )
        self.opt.__dict__.update(model_param)

        trainable_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_parameters = sum(p.numel() for p in self.model.parameters())
        self.opt.trainable_parameters = trainable_parameters
        self.opt.epoch = opt.n_training_steps/opt.training_size
        logger.info(print_args(self.opt))
        logger.info(f'For someone who needs the number of training epoches, the number is {self.opt.epoch:5.5f}')
        logger.info(f'The number of trainable model parameters is {self.opt.trainable_parameters} out of {total_parameters}.')
    
        '''
        Due to the complexity of learning rate scheduler, the scheduler is fixed. 
        If you want to use another learning rate scheduler, plz modify it in src.optim.
        '''
        self.sched_optimizer = ScheduledOptim(opt, self.model)

        if self.opt.cuda:
            self.model = DP(self.model, device_ids = [self.opt.cuda_device, ])

        self.model = torch.compile(self.model, \
                                   options = {'triton.cudagraphs': True, 'max_autotune': True, 'shape_padding': True, 'epilogue_fusion': True}, \
                                   disable = True)

        self.task()
    
    
    def task(self):
        '''
        Directory preparation
        
        Create log and model-saving dirs if they are not present.
        '''
        self.folder_suffix = suffix(self.opt, 'model_name', 'lr', 'training_batch_size', 'n_training_steps', 'dataloader_config', 'model_config')
        self.output_checkpoint_folder = 'model_' + self.folder_suffix
        self.log_folder = 'log_' + self.folder_suffix

        mkdir_if_not_exist(os.path.join(self.opt.save_model, self.output_checkpoint_folder))
        mkdir_if_not_exist(os.path.join(self.opt.log, self.log_folder))

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
        if self.opt.log and self.opt.wandb:
            import wandb
            wandb.init(project = 'Marked Temporal Point Process Training', \
                       config = vars(self.opt), group = self.opt.dataset_name, \
                       name = '-'.join([self.opt.model_name, str(self.opt.model_config), \
                                        self.opt.dataset_name, str(self.opt.dataloader_config)]), \
                       dir = os.path.join(self.opt.log, self.log_folder), \
                       resume = 'never',
                       notes = f'Training {self.opt.model_name} with config file {str(self.opt.model_config)} on dataset {self.opt.dataset_name}.'
                       )
            wandb.watch(self.model, log = 'all', log_freq = self.opt.n_report_steps, log_graph = True)
    
        '''
        Metric checker for choosing the best model during training.
        '''
        self.metric_checker = Metric(self.model_class.metric_number, getattr(self.model_class, 'smaller_is_better', None))
        self.format_dict_length = self.model_class.format_dict_length
        self.report_sum = [0] * self.format_dict_length
        self.training_flop = 0
    
        desc = '  - (Training)   '
        step_range = range(1, self.opt.n_training_steps + 1)
        training_iter = cycle(iter(self.raw_data['Training']))
        self.sched_optimizer.zero_grad()

        '''
        Start training.
        '''
        self.evaluation_report(0)
        for current_step in tqdm(step_range, desc = desc, leave = False):
            data = next(training_iter)

            if self.opt.fpcounter:
                with FlopCounterMode(display = False) as counter:
                    step_result = self.model_class.train_step(self.model, data, device = self.opt.device)
                
                self.training_flop += sum(counter.flop_counts['Global'].values())
            else:
                step_result = self.model_class.train_step(self.model, data, device = self.opt.device)
                
                self.training_flop += 0

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

            if self.opt.wandb:
                wandb.finish()
        
        logger.warning('Training finished!')


    def train_report(self, current_step):
        logger.warning(f'Brief training status report at step {current_step}.')

        dict_flops = {'FLOPS': pack_one_value_to_dict(self.training_flop / 1000**4, '8.5f', 'TFlops')}
        report_sum = self.model_class.postprocess(self.report_sum, procedure = 'Training')
        
        log_print_format_dict = self.model_class.log_print_format(report_sum, procedure = 'Training')
        procedure_monitor_dict = self.get_procedure_monitor_dict(dict_flops)
        plain_training_results = only_keep_data(log_print_format_dict)

        log_print_format_dict.update(procedure_monitor_dict)
        print_performances(logger = logger, procedure = 'Training', data_dict = log_print_format_dict)
        
        if self.opt.log:
            self.transform_report_sum_into_recording_df(procedure = 'Training', current_step = current_step, data = plain_training_results)
            if self.opt.wandb:
                import wandb
                wandb.log({'Training': plain_training_results}, commit = False, step = current_step)
                wandb.log({'lr': self.sched_optimizer.get_lr()}, step = current_step)

        self.report_sum = [0] * self.format_dict_length
        self.training_flop = 0
    

    def evaluation(self, dataset_name, current_step):
        evaluation_results = get_evaluation_results(self.raw_data[dataset_name], self.model, self.model_class, device = self.opt.device, \
                                                    output_length = self.format_dict_length, desc = f'  - ({dataset_name})   ')
        # dict_flops = {'FLOPS': {'data': evaluation_results['flops'] / 1000**4, 'num_format': '8.5f', 'suffix': 'TFlops'}}
        report = self.model_class.postprocess(evaluation_results['results'], procedure = dataset_name)

        log_print_format_dict = self.model_class.log_print_format(report, procedure = dataset_name)
        procedure_monitor_dict = self.get_procedure_monitor_dict()
        plain_evaluation_results = only_keep_data(log_print_format_dict)

        log_print_format_dict.update(procedure_monitor_dict)
        print_performances(logger = logger, procedure = dataset_name, data_dict = log_print_format_dict)

        if self.opt.log:
            self.transform_report_sum_into_recording_df(procedure = dataset_name, current_step = current_step, data = plain_evaluation_results)
            if self.opt.wandb:
                import wandb
                wandb.log({dataset_name: plain_evaluation_results}, step = current_step)
        
        return plain_evaluation_results


    def evaluation_report(self, current_step):
        logger.warning(f'Model evaluation and checkpoint saving at step {current_step}.')

        '''
        Evaluation and checkpoint saving.
        '''
        evaluation_results = self.evaluation('Evaluation', current_step)
        test_results = self.evaluation('Test', current_step)
        self.save(current_step, evaluation_results, test_results)


    def should_we_save_model(self, mode, metric_data, current_step, warmup):
        def checker_for_mode_all(metric_data):
            return True

        def checker_for_mode_bests(metric_data):
            return self.metric_checker.compare(metric_data.values())
        
        dict_save_model_checkers_and_checkpoint_names = {
            'all': [checker_for_mode_all, f'checkpoint_at_step_{current_step}.chkpt'],
            'best': [checker_for_mode_bests, 'checkpoint.chkpt'],
            'last': [checker_for_mode_all, 'checkpoint.chkpt']
        }

        save_should_or_not = False
        checker, checkpoint_name = dict_save_model_checkers_and_checkpoint_names[mode]
        if current_step >= warmup and checker(metric_data):
            save_should_or_not = True

        return save_should_or_not, checkpoint_name


    def save(self, current_step, evaluation_results, test_results):
        # We will store the checkpoint after model evaluation.
        checkpoint = {'step': current_step, 'settings': self.opt, 'model': self.model.module.state_dict() if self.opt.cuda else self.model.state_dict(),
                      'optimizer': self.sched_optimizer.state_dict()}

        metric_values, metric_names = self.model_class.choose_metric(evaluation_results, test_results)
        assert len(metric_values) == len(metric_names), "metric_values mismatches metric_names!"
        metric_data = dict(zip(metric_names, metric_values))

        save_should_or_not, checkpoint_name \
            = self.should_we_save_model(mode = self.opt.save_mode, metric_data = metric_data, \
                                        current_step = current_step, warmup = self.opt.n_warmup_steps)

        if save_should_or_not:
            model_name = os.path.join(self.opt.save_model, 'model_' + self.folder_suffix, checkpoint_name)
            torch.save(checkpoint, model_name)
            self.transform_report_sum_into_recording_df(procedure = 'Best', current_step = current_step, data = metric_data)
            logger.warning(f'----> We stored the model in {checkpoint_name} at step {current_step}. <----')


    def transform_report_sum_into_recording_df(self, procedure, current_step, data):
        new_df_perline_dict = {'current_step': current_step}
        new_df_perline_dict.update(data)

        if self.df_records[procedure] is None:
            empty_execution_log_dict = {}
            for key in new_df_perline_dict.keys():
                empty_execution_log_dict[key] = []
            self.df_records[procedure] = empty_execution_log_dict
        
        for key in self.df_records[procedure].keys():
            self.df_records[procedure][key].append(new_df_perline_dict[key])