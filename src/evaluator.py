import os
from torch.nn import DataParallel as DP

from src.toolbox.misc import get_logger, read_yaml, print_args
from src.toolbox.evaluation import basic_evaluation_loop, basic_evaluation
from src.toolbox.evaluation import load_checkpoint, possible_checkpoint_detect
from src.toolbox.dataloader import prepare_dataloaders

'''
Detailed training procedure after all required data are ready.
Define the logger.
'''
logger = get_logger(__name__)


class Evaluator:
    def __init__(self, opt, procedure):
        # Store required initial information.
        self.opt = opt
        self.opt.replace_index = ['',] if opt.replace else possible_checkpoint_detect(opt, opt.model_identifier)

        # load the model.
        self.get_model = getattr(procedure, 'get_model')
        self.get_dataloader = getattr(procedure, 'get_dataloader')
        self.task_dict = getattr(procedure, 'get_evaluation_funcs')

        '''
        ========= Restore Model from the checkpoint =========
        '''
        self.checkpoint_folder_suffix = 'model_' + opt.model_identifier
        self.results_folder_suffix = 'results_' + opt.model_identifier


    def work(self):
        '''
        ========= Load Dataset =========
        '''
        if self.opt.data_path:
            self.raw_data = prepare_dataloaders(self.opt, self.get_dataloader)
        else:
            raise logger.exception("Wrong input data path.")
    
        model_param = read_yaml(self.opt.abs_model_config) if self.opt.abs_model_config else {}
        self.param_names = list(model_param.keys())
        logger.info(f'The input model hyperparameters are {model_param}.')
        self.model_class = self.get_model(self.opt)
        model = self.model_class(device = self.opt.device, info_dict = self.opt.info_dict,
            **model_param
        )
        self.opt.__dict__.update(model_param)
        if len(self.opt.replace_index) == 0:
            logger.warning('The evaluation exited because NO checkpoint has been found.')
            logger.warning('Perhaps, you have forgot the --replace in your script.')

        for index in self.opt.replace_index:
            # locate where checkpoints are stored.
            self.opt.checkpoint_folder = os.path.join(self.opt.checkpoint_of_this_procedure, str(index), self.opt.dataset_name, self.checkpoint_folder_suffix)
            # where figures, records are stored.
            self.opt.store_dir = os.path.join(self.opt.results_of_this_procedure, str(index), self.opt.dataset_name, self.results_folder_suffix)
            logger.info(f'We will load the model checkpoint in {self.opt.checkpoint_folder}.')
            logger.info(f'Results will be stored in {self.opt.store_dir}.')

            '''
            Here, we need to 1. restore the model weights from the checkpoint, 2. convert it into a DP if possible.
            '''
            model = load_checkpoint(logger, os.path.join(self.opt.checkpoint_folder, 'checkpoint.chkpt'), model, device = self.opt.device)
            logger.info(print_args(self.opt, 'Evaluation Info'))

            if self.opt.cuda:
                self.model = DP(model, device_ids = [self.opt.cuda_device, ])
            else:
                self.model = model
    
            # Fix module behaviours during evaluation.
            self.model.eval()
            self.task()
        
        
    def finish_task(self):
        pass


    def task(self):
        if self.opt.task_name == 'evaluation_dataset':
            return self.evaluation_on_entire_dataset(self.task_dict[self.opt.subtask_name])
        elif self.opt.task_name == 'evaluation_per_seq':
            return self.evaluation_per_seq()
        else:
            Exception('Unknown task.')
    

    def evaluation_per_seq(self):
        # We will get three records from the training set, test set, and evaluation set, respectively.
        if self.opt.training_data_name is not None:
            for idx, train_data in enumerate(self.raw_data['Training']):
                basic_evaluation(self.model, train_data, 'train', batch_idx = idx, opt = self.opt)
                if idx >= self.opt.figure_count - 1:
                    break
    
        if self.opt.evaluate_data_name is not None:
            for idx, evaluation_data in enumerate(self.raw_data['Evaluation']):
                basic_evaluation(self.model, evaluation_data, 'evaluation', batch_idx = idx, opt = self.opt)
                if idx >= self.opt.figure_count - 1:
                    break
    
        if self.opt.test_data_name is not None:
            for idx, test_data in enumerate(self.raw_data['Test']):
                basic_evaluation(self.model, test_data, 'test', batch_idx = idx, opt = self.opt)
                if idx >= self.opt.figure_count - 1:
                    break
    
    
    def evaluation_on_entire_dataset(self, evaluation_func):
        if isinstance(evaluation_func, list):
            # We will get three records from the training set, test set, and evaluation set, respectively.
            if self.opt.training_data_name is not None:
                basic_evaluation_loop(self.model, self.raw_data['Training'], 'train', opt = self.opt, early_offload = False, *evaluation_func)
        
            if self.opt.evaluate_data_name is not None:
                basic_evaluation_loop(self.model, self.raw_data['Evaluation'], 'evaluation', opt = self.opt, early_offload = False, *evaluation_func)
        
            if self.opt.test_data_name is not None:
                basic_evaluation_loop(self.model, self.raw_data['Test'], 'test', opt = self.opt, early_offload = True, *evaluation_func)
        elif isinstance(evaluation_func, dict):
            # We will get three records from the training set, test set, and evaluation set, respectively.
            if self.opt.training_data_name is not None:
                basic_evaluation_loop(self.model, self.raw_data['Training'], 'train', opt = self.opt, early_offload = False, **evaluation_func)
        
            if self.opt.evaluate_data_name is not None:
                basic_evaluation_loop(self.model, self.raw_data['Evaluation'], 'evaluation', opt = self.opt, early_offload = False, **evaluation_func)
        
            if self.opt.test_data_name is not None:
                basic_evaluation_loop(self.model, self.raw_data['Test'], 'test', opt = self.opt, early_offload = True, **evaluation_func)
        elif isinstance(evaluation_func, function):
            # We will get three records from the training set, test set, and evaluation set, respectively.
            if self.opt.training_data_name is not None:
                evaluation_func(self.model, self.raw_data['Training'], 'train', opt = self.opt, early_offload = False)
        
            if self.opt.evaluate_data_name is not None:
                evaluation_func(self.model, self.raw_data['Evaluation'], 'evaluation', opt = self.opt, early_offload = False)
        
            if self.opt.test_data_name is not None:
                evaluation_func(self.model, self.raw_data['Test'], 'test', opt = self.opt, early_offload = True)
        else:
            raise Exception('Unknown evaluation func!')