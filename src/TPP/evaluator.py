import os, torch
from torch.nn import DataParallel as DP

from src.toolbox.misc import get_logger

from src.TPP.utils import read_yaml, print_args, suffix, load_checkpoint
from src.TPP.resources.evaluator_evaluation_functions import *
from src.TPP.model import get_model
from src.TPP.dataloader import prepare_dataloaders


'''
Detailed training procedure after all required data are ready.
Define the logger.
'''
logger = get_logger(__name__)


class TPPEvaluator:
    def work(self, opt):
        '''
        Store required initial information
        '''
        self.opt = opt

        '''
        ========= Load Dataset =========
        '''
        if self.opt.data_path:
            self.raw_data = prepare_dataloaders(opt)
        else:
            raise logger.exception("Wrong input data path.")
    
        model_param = read_yaml(self.opt.abs_model_config) if self.opt.abs_model_config else {}
        self.param_names = list(model_param.keys())
        logger.info(f'The input model hyperparameters are {model_param}.')
        self.model_class = get_model(self.opt)
        model = self.model_class(device = self.opt.device, info_dict = self.opt.info_dict,
            **model_param
        )
        self.opt.__dict__.update(model_param)
        if len(opt.replace_index) == 0:
            logger.warning('The evaluation exited because NO checkpoint has been found.')
            logger.warning('Perhaps, you have forgot the --replace in your script.')

        '''
        ========= Restore Model from the checkpoint =========
        '''
        model_hyperparameters = suffix(opt, 'model_name', 'lr', 'used_batch_size', 'n_training_steps', 'used_dataloader_config', 'model_config')
        checkpoint_folder_suffix = 'model_' + model_hyperparameters
        results_folder_suffix = 'results_' + model_hyperparameters
        for index in opt.replace_index:
            # locate where checkpoints are stored.
            opt.checkpoint_folder = os.path.join(opt.checkpoint_of_this_procedure, str(index), opt.dataset_name, checkpoint_folder_suffix)
            # where figures, records are stored.
            opt.store_dir = os.path.join(opt.results_of_this_procedure, str(index), opt.dataset_name, results_folder_suffix)
            logger.info(f'We will load the model checkpoint in {opt.checkpoint_folder}.')
            logger.info(f'Results will be stored in {opt.store_dir}.')

            '''
            Here, we need to 1. restore the model weights from the checkpoint, 2. convert it into a DP if possible.
            '''
            model = load_checkpoint(logger, os.path.join(self.opt.checkpoint_folder, 'checkpoint.chkpt'), model, device = opt.device)
            logger.info(print_args(self.opt))

            if self.opt.cuda:
                self.model = DP(model, device_ids = [self.opt.cuda_device, ])
            else:
                self.model = model
    
            # Fix module behaviours during evaluation.
            self.model.eval()
            self.task()


    def task(self):
        task_dict = {
        'best':{
            # Follwoing tasks involves a part of the dataset.
            'graph': self.task_graph,

            # Following tasks involves the entire dataset.
            'spearman_and_l1': self.task_evaluation_on_entire_dataset,
            'mae_and_f1': self.task_evaluation_on_entire_dataset,
            'mae_e_and_f1': self.task_evaluation_on_entire_dataset,
            'mae_e_and_f1_by_time_event': self.task_evaluation_on_entire_dataset,
            'which_event_occurs_first': self.task_evaluation_on_entire_dataset,
            'samples_from_et': self.task_evaluation_on_entire_dataset,

            'mae_and_f1_of_imputated_events': self.task_mae_and_f1_of_imputated_events},
        'all':{
            'sample': self.task_sample,}
        }

        return task_dict[self.opt.save_mode][self.opt.task_name]()
    

    def task_graph(self):
        # We will get three records from the training set, test set, and evaluation set, respectively.
        if self.opt.training_data_name is not None:
            for idx, train_data in enumerate(self.raw_data['Training']):
                draw(self.model, train_data, 'train', batch_idx = idx, opt = self.opt)
                if idx >= self.opt.figure_count - 1:
                    break

        if self.opt.evaluate_data_name is not None:
            for idx, evaluation_data in enumerate(self.raw_data['Evaluation']):
                draw(self.model, evaluation_data, 'evaluation', batch_idx = idx, opt = self.opt)
                if idx >= self.opt.figure_count - 1:
                    break

        if self.opt.test_data_name is not None:
            for idx, test_data in enumerate(self.raw_data['Test']):
                draw(self.model, test_data, 'test', batch_idx = idx, opt = self.opt)
                if idx >= self.opt.figure_count - 1:
                    break


    def task_evaluation_on_entire_dataset(self):
        # We will get three records from the training set, test set, and evaluation set, respectively.
        if self.opt.training_data_name is not None:
            basic_evaluation_loop(self.model, self.raw_data['Training'], 'train', opt = self.opt, early_offload = False)

        if self.opt.evaluate_data_name is not None:
            basic_evaluation_loop(self.model, self.raw_data['Evaluation'], 'evaluation', opt = self.opt, early_offload = False)

        if self.opt.test_data_name is not None:
            basic_evaluation_loop(self.model, self.raw_data['Test'], 'test', opt = self.opt, early_offload = True)


    def task_mae_and_f1_of_imputated_events(self):
        # We will get three records from the training set, test set, and evaluation set, respectively.
        if self.opt.training_data_name is not None:
            mae_and_f1_of_imputated_events(self.model, self.raw_data['Training'], 'train', opt = self.opt)

        if self.opt.evaluate_data_name is not None:
            mae_and_f1_of_imputated_events(self.model, self.raw_data['Evaluation'], 'evaluation', opt = self.opt)

        if self.opt.test_data_name is not None:
            mae_and_f1_of_imputated_events(self.model, self.raw_data['Test'], 'test', opt = self.opt)


    def task_sample(self):
        pass