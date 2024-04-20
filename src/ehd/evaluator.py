import os, torch
from torch.nn import DataParallel as DP

from src.taskhost_utils import get_logger
from src.ehd.utils import read_yaml, print_args
from src.ehd.evaluator_evaluation_functions import draw, spearman_and_l1, mae_and_f1, mae_e_and_f1, label_of_all_events
from src.ehd.model import get_model
from src.ehd.dataloader import prepare_dataloaders


'''
Detailed training procedure after all required data are ready.
Define the logger.
'''
logger = get_logger(__name__)


class ehdEvaluator:
    def __init__(self):
        '''
        Now, we use pd.DataFrame to record training records.
        '''
        pass


    def work(self, opt):
        '''
        Store required initial information
        '''
        self.opt = opt

        '''
        ========= Load Dataset =========
        '''
        if self.opt.data_path:
            self.training_data, self.evaluation_data, self.test_data = prepare_dataloaders(opt)
        else:
            raise logger.exception("Wrong input data path.")
    
        model_param = read_yaml(self.opt.abs_model_config) if self.opt.abs_model_config else {}
        self.param_names = list(model_param.keys())
        logger.info(f'The input model hyperparameters are {model_param}')

        '''
        ========= Restore Model from the checkpoint =========
        '''

        logger.info(f'Choosed model checkpoint file is in directory {self.opt.checkpoint_folder}.')
        self.model_class = get_model(self.opt)
        model = self.model_class(device = self.opt.device, training = False, opt = self.opt, **model_param)

        self.opt.__dict__.update(model_param)
        '''
        Here, we need to 1. restore the model weights from the checkpoint, 2. convert it into a DDP.
        '''
        model_raw = torch.load(os.path.join(self.opt.checkpoint_folder, 'checkpoint.chkpt'), map_location=opt.device)
        model_state_dict = model_raw['model']
        missing_keys, unexpected_keys = model.load_state_dict(model_state_dict)
        model.requires_grad_(requires_grad = False)
        logger.info(print_args(self.opt))
        trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        self.opt.trainable_parameters = trainable_parameters
        logger.info(f'Model restore completed. The number of trainable parameters in this model: {trainable_parameters} out of {total_params}.')
        
        self.model = DP(model)
        
        self.task()
    

    def task(self):
        task_dict = {
            'best':{
            'graph': self.task_graph,
            'spearman_and_l1': self.task_spearman_and_l1,
            'mae_and_f1': self.task_mae_and_f1,
            'mae_e_and_f1': self.task_mae_e_and_f1,
            'label_of_all_events': self.task_label_of_all_events
        },
        'all':{
            'sample': self.task_sample,
        }
        }

        return task_dict[self.opt.save_mode][self.opt.task_name]()
    

    def task_graph(self):
        # We will get three records from the training set, test set, and evaluation set, respectively.
        if self.opt.train:
            for idx, train_data in enumerate(self.training_data):
                draw(self.model, train_data, 'train', batch_idx = idx, opt = self.opt)
                if idx >= self.opt.figure_count - 1:
                    break

        if self.opt.evaluation:
            for idx, evaluation_data in enumerate(self.evaluation_data):
                draw(self.model, evaluation_data, 'evaluation', batch_idx = idx, opt = self.opt)
                if idx >= self.opt.figure_count - 1:
                    break

        if self.opt.test:
            for idx, test_data in enumerate(self.test_data):
                draw(self.model, test_data, 'test', batch_idx = idx, opt = self.opt)
                if idx >= self.opt.figure_count - 1:
                    break


    def task_spearman_and_l1(self):
        # We will get three records from the training set, test set, and evaluation set, respectively.
        if self.opt.train:
            spearman_and_l1(self.model, self.training_data, 'train', opt = self.opt)

        if self.opt.evaluation:
            spearman_and_l1(self.model, self.evaluation_data, 'evaluation', opt = self.opt)

        if self.opt.test:
            spearman_and_l1(self.model, self.test_data, 'test', opt = self.opt)


    def task_mae_and_f1(self):
        # We will get three records from the training set, test set, and evaluation set, respectively.
        if self.opt.train:
            mae_and_f1(self.model, self.training_data, 'train', opt = self.opt)

        if self.opt.evaluation:
            mae_and_f1(self.model, self.evaluation_data, 'evaluation', opt = self.opt)

        if self.opt.test:
            mae_and_f1(self.model, self.test_data, 'test', opt = self.opt)


    def task_mae_e_and_f1(self):
        # We will get three records from the training set, test set, and evaluation set, respectively.
        if self.opt.train:
            mae_e_and_f1(self.model, self.training_data, 'train', opt = self.opt)

        if self.opt.evaluation:
            mae_e_and_f1(self.model, self.evaluation_data, 'evaluation', opt = self.opt)

        if self.opt.test:
            mae_e_and_f1(self.model, self.test_data, 'test', opt = self.opt)


    def task_label_of_all_events(self):
        # We will get three records from the training set, test set, and evaluation set, respectively.
        if self.opt.train:
            label_of_all_events(self.model, self.training_data, 'train', opt = self.opt)

        if self.opt.evaluation:
            label_of_all_events(self.model, self.evaluation_data, 'evaluation', opt = self.opt)

        if self.opt.test:
            label_of_all_events(self.model, self.test_data, 'test', opt = self.opt)


    def task_sample(self):
        pass