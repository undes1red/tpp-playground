import os, torch
from tqdm import tqdm
import pandas as pd
from itertools import cycle
from torch.nn.parallel import DistributedDataParallel as DDP

from src.TPP.utils import print_performances, suffix, lst_add_lst, read_json, \
                          lst_divide, evaluation, Metric, add_prefix_to_keys, \
                          print_args, getLogger
from src.TPP.tpp_plotter import draw
from src.TPP.model import get_model
from src.TPP.dataloader import prepare_dataloaders


'''
Detailed training procedure after all required data are ready.
Define the logger.
'''
logger = getLogger(__name__)


class TPPPlotter:
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


    def work(self, rank, opt):
        '''
        Store required initial information
        '''
        self.opt = opt
        self.rank = rank

        '''
        ========= Load Dataset =========
        '''
        if self.opt.data_path:
            self.training_data, self.evaluation_data, self.test_data = prepare_dataloaders(opt, rank = rank)
        else:
            raise logger.exception("Wrong input data path.")
    
        model_param = read_json(self.opt.abs_model_config) if self.opt.abs_model_config else {}
        self.param_names = list(model_param.keys())
        if rank == 0:
            logger.info(f'The input model hyperparameters are {model_param}')
        
        '''
        ========= Restore Model from the checkpoint =========
        '''

        logger.info(f'Choosed model checkpoint file is in directory {self.opt.checkpoint_folder}.')
        self.model_class = get_model(self.opt.model_name, rank = rank)
        model = self.model_class(device = self.opt.device, num_events = self.opt.num_events,
            **model_param
        )
        model.eval()

        self.opt.__dict__.update(model_param)

        if rank == 0:
            logger.info(print_args(self.opt))
        
        '''
        Here, we need to 1. restore the model weights from the checkpoint, 2. convert it into a DDP.
        '''
        if rank == 0:
            model_raw = torch.load(os.path.join(self.opt.checkpoint_folder, 'checkpoint.chkpt'), map_location=opt.device)
            model_state_dict = model_raw['model']
            model_setting = model_raw['settings']
            model.load_state_dict(model_state_dict)
            logger.info(print_args(self.opt))
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f'Model restore completed. The number of trainable parameters in this model: {total_params}.')


        self.model = DDP(model, device_ids = [rank] if opt.cuda else None, find_unused_parameters = True)
        self.task()
    

    def task(self):
        task_dict = {
            'best':{
            'graph': self.task_graph,
            'spearman_and_f1': self.task_spearman_and_f1,
            'mae': self.task_mae,
            'mae_e': self.task_mae_e
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
                draw(self.model.module, train_data, 'train', batch_idx = idx, opt = self.opt)
                if idx >= self.opt.figure_count - 1:
                    break

        if self.opt.evaluation:
            for idx, evaluation_data in enumerate(self.evaluation_data):
                draw(self.model.module, evaluation_data, 'evaluation', batch_idx = idx, opt = self.opt)
                if idx >= self.opt.figure_count - 1:
                    break

        if self.opt.test:
            for idx, test_data in enumerate(self.test_data):
                draw(self.model.module, test_data, 'test', batch_idx = idx, opt = self.opt)
                if idx >= self.opt.figure_count - 1:
                    break


    def task_spearman_and_f1(self):
        pass


    def task_mae(self):
        pass

    def task_mae_e(self):
        pass


    def task_sample(self):
        pass