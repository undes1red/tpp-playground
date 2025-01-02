import torch
import numpy as np
from sklearn.metrics import accuracy_score
from einops import rearrange, repeat, reduce, pack

from src.toolbox.metrics import otd
from src.toolbox.misc import pack_one_value_to_dict, easy_model_load

from src.LH.model.basic_tpp_model import BasicModel
from src.LH.model.hypro.submodel import HYPRO
from src.LH.model.utils import *


class CPPODWrapper(BasicModel):
    def __init__(self, opt, device, config_loaded_model):
        super(CPPODWrapper, self).__init__()
        self.device = device
        self.num_events = opt.info_dict['num_events']
        self.start_time = opt.info_dict['t_0']
        self.end_time = opt.info_dict['T']

        self.mtpp_model = easy_model_load('TPP', opt.root_path, device = self.device, **config_loaded_model)
        
    
    def forward(self, task_name, *args, **kwargs):

        task_mapper = {
            'train': self.train_procedure,
            'evaluate': self.evaluate_procedure,
            
            'cppod_evaluation': self.cppod_evaluation,
        }

        return task_mapper[task_name](*args, **kwargs)
    
    
    def train_procedure(self, time_seq, events_seq, mark, mask_seq, mean, std):
        pass
        
    
    def evaluate_procedure(self, time_seq, events_seq, mark, mask_seq, mean, std):
        pass
    
    
    def convert_missing_mask_to_gap_mask(missing_mask):
        # input shape: [num_samples, seq_len]
        
        masks = []
        for missing_mask_per_seq in missing_mask[1:]:
            current_in_missing = False
            mask_current_seq = []
            for item in missing_mask_per_seq:
                if item == 0 and not current_in_missing:
                    mask_current_seq.append(0)
                elif item == 0 and current_in_missing:
                    current_in_missing = False
                elif item == 1 and not current_in_missing:
                    mask_current_seq.append(1)
                    current_in_missing = True
                else:
                    continue
            
            masks.append(mask_current_seq)
        
        return masks


    def cppod_evaluation(self, input_data, opt):
        '''
        Take care. This function only evaluates the omission outlier.
        Interestingly, the original CPPOD code seems only focusing on omission too as only omission scores are recorded in model.detect_outlier().
        '''
        forward_complete_data, backward_complete_data, padded_obs_data, padded_backward_obs_event_seq, (mean, std) \
            = input_data
        
        for obs_time_for_one_seq, obs_events_for_one_seq, obs_mask_for_one_seq, missing_mask_for_one_seq, _ in padded_obs_data:
            missing_mask_for_one_seq = self.convert_missing_mask_to_gap_mask(missing_mask_for_one_seq)
                                                                               # [num_samples, ...]
            integral_all_events, intensity_all_events \
                = self.mtpp_model.model(obs_time_for_one_seq, obs_events_for_one_seq, obs_mask_for_one_seq)
                                                                               # [num_samples, seq_len, num_events]
            
            integral_sum = integral_all_events.sum(dim = -1)                   # [num_samples, seq_len]
            intensity_sum = intensity_all_events.sum(dim = -1)                 # [num_samples, seq_len]
            
            
            
            
    
    '''
    Static methods
    '''
    def train_step(model, minibatch, device):
        ''' Epoch operation in training phase'''
        pass
    

    def evaluation_step(model, minibatch, device):
        ''' Epoch operation in evaluation phase '''
        pass
    
    
    def postprocess(input, procedure):
        pass
    
    
    def log_print_format(input, procedure):
        pass

    format_dict_length = 0

    def choose_metric(evaluation_report_format_dict, test_report_format_dict):
        '''
        [relative loss on evaluation dataset, relative loss on test dataset, event loss on test dataset]
        '''
        return [], []

    metric_number = 0 # metric number is the length of the output of choose_metric
    smaller_is_better = []