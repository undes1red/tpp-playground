import torch, copy
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from einops import rearrange, repeat, reduce, pack
from scipy.stats import spearmanr

from src.toolbox.metrics import L1_distance_across_events
from src.toolbox.misc import pack_one_value_to_dict, compile_model

from src.OD.model.basic_tpp_model import BasicModel
from src.OD.model.llmtpp.submodel import LLMTPP
from src.OD.model.utils import *
from src.OD.model.llmtpp.plot import *


class LLMTPPModel(BasicModel):
    def __init__(self, opt, llm_class_name, full_llm_name, d_model, \
                 d_embedding, lm_layers, device, dropout, epsilon = 1e-20):
        super(LLMTPPModel, self).__init__()
        self.device = device
        self.num_events = opt.info_dict['num_events']
        self.start_time = opt.info_dict['t_0']
        self.end_time = opt.info_dict['T']
        self.epsilon = epsilon

        self.model = LLMTPP(llm_class_name = llm_class_name, full_llm_name = full_llm_name, \
                            num_events = self.num_events, d_model = d_model, d_embedding = d_embedding, \
                            lm_layers = lm_layers, dropout = dropout, device = device)
        
        self.model = compile_model(self.model, opt.compile)


    def divide_history_and_next(self, input):
        input_history, input_next = input[:, :-1].clone(), input[:, 1:].clone()
        return input_history, input_next


    def forward(self, task_name, *args, **kwargs):
        '''
        The entrance of the IFIB-C wrapper.
        
        Args:
        * input_time    type: torch.tensor shape: [batch_size, seq_len + 1]
                        The original time sequence. We should extract the history and target sequence from it
                        by divide_history_and_next().
        * input_events  type: torch.tensor shape: [batch_size, seq_len + 1]
                        The original event sequence. We should extract the history and target sequence from it
                        by divide_history_and_next().
        * mask          type: torch.tensor shape: [batch_size, seq_len + 1]
                        We use mask to mask out unneeded outputs.
        * mean          type: float shape: N/A
                        The mean of all $ t_i - t_{i - 1} $ in the entire dataset. Dataloader is responsible to provide
                        this value if needed.
        * std           type: float shape: N/A
                        The mean of all $ t_i - t_{i - 1} $ in the entire dataset. Dataloader is responsible to provide
                        this value if needed.
        * evaluate      type: bool shape: N/A
                        perform a model training step when evaluate == False
                        perform a model evaluate step when evaluate == True
        
        Outputs:
        Refers to train() and evaluate()'s documentation for detailed information.

        '''
        task_mapper = {
            'train': self.train_procedure,
            'evaluate': self.evaluate_procedure,
            'omission_roc_auc': self.omission_outlier,
        }

        return task_mapper[task_name](*args, **kwargs)
    

    def remove_dummy_event_from_mask(self, mask):
        '''
        Remove the probability of the dummy event by mask.
        '''
        mask_without_dummy = torch.zeros_like(mask)                            # [batch_size, seq_len - 1]
        for idx, mask_per_seq in enumerate(mask):
            dummy_index = mask_per_seq.sum() - 1
            mask_without_dummy_per_seq = copy.deepcopy(mask_per_seq.detach())
            mask_without_dummy_per_seq[dummy_index] = 0
            mask_without_dummy[idx] = mask_without_dummy_per_seq
        
        return mask_without_dummy


    def train_procedure(self, input_time, input_events, input_seq_mask, interval_has_missing, \
                        interval_has_missing_mask, mean, std):

        missing_score = self.model('train', input_time = input_time, input_events = input_events, \
                                            input_seq_mask = input_seq_mask, mean = mean, std = std)
                                                                               # [batch_size, seq_len, num_events] * 2
        loss = torch.nn.functional.binary_cross_entropy(missing_score, interval_has_missing.float(), reduction = 'none')
                                                                               # [batch_size, sample_size, seq_len]
        masked_loss = loss * interval_has_missing_mask
        loss = masked_loss.sum() / interval_has_missing_mask.sum()
    
        return loss


    def evaluate_procedure(self, input_time, input_events, input_seq_mask, interval_has_missing, \
                           interval_has_missing_mask, mean, std):
        
        missing_score = self.model('evaluate', input_time = input_time, input_events = input_events, \
                                   input_seq_mask = input_seq_mask, mean = mean, std = std)
                                                                               # [batch_size, sample_size, seq_len]
        loss = torch.nn.functional.binary_cross_entropy(missing_score, interval_has_missing.float(), reduction = 'none')
                                                                               # [batch_size, sample_size, seq_len]
        masked_loss = loss * interval_has_missing_mask
        loss = masked_loss.sum().item() / interval_has_missing_mask.sum().item()
        
        # Metric: roc_auc.
        packed_data = zip(missing_score, interval_has_missing, interval_has_missing_mask)
        all_roc_auc = []
        for (missing_score_per_batch, interval_has_missing_per_batch, interval_has_missing_mask_per_batch) \
            in packed_data:
            packed_data_per_batch = zip(missing_score_per_batch, interval_has_missing_per_batch, interval_has_missing_mask_per_batch)
            for missing_score_per_batch_per_seq, interval_has_missing_per_batch_per_seq, interval_has_missing_mask_per_batch_per_seq \
                in packed_data_per_batch:
                selected_missing_score = missing_score_per_batch_per_seq[interval_has_missing_mask_per_batch_per_seq == 1]
                selected_interval_has_missing = interval_has_missing_per_batch_per_seq[interval_has_missing_mask_per_batch_per_seq == 1]
                selected_interval_has_missing, selected_missing_score = move_from_tensor_to_ndarray(selected_interval_has_missing, selected_missing_score)
                all_roc_auc.append(roc_auc_score(y_true = selected_interval_has_missing, y_score = selected_missing_score))
        
        auroc = np.mean(all_roc_auc).item()

        return loss, auroc


    def loss(self, pred_time, time_next, event_next, mask_next):
        '''
        The definition of loss.
    
        Args:
            probability:        [batch_size, seq_len, num_events]
            events_next:        [batch_size, seq_len]
            mask_next:          [batch_size, seq_len]
        '''
        # pick the time.
        event_next_mask = torch.nn.functional.one_hot(event_next, num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]
        selected_pred_time = (pred_time * event_next_mask).sum(dim = -1)       # [batch_size, seq_len]
        gap = torch.abs(selected_pred_time - time_next)                        # [batch_size, seq_len]
        masked_gap = gap * mask_next                                           # [batch_size, seq_len]
        loss = torch.pow(masked_gap, 1)                                        # [batch_size, seq_len]
        loss = torch.sum(loss)

        return loss
    
    
    def omission_outlier(self, input_data, opt):
        padded_obs_time, padded_obs_event, padded_obs_mask, padded_interval_has_missing, padded_interval_has_missing_mask, \
            (mean, std) = input_data

        missing_score = self.model('evaluate', input_time = padded_obs_time, input_events = padded_obs_event, \
                                   input_seq_mask = padded_obs_mask, mean = mean, std = std)
                                                                               # [batch_size, sample_size, seq_len]
        # Metric: roc_auc.
        packed_data = zip(missing_score, padded_interval_has_missing, padded_interval_has_missing_mask)
        all_roc_auc = []
        for (missing_score_per_batch, interval_has_missing_per_batch, interval_has_missing_mask_per_batch) \
            in packed_data:
            packed_data_per_batch = zip(missing_score_per_batch, interval_has_missing_per_batch, interval_has_missing_mask_per_batch)
            for missing_score_per_batch_per_seq, interval_has_missing_per_batch_per_seq, interval_has_missing_mask_per_batch_per_seq \
                in packed_data_per_batch:
                selected_missing_score = missing_score_per_batch_per_seq[interval_has_missing_mask_per_batch_per_seq == 1]
                selected_interval_has_missing = interval_has_missing_per_batch_per_seq[interval_has_missing_mask_per_batch_per_seq == 1]
                selected_interval_has_missing, selected_missing_score = move_from_tensor_to_ndarray(selected_interval_has_missing, selected_missing_score)
                all_roc_auc.append(roc_auc_score(y_true = selected_interval_has_missing, y_score = selected_missing_score))
        
        auroc = np.mean(all_roc_auc).item()

        return auroc


    '''
    All static methods
    '''
    def train_step(model, minibatch, device):
        ''' 
        Epoch operation in training phase.
        The input minibatch comprise time sequences.

        Args:
            minibatch: [batch_size, seq_len]
                       contains [time_seq, event_seq, score, mask]
        '''
    
        model.train()
        padded_obs_time, padded_obs_event, padded_obs_mask, padded_interval_has_missing, padded_interval_has_missing_mask, \
            (mean, std) = minibatch
        
        '''
        Here, we need the complete forward data and the backward time.
        '''
        loss = model(task_name = 'train', \
                       input_time = padded_obs_time, \
                       input_events = padded_obs_event, \
                       input_seq_mask = padded_obs_mask, \
                       interval_has_missing = padded_interval_has_missing, \
                       interval_has_missing_mask = padded_interval_has_missing_mask, \
                       mean = mean, \
                       std = std)
        
        loss.backward()

        loss = loss.item()
        
        return loss
    

    def evaluation_step(model, minibatch, device):
        ''' Epoch operation in evaluation phase '''
    
        model.eval()
        padded_obs_time, padded_obs_event, padded_obs_mask, padded_interval_has_missing, padded_interval_has_missing_mask, \
            (mean, std) = minibatch

        loss, auroc = model(task_name = 'evaluate', \
                            input_time = padded_obs_time, \
                            input_events = padded_obs_event, \
                            input_seq_mask = padded_obs_mask, \
                            interval_has_missing = padded_interval_has_missing, \
                            interval_has_missing_mask = padded_interval_has_missing_mask, \
                            mean = mean, \
                            std = std)
            
        return loss, auroc


    def postprocess(input, procedure):
        def train_postprocess(input):
            '''
            Training process
            [absolute loss, relative loss, events loss]
            '''
            return [input[0],]
        
        def test_postprocess(input):
            '''
            Evaluation process
            [absolute loss, relative loss, events loss, mae value]
            '''
            return [input[0], input[1],]
        
        return (train_postprocess(input) if procedure == 'Training' else test_postprocess(input))
    

    def log_print_format(input, procedure):
        def train_log_print_format(input):
            format_dict = {}
            format_dict['loss'] = pack_one_value_to_dict(input[0])
            return format_dict

        def test_log_print_format(input):
            format_dict = {}
            format_dict['loss'] = pack_one_value_to_dict(input[0])
            format_dict['auroc'] = pack_one_value_to_dict(input[1])
            return format_dict
        
        return (train_log_print_format(input) if procedure == 'Training' else test_log_print_format(input))

    format_dict_length = 2
    
    def choose_metric(evaluation_report_format_dict, test_report_format_dict):
        '''
        [relative loss on evaluation dataset, relative loss on test dataset, event loss on test dataset]
        '''
        return [evaluation_report_format_dict['loss'], 
                test_report_format_dict['loss']], \
               ['evaluation_loss', 'test_loss']
    
    metric_number = 2 # metric number is the length of the output of choose_metric