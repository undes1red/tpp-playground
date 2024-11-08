import torch
import numpy as np
from sklearn.metrics import f1_score
from einops import rearrange, repeat, reduce, pack

from src.toolbox.metrics import otd
from src.toolbox.misc import pack_one_value_to_dict

from src.LH.model.basic_tpp_model import BasicModel
from src.LH.model.llmtpp.submodel import LLMTPP
from src.LH.model.utils import *
from src.LH.model.llmtpp.plot import *


class LLMTPPModel(BasicModel):
    def __init__(self, opt, llm_class_name, full_llm_name, d_model, \
                 d_embedding, lm_layers, device, dropout, epsilon = 1e-20, lambda_t = 1.0, lambda_e = 1.0, \
                 long_horizon_time = 1, otd_event_removal_cost = 1.0, move_event_cost = 1.0):
        super(LLMTPPModel, self).__init__()
        self.device = device
        self.num_events = opt.info_dict['num_events']
        self.start_time = opt.info_dict['t_0']
        self.end_time = opt.info_dict['T']
        self.lh_length = opt.info_dict['hypro_length']
        self.epsilon = epsilon
        self.lambda_t = lambda_t
        self.lambda_e = lambda_e
        self.long_horizon_time = long_horizon_time
        self.otd_event_removal_cost = otd_event_removal_cost
        self.otd_move_event_cost = move_event_cost

        self.model = LLMTPP(llm_class_name = llm_class_name, full_llm_name = full_llm_name, \
                            num_events = self.num_events, d_model = d_model, d_embedding = d_embedding, \
                            lm_layers = lm_layers, dropout = dropout, lh_length = self.lh_length, device = device)


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
            'long_horizon_pred': self.long_horizon_pred
        }
        '''
            'spearman_and_l1': self.get_spearman_and_l1,
            'mae_and_f1': self.get_mae_and_f1,
            'mae_e_and_f1': self.get_mae_e_and_f1,
            'intensity': self.intensity,
            'integral': self.integral,
            'probability': self.probability,
            'debug': self.debug
        '''

        return task_mapper[task_name](*args, **kwargs)


    def train_procedure(self, history_time_seq, history_events_seq, history_mask_seq, \
                        observed_time_seq, observed_events_seq, mean, std):
        '''
        Remove the probability of the dummy event by mask.
        '''
        pred_time, pred_events_prob = self.model('train', events_history = history_events_seq, time_history = history_time_seq, \
                                                 mask_history = history_mask_seq, mean = mean, std = std)
                                                                               # [batch_size, lh_length, num_events] * 2

        time_loss, events_loss = self.loss(pred_time = pred_time, observed_time = observed_time_seq, \
                                           pred_events_prob = pred_events_prob, observed_events = observed_events_seq)
        
        time_loss = self.lambda_t * time_loss
        events_loss = self.lambda_e * events_loss
        loss = time_loss + events_loss
        
        the_number_of_events = torch.numel(observed_time_seq)

        return loss / the_number_of_events, time_loss / the_number_of_events, events_loss / the_number_of_events


    def evaluate_procedure(self, history_time_seq, history_events_seq, history_mask_seq, \
                           observed_time_seq, observed_events_seq, mean, std):
        '''
        Remove the probability of the dummy event by mask.
        '''
        pred_time, pred_events_prob = self.model('evaluate', events_history = history_events_seq, time_history = history_time_seq, \
                                                 mask_history = history_mask_seq, mean = mean, std = std)
                                                                               # [batch_size, lh_length, num_events] * 2

        time_loss, events_loss = self.loss(pred_time = pred_time, observed_time = observed_time_seq, \
                                           pred_events_prob = pred_events_prob, observed_events = observed_events_seq)
        
        time_loss = self.lambda_t * time_loss
        events_loss = self.lambda_e * events_loss
        loss = time_loss + events_loss
        
        # f1 and mae
        pred_events = pred_events_prob.argmax(dim = -1)                        # [batch_size, hypro_length]
        observed_event_mask = torch.nn.functional.one_hot(pred_events, num_classes = self.num_events)
                                                                               # [batch_size, lh_length, num_events]
        pred_events, observed_events_seq = move_from_tensor_to_ndarray(pred_events, observed_events_seq)
        f1s = [f1_score(y_pred = pred_events_per_seq, y_true = observed_events_seq_per_seq, average = 'macro')
               for pred_events_per_seq, observed_events_seq_per_seq in zip(pred_events, observed_events_seq)]
        f1 = np.mean(f1s).item()

        mae = torch.abs((pred_time * observed_event_mask).sum(dim = -1) - observed_time_seq).mean().item()
        the_number_of_events = torch.numel(observed_time_seq)

        return loss / the_number_of_events, time_loss / the_number_of_events, events_loss / the_number_of_events, f1, mae


    def loss(self, pred_time, pred_events_prob, observed_time, observed_events):
        '''
        The definition of loss.
    
        Args:
            probability:        [batch_size, seq_len, num_events]
            events_next:        [batch_size, seq_len]
            mask_next:          [batch_size, seq_len]
        '''
        # Time loss
        observed_event_mask = torch.nn.functional.one_hot(observed_events, num_classes = self.num_events)
                                                                               # [batch_size, lh_length, num_events]
        selected_pred_time = (pred_time * observed_event_mask).sum(dim = -1)   # [batch_size, lh_length]
        gap = torch.abs(selected_pred_time - observed_time)                    # [batch_size, seq_len]
        time_loss = torch.sum(gap)
        
        # Event loss
        # cross entropy loss between p_{real} and p_{pred}.
        events_loss = torch.nn.functional.cross_entropy(rearrange(pred_events_prob, 'b lhl ne -> b ne lhl'), \
                                                                  observed_events.long(), reduction = 'none')
                                                                               # [batch_size, lh_length]
        events_loss = events_loss.sum()

        return time_loss, events_loss


    def long_horizon_pred(self, input_data, opt):
        (history_time_seq, history_events_seq, history_mask_seq, \
         observed_time_seq, observed_events_seq), (mean, std) = input_data
        pred_time, pred_events_prob = self.model('evaluate', events_history = history_events_seq, time_history = history_time_seq, \
                                                 mask_history = history_mask_seq, mean = mean, std = std)
                                                                               # [batch_size, lh_length, num_events] * 2
        pred_events = pred_events_prob.argmax(dim = -1)                        # [batch_size, hypro_length]
        observed_event_mask = torch.nn.functional.one_hot(pred_events, num_classes = self.num_events)
                                                                               # [batch_size, lh_length, num_events]
        selected_pred_time = (pred_time * observed_event_mask).sum(dim = -1)   # [batch_size, lh_length]
        
        pred_cum_time_seq = selected_pred_time.cumsum(dim = -1)                # [batch_size, lh_length]
        observed_cum_time_seq = observed_time_seq.cumsum(dim = -1)             # [batch_size, lh_length]
                
        # Generate new sequences which end at start_time + self.long_horizon_time
        # predicted sequences
        pred_seq_mask = pred_cum_time_seq < self.long_horizon_time             # [batch_size, lh_length]                         
        # observed sequences
        observed_seq_mask = observed_cum_time_seq < self.long_horizon_time     # [batch_size, lh_length]
   
        #1: Calculating OTD
        otds = []
        packed_data = zip(pred_cum_time_seq, pred_events, pred_seq_mask, \
                          observed_cum_time_seq, observed_events_seq, observed_seq_mask)
        for pred_cum_time_per_batch, pred_events_per_batch, pred_seq_mask_per_batch, \
            observed_cum_time_seq_per_batch, observed_events_seq_per_batch, observed_seq_mask_per_batch in packed_data:
                
            pred_cum_time_per_batch, pred_events_per_batch, pred_seq_mask_per_batch, \
            observed_cum_time_seq_per_batch, observed_events_seq_per_batch, observed_seq_mask_per_batch = \
                move_from_tensor_to_ndarray(pred_cum_time_per_batch, pred_events_per_batch, pred_seq_mask_per_batch, \
                                            observed_cum_time_seq_per_batch, observed_events_seq_per_batch, observed_seq_mask_per_batch)
        
            otds.append(
                otd(pred_time_seq = pred_cum_time_per_batch[pred_seq_mask_per_batch], pred_event_seq = pred_events_per_batch[pred_seq_mask_per_batch], \
                    true_time_seq = observed_cum_time_seq_per_batch[observed_seq_mask_per_batch], true_event_seq = observed_events_seq_per_batch[observed_seq_mask_per_batch], \
                    num_events = self.num_events, add_remove_event_cost = self.otd_event_removal_cost, move_event_cost = self.otd_move_event_cost, average = 'none')
            )
                        
        otds = np.array(otds)                                                  # [batch_size, fake_seq_num, ...]
        
        #2: Calculating mark prediction accuracy.
        
        return otds


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
        (history_time_seq, history_events_seq, history_mask_seq, \
         observed_time_seq, observed_events_seq), (mean, std) = minibatch
        loss, time_loss, events_loss \
            = model(         
                task_name = 'train', \
                history_time_seq = history_time_seq, history_events_seq = history_events_seq, history_mask_seq = history_mask_seq, \
                observed_time_seq = observed_time_seq, observed_events_seq = observed_events_seq, \
                mean = mean, std = std)
        
        loss.backward()

        loss = loss.item()
        time_loss = time_loss.item()
        events_loss = events_loss.item()
        
        return loss, time_loss, events_loss
    

    def evaluation_step(model, minibatch, device):
        ''' Epoch operation in evaluation phase '''
    
        model.eval()
        (history_time_seq, history_events_seq, history_mask_seq, \
         observed_time_seq, observed_events_seq), (mean, std) = minibatch
        loss, time_loss, events_loss, f1_pred, mae \
            = model(         
                task_name = 'evaluate', \
                history_time_seq = history_time_seq, history_events_seq = history_events_seq, history_mask_seq = history_mask_seq, \
                observed_time_seq = observed_time_seq, observed_events_seq = observed_events_seq, \
                mean = mean, std = std)
        
        loss = loss.item()
        time_loss = time_loss.item()
        events_loss = events_loss.item()
            
        return loss, time_loss, events_loss, f1_pred, mae


    def postprocess(input, procedure):
        def train_postprocess(input):
            '''
            Training process
            [absolute loss, relative loss, events loss]
            '''
            return [input[0], input[1], input[2]]
        
        def test_postprocess(input):
            '''
            Evaluation process
            [absolute loss, relative loss, events loss, mae value]
            '''
            return [input[0], input[1], input[2], input[3], input[4]]
        
        return (train_postprocess(input) if procedure == 'Training' else test_postprocess(input))
    

    def log_print_format(input, procedure):
        def train_log_print_format(input):
            format_dict = {}
            format_dict['loss'] = pack_one_value_to_dict(input[0])
            format_dict['time_loss'] = pack_one_value_to_dict(input[1])
            format_dict['events_loss'] = pack_one_value_to_dict(input[2])
            return format_dict

        def test_log_print_format(input):
            format_dict = {}
            format_dict['loss'] = pack_one_value_to_dict(input[0])
            format_dict['time_loss'] = pack_one_value_to_dict(input[1])
            format_dict['events_loss'] = pack_one_value_to_dict(input[2])
            format_dict['f1'] = pack_one_value_to_dict(input[3])
            format_dict['mae'] = pack_one_value_to_dict(input[4], '2.8f')
            return format_dict
        
        return (train_log_print_format(input) if procedure == 'Training' else test_log_print_format(input))

    format_dict_length = 5
    
    def choose_metric(evaluation_report_format_dict, test_report_format_dict):
        '''
        [relative loss on evaluation dataset, relative loss on test dataset, event loss on test dataset]
        '''
        return [evaluation_report_format_dict['loss'], 
                test_report_format_dict['loss']], \
               ['evaluation_loss', 'test_loss']
    
    metric_number = 2 # metric number is the length of the output of choose_metric