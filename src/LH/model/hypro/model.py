import torch
import numpy as np
from sklearn.metrics import accuracy_score
from einops import rearrange, repeat, reduce, pack

from src.toolbox.metrics import otd
from src.toolbox.misc import pack_one_value_to_dict

from src.LH.model.basic_tpp_model import BasicModel
from src.LH.model.hypro.submodel import HYPRO
from src.LH.model.utils import *
# from src.LH.model.hypro.plot import *


class HYPROModel(BasicModel):
    def __init__(self, opt, device, d_input, dropout, d_hidden, n_layers, n_head, 
                 d_qk, d_v, epsilon = 1e-20, loss = 'binary',
                 long_horizon_time = 1, otd_event_removal_cost = 1.0, move_event_cost = 1.0):
        super(HYPROModel, self).__init__()
        self.device = device
        self.num_events = opt.info_dict['num_events']
        self.start_time = opt.info_dict['t_0']
        self.end_time = opt.info_dict['T']
        self.hypro_length = opt.info_dict['hypro_length']
        self.hypro_negative_samples = opt.info_dict['hypro_negative_samples']
        self.epsilon = epsilon
        self.loss = loss
        self.long_horizon_time = long_horizon_time
        self.otd_event_removal_cost = otd_event_removal_cost
        self.otd_move_event_cost = move_event_cost
        
        self.model = HYPRO(num_events = self.num_events, d_input = d_input, \
                           d_hidden = d_hidden, n_layers = n_layers, n_head = n_head, d_qk = d_qk, \
                           d_v = d_v, dropout = dropout, device = device)
        
    
    def forward(self, task_name, *args, **kwargs):

        task_mapper = {
            'train': self.train_procedure,
            'evaluate': self.evaluate_procedure,
            
            'long_horizon_pred': self.long_horizon_pred,
        }
        '''
        'spearman_and_l1': self.get_spearman_and_l1,
        'mae_and_f1': self.get_mae_and_f1,
        'mae_e_and_f1': self.get_mae_e_and_f1,

        # figure drawing funtions
        'intensity': self.figure_intensity,
        'integral': self.figure_integral,
        'probability': self.figure_probability,
        'debug': self.figure_debug
        '''

        return task_mapper[task_name](*args, **kwargs)
    
    
    def train_procedure(self, time_seq, events_seq, mark, mask_seq, mean, std):
        batch_size, _ = mask_seq.shape
        mask_seq = mask_seq.unsqueeze(dim = -2)                                # [batch_size, 1 + number_of_negative_samples, seq_len]
        seq_energy = self.model(time_seq, events_seq, mask_seq)                # [batch_size, 1 + number_of_negative_samples]
        loss = self.loss_function(seq_energy, mark)
        
        return loss, batch_size
        
    
    def evaluate_procedure(self, time_seq, events_seq, mark, mask_seq, mean, std):
        batch_size, _ = mask_seq.shape
        mask_seq = mask_seq.unsqueeze(dim = -2)                                # [batch_size, 1 + number_of_negative_samples, seq_len]
        seq_energy = self.model(time_seq, events_seq, mask_seq)                # [batch_size, 1 + number_of_negative_samples]
        loss = self.loss_function(seq_energy, mark)
        
        selected_index = seq_energy.argmin(dim = -1)                           # [batch_size]
        true_index = mark.argmax(dim = -1)                                     # [batch_size]
        np_selected_index, np_true_index = move_from_tensor_to_ndarray(selected_index, true_index)
        acc = accuracy_score(y_true = np_true_index, y_pred = np_selected_index)
        
        selection_mask = torch.nn.functional.one_hot(selected_index, num_classes = self.hypro_negative_samples + 1)
        energy_selected_seq_selection_mask = seq_energy[selection_mask == 1].mean().item()
        energy_noise_seq_selection_mask = seq_energy[selection_mask != 1].mean().item()
        
        energy_selected_seq_true_mask = seq_energy[mark == 1].mean().item()
        energy_noise_seq_true_mask = seq_energy[mark != 1].mean().item()

        return loss, acc, batch_size, \
               energy_selected_seq_selection_mask, energy_noise_seq_selection_mask, \
               energy_selected_seq_true_mask, energy_noise_seq_true_mask
        

    def loss_function(self, seq_energy, mark):
        loss = 0
        
        if self.loss == 'binary':
            reversed_mark = 1 - mark
            true_seq_energy = (seq_energy * mark).sum(dim = -1)                # [batch_size]
            noise_seq_energy = seq_energy * reversed_mark                      # [batch_size, 1 + number_of_negative_samples]
            loss_true = torch.nn.functional.logsigmoid(-true_seq_energy)       # [batch_size]
            loss_noise = torch.nn.functional.logsigmoid(noise_seq_energy).sum(dim = -1)
                                                                               # [batch_size]
            loss = -(loss_true + loss_noise).sum()
        elif self.loss == 'multi':
            seq_energy = -seq_energy                                           # [batch_size, 1 + number_of_negative_samples]
            true_seq_energy = (seq_energy * mark).sum(dim = -1)                # [batch_size]
            noise_seq_energy = torch.logsumexp(seq_energy.masked_fill_(mark.bool(), value = -1e30), dim = -1)
                                                                               # [batch_size]
            loss = -(true_seq_energy - noise_seq_energy).sum()
        else:
            raise Exception('Unknown loss function. The expected loss is either binary or multi.')
        
        return loss
    
    
    def long_horizon_pred(self, input_data, opt):
        (time_seq, events_seq, mark, mask_seq), (mean, std) = input_data       # 2 * [batch_size, real_and_fake_seq_num, seq_len] + [batch_size, real_and_fake_seq_num] + [batch_size, seq_len]
        batch_size, fake_seq_num_add_1, _ = time_seq.shape
        mask_seq = repeat(mask_seq, 'bs sl -> bs fsn sl', fsn = fake_seq_num_add_1)
                                                                               # [batch_size, fake_seq_num + 1, seq_len]
        # split the true sequence from the noises.
        true_time_seq = time_seq[mark == 1]                                    # [batch_size, seq_len]
        true_event_seq = events_seq[mark == 1]                                 # [batch_size, seq_len]
        true_mask_seq = mask_seq[:, 0]                                         # [batch_size, seq_len]
        
        noise_time_seq = rearrange(time_seq[mark != 1], '(bs sn) sl -> bs sn sl', bs = batch_size)
                                                                               # [batch_size, fake_seq_num, seq_len]
        noise_event_seq = rearrange(events_seq[mark != 1], '(bs sn) sl -> bs sn sl', bs = batch_size)
                                                                               # [batch_size, fake_seq_num, seq_len]
        noise_mask_seq = mask_seq[:, 1:]                                       # [batch_size, fake_seq_num, seq_len]
        
        # the energy of noise sequences.
        noise_seq_energy = self.model(noise_time_seq, noise_event_seq, noise_mask_seq)
                                                                               # [batch_size, fake_seq_num]
        noise_best_seq_by_energy = noise_seq_energy.argmin(dim = -1)           # [batch_size, fake_seq_num]

        # judge where the long horizon prediction starts.
        true_cum_time_seq = true_time_seq.cumsum(dim = -1)                     # [batch_size, seq_len]
        noise_cum_time_seq = noise_time_seq.cumsum(dim = -1)                   # [batch_size, fake_seq_num, seq_len]
        
        start_time = true_cum_time_seq[:, -self.hypro_length - 1]              # [batch_size]
        
        # Generate new sequences which end at start_time + self.long_horizon_time
        # true sequences
        true_time_seq_mask = true_cum_time_seq < (start_time + self.long_horizon_time)
                                                                               # [batch_size, seq_len]                         
        # noise sequences
        noise_time_seq_mask = noise_cum_time_seq < (start_time + self.long_horizon_time)
                                                                               # [batch_size, fake_seq_num, seq_len]       
        
        #1: Calculating OTD
        otds = []
        packed_data = zip(true_time_seq, true_event_seq, true_mask_seq, true_time_seq_mask, \
                          noise_time_seq, noise_event_seq, noise_mask_seq, noise_time_seq_mask)
        for true_time_seq_per_batch, true_event_seq_per_batch, true_seq_mask_per_batch, true_time_seq_mask_per_batch, \
            noise_time_seq_per_batch, noise_event_seq_per_batch, noise_mask_seq_per_batch, noise_time_seq_mask_per_batch in packed_data:
            true_time_seq_per_batch = true_time_seq_per_batch[(true_seq_mask_per_batch == 1) & true_time_seq_mask_per_batch]
            true_event_seq_per_batch = true_event_seq_per_batch[(true_seq_mask_per_batch == 1) & true_time_seq_mask_per_batch]
            
            true_time_seq_per_batch, true_event_seq_per_batch = \
                move_from_tensor_to_ndarray(true_time_seq_per_batch, true_event_seq_per_batch)
            
            otds_all_samples = []
            sub_packed_data = zip(noise_time_seq_per_batch, noise_event_seq_per_batch, noise_mask_seq_per_batch, noise_time_seq_mask_per_batch)
            for noise_time_seq_per_batch_per_sample, noise_event_seq_per_batch_per_sample, noise_mask_seq_per_batch_per_sample, noise_time_seq_mask_per_batch_per_sample, in sub_packed_data:
                noise_time_seq_per_batch_per_sample = noise_time_seq_per_batch_per_sample[(noise_mask_seq_per_batch_per_sample == 1) & noise_time_seq_mask_per_batch_per_sample]
                noise_event_seq_per_batch_per_sample = noise_event_seq_per_batch_per_sample[(noise_mask_seq_per_batch_per_sample == 1) & noise_time_seq_mask_per_batch_per_sample]
                
                noise_time_seq_per_batch_per_sample, noise_event_seq_per_batch_per_sample = \
                    move_from_tensor_to_ndarray(noise_time_seq_per_batch_per_sample, noise_event_seq_per_batch_per_sample)
                otds_all_samples.append(
                    otd(pred_time_seq = noise_time_seq_per_batch_per_sample.cumsum(axis = -1), pred_event_seq = noise_event_seq_per_batch_per_sample, \
                        true_time_seq = true_time_seq_per_batch.cumsum(axis = -1), true_event_seq = true_event_seq_per_batch, \
                        num_events = self.num_events, add_remove_event_cost = self.otd_event_removal_cost, move_event_cost = self.otd_move_event_cost, average = 'none')
                )
            
            otds.append(otds_all_samples)
            
        otds = torch.tensor(np.array(otds), device = self.device)              # [batch_size, fake_seq_num, ...]
        
        otd_picked_by_hypro = otds[:, noise_best_seq_by_energy].mean(dim = 0)  # [...]
        otd_average_sample = reduce(otds, 'bs fsn ... -> ...', 'mean')         # [...]
        
        otd_picked_by_hypro, otd_average_sample = move_from_tensor_to_ndarray(otd_picked_by_hypro, otd_average_sample)
        
        #2: Calculating mark prediction accuracy.
        
        return otd_picked_by_hypro, otd_average_sample

    
    '''
    Static methods
    '''
    def train_step(model, minibatch, device):
        ''' Epoch operation in training phase'''
        
        model.train()

        (time_seq, events_seq, mark, mask_seq), (mean, std) = minibatch        # 2 * [batch_size, real_and_fake_seq_num, seq_len] + [batch_size, real_and_fake_seq_num] + [batch_size, seq_len]
        loss, batch_size = model('train', time_seq, events_seq, mark, mask_seq, mean = mean, std = std)

        loss.backward()
    
        loss = loss.item() / batch_size
        
        return loss
    

    def evaluation_step(model, minibatch, device):
        ''' Epoch operation in evaluation phase '''
        
        model.eval()
        
        (time_seq, events_seq, mark, mask_seq), (mean, std) = minibatch        # 2 * [batch_size, real_and_fake_seq_num, seq_len] + [batch_size, real_and_fake_seq_num] + [batch_size, seq_len]
        loss, acc, batch_size, energy_selected_seq_selection_mask, energy_noise_seq_selection_mask, \
        energy_selected_seq_true_mask, energy_noise_seq_true_mask \
            = model('evaluate', time_seq, events_seq, mark, mask_seq, mean = mean, std = std)

        loss = loss.item() / batch_size

        return loss, acc, energy_selected_seq_selection_mask, energy_noise_seq_selection_mask, energy_selected_seq_true_mask, energy_noise_seq_true_mask
    
    
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
            return [input[0], input[1], input[2], input[3], input[4], input[5]]
        
        return (train_postprocess(input) if procedure == 'Training' else test_postprocess(input))
    
    
    def log_print_format(input, procedure):
        def train_log_print_format(input):
            format_dict = {}
            format_dict['loss'] = pack_one_value_to_dict(input[0])
            return format_dict

        def test_log_print_format(input):
            format_dict = {}
            format_dict['loss'] = pack_one_value_to_dict(input[0])
            format_dict['accuracy'] = pack_one_value_to_dict(input[1])
            format_dict['energy_of_selected_seq_by_selection_mask'] = pack_one_value_to_dict(input[2])
            format_dict['energy_of_noise_seq_by_selection_mask'] = pack_one_value_to_dict(input[3])
            format_dict['energy_of_selected_seq_by_true_mask'] = pack_one_value_to_dict(input[4])
            format_dict['energy_of_noise_seq_by_true_mask'] = pack_one_value_to_dict(input[5])

            return format_dict
        
        return (train_log_print_format(input) if procedure == 'Training' else test_log_print_format(input))


    format_dict_length = 6

    
    def choose_metric(evaluation_report_format_dict, test_report_format_dict):
        '''
        [relative loss on evaluation dataset, relative loss on test dataset, event loss on test dataset]
        '''
        return [evaluation_report_format_dict['accuracy'], 
                test_report_format_dict['accuracy']], \
               ['evaluation_accuracy', 'test_accuracy']

    metric_number = 2 # metric number is the length of the output of choose_metric
    smaller_is_better = [False, False]