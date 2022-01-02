import torch
import torch.nn.functional as F

from .transformers import TransformerTPP
from ..utils import BasicModule
from .utils import *

class THP(BasicModule):
    def __init__(self, num_types, device, d_input = 64, d_rnn = 64, d_hidden = 256, n_layers = 3,
                 n_head = 3, d_qk = 64, d_v = 64, dropout = 0.1, beta = 1):
        super(THP, self).__init__()
        self.device = device
        self.num_types = num_types if num_types > 0 else 1

        # parameter for the weight of time difference
        self.alpha = nn.Parameter(torch.tensor(-5, dtype = torch.float32, device = self.device, requires_grad = True))

        # parameter for the softplus function
        self.beta = beta

        self.model = TransformerTPP(num_types, device = self.device, d_input = d_input, d_rnn = d_rnn, d_hidden = d_hidden,\
                                    n_layers = n_layers, n_head = n_head, d_qk = d_qk, d_v = d_v, dropout = dropout)
    
    '''
    Functions for model propagation and evaluation
    '''
    def forward(self, minibatch):
        '''
        Check if events data is present.
        Now, we assume that no event data is available.
        Args:
        1. minibatch: the input data from dataloaders. shape: ['time': [batch_size, seq_len + 1]]
        '''

        time = minibatch[0]
        events = None

        history, prediction = self.model(time[:, :-1], events)                 # [batch_size, seq_len, num_types]

        log_likeli_loss = self.log_likelihood(
             history = history, time = time[:, 1:], events = events
        )

        return log_likeli_loss
    
    def evaluate(self, minibatch):
        '''
        Check if events data is present.
        Now, we assume that no event data is available.
        Args:
        1. minibatch: the input data from dataloaders. shape: ['time': [batch_size, seq_len + 1]]
        '''

        time = minibatch[0]
        events = None

        history, prediction = self.model(time[:, :-1], events)                 # [batch_size, seq_len, num_types]

        return history, prediction

    '''
    Loss functions
    '''
    def log_likelihood(self, history, time, events):
        """ Log-likelihood of sequence. """
    
        '''
        Currently, we assume no event data is available.
        '''
        non_pad_mask = get_non_pad_mask(time).squeeze(2)                       # [batch_size, seq_len]
    
        if events:
            type_mask = torch.zeros([*events.size(), self.num_types], device=history.device)
            for i in range(self.num_types):
                type_mask[:, :, i] = (events == i + 1).bool().to(history.device)
                                                                               # [batch_size, seq_len, num_types]
        else:
            '''
            All except the first events are included.
            '''
            type_mask = torch.ones_like(history, device = history.device)      # [batch_size, seq_len, num_types]


        '''
        Obtain values from intensity functions. But why they flitered other intensity values?
        '''
        type_lambda = torch.sum(F.softplus(history + self.alpha * time.unsqueeze(dim = -1), self.beta) * type_mask, dim=2)
                                                                               # [batch_size, seq_len]
    
        # event log-likelihood
        event_ll = compute_event(type_lambda, non_pad_mask)                    # [batch_size, seq_len]
        event_ll = torch.sum(event_ll, dim=-1)                                 # [batch_size]
    
        # non-event log-likelihood, either numerical integration or MC integration
        # non_event_ll = compute_integral_biased(type_lambda, time, non_pad_mask)
        non_event_ll = self.compute_integral_unbiased(history, time, non_pad_mask, type_mask)
                                                                               # [batch_size, seq_len]
        non_event_ll = torch.sum(non_event_ll, dim=-1)                         # [batch_size]
    
        event_loss = -torch.sum(event_ll - non_event_ll)
        return event_loss
    
    def compute_integral_unbiased(self, history, time, non_pad_mask, type_mask):
        """ Log-likelihood of non-events, using Monte Carlo integration. """
    
        num_samples = 100
    
        diff_time = time * non_pad_mask
        temp_time = diff_time.unsqueeze(2) * \
                    torch.rand([*diff_time.size(), num_samples], device=history.device)
                                                                               # [batch_size, seq_len, num_samples]
        # temp_time /= (time[:, :-1] + 1).unsqueeze(2)
    
        temp_hid = torch.sum(history * type_mask, dim=2, keepdim=True)         # [batch_size, seq_len, 1]
    
        all_lambda = F.softplus(temp_hid + self.alpha * temp_time, self.beta)  # [batch_size, seq_len, num_samples]
        all_lambda = torch.mean(all_lambda, dim=2)                             # [batch_size, seq_len]
    
        unbiased_integral = all_lambda * diff_time                             # [batch_size, seq_len]
        return unbiased_integral

    def function_prober(self, input_data, resolution):
        '''
        Probe the learned intensity function from the model.
        This task should be pretty easy for the explicit form of intensity functions.
        '''
        time = input_data[0][:, 1:]                                            # [batch_size, seq_len]
        batch_size, seq_len = time.shape
        events = None
        history, prediction = self.evaluate(input_data)                        # [batch_size, seq_len, num_types]

        # intensity part
        if events:
            type_mask = torch.zeros([*events.size(), self.num_types], device=history.device)
            for i in range(self.num_types):
                type_mask[:, :, i] = (events == i + 1).bool().to(history.device)
                                                                               # [batch_size, seq_len, num_types]
        else:
            '''
            All except the first events are included.
            '''
            type_mask = torch.ones_like(history, device = history.device)      # [batch_size, seq_len, num_types]

        time_multiplier = torch.linspace(0, 1, resolution, device = self.device)
        expanded_time = time.unsqueeze(-1) * time_multiplier                   # [batch_size, seq_len, resolution]
        history = torch.sum(history * type_mask, dim=-1, keepdim=True)         # [batch_size, seq_len, 1]
        
        expanded_intensity = F.softplus(self.alpha * expanded_time + history, self.beta)
                                                                               # [batch_size, seq_len, resolution]
        expanded_intensity = expanded_intensity.reshape(batch_size, -1)        # [batch_size, seq_len * resolution]

        # aggregated timestamp
        timestamp = torch.cat(
            (torch.zeros((batch_size, seq_len, 1), device = self.device), expanded_time.diff(dim = -1)),
            dim = -1)                                                          # [batch_size, seq_len, resolution]
        timestamp = timestamp.reshape(batch_size, seq_len * resolution)        # [batch_size, seq_len * resolution]
        
        return torch.zeros_like(expanded_intensity), expanded_intensity, timestamp


    '''
    Static methods
    '''
    def train_step(model, minibatch, device):
        ''' Epoch operation in training phase'''
        model.train()

        '''
        Maybe need another function to extract data from minibatches.
        Currently, we don't acquire any prediction loss to assist the model training.  
        '''
        loss = model(minibatch)                                                # [batch_size, seq_len, 1]
        loss.backward()

        loss = loss.item()
        fact = minibatch[1].sum()
    
        return loss, fact
    
    def evaluation_step(model, minibatch, device):
        ''' Epoch operation in evaluation phase '''
    
        model.eval()
        loss = model(minibatch)                                                # [batch_size, seq_len, 1]

        loss = loss.item()
        fact = minibatch[1].sum()

        return loss, fact

    def postprocess(input):
        def train_postprocess(input):
            '''
            Training process
            [absolute loss, relative loss]
            '''
            return [input[0], input[0] - input[1]]
        
        def test_postprocess(input):
            '''
            Evaluation process
            [absolute loss, relative loss, mae value]
            '''
            return [input[0], input[0] - input[1], input[2]]
        
        return (train_postprocess(input) if len(input) == 2 else test_postprocess(input))

    def log_print_format(input):
        def train_log_print_format(input):
            format_dict = {}
            format_dict['absolute_loss'] = input[0]
            format_dict['relative_loss'] = input[1]
            format_dict['num_format'] = {'absolute_loss': ':8.5f', 'relative_loss': ':8.5f'}
            return format_dict

        def test_log_print_format(input):
            format_dict = {}
            format_dict['absolute_loss'] = input[0]
            format_dict['relative_loss'] = input[1]
            format_dict['mae'] = input[2]
            format_dict['num_format'] = {'absolute_loss': ':8.5f', 'relative_loss': ':8.5f', 'mae': ''}
            return format_dict
        
        return (train_log_print_format(input) if len(input) == 2 else test_log_print_format(input))

    format_dict_length = 3
    
    logfile_format = {'step': '', 'absolute loss': ':8.5f', 'relative loss': ':8.5f'}

    def logfile_print_format(input):
        format_dict = {}
        format_dict['absolute loss'] = input[0]
        format_dict['relative loss'] = input[1]
        return format_dict
    
    def choose_metric(evaluation_report, test_report):
        '''
        [relative loss on evaluation dataset, relative loss on test dataset]
        '''
        return [evaluation_report[1].item(), test_report[1].item()]
    
    metric_number = 2 # metric number is the length of the output of choose_metric