import torch
import torch.nn.functional as F

from .transformers import TransformerTPP
from ..utils import BasicModule
from .utils import *

class THP(BasicModule):
    def __init__(self, num_events, device, d_input = 64, d_rnn = 64, d_hidden = 256, n_layers = 3,
                 n_head = 3, d_qk = 64, d_v = 64, dropout = 0.1, beta = 1):
        super(THP, self).__init__()
        self.device = device
        self.num_events = num_events if num_events > 0 else 1

        # parameter for the weight of time difference
        self.alpha = nn.Parameter(torch.tensor(-5, dtype = torch.float32, device = self.device, requires_grad = True))

        # parameter for the softplus function
        self.beta = beta

        self.model = TransformerTPP(num_events, device = self.device, d_input = d_input, d_rnn = d_rnn, d_hidden = d_hidden,\
                                    n_layers = n_layers, n_head = n_head, d_qk = d_qk, d_v = d_v, dropout = dropout)
    
    '''
    Functions for model propagation and evaluation
    '''
    def forward(self, time, events, mask):
        '''
        Check if events data is present.
        Now, we assume that no event data is available.
        Args:
        1. time: the sequence containing events' timestamps. shape: [batch_size, seq_len + 1]
        2. events: the sequence containing information about events. shape: [batch_size, seq_len + 1]
        3. mask: filter out the padding events in the event batches. shape: [batch_size, seq_len + 1]
        '''

        time_history, events_history = time[:, :-1], events[:, :-1]            # [batch_size, seq_len] * 2
        time_next, events_next = time[:, 1:], events[:, 1:]                    # [batch_size, seq_len] * 2
        mask_next = mask[:, 1:]

        history, (events_pred, time_pred) = \
            self.model(time_history, events_history)                           # 2 * [batch_size, seq_len, num_types]

        # temporal point process loss
        log_likeli_loss = self.log_likelihood(
             history = history, time = time_next, events = events_next, mask = mask_next
        )
        # event loss
        events_loss = F.cross_entropy(input = events_pred.reshape(-1, self.num_events),\
                                      target = events_next.reshape(-1).to(torch.long), reduction='none')
        events_loss *= mask_next.reshape(-1)
        events_loss = events_loss.sum()

        the_number_of_events = mask_next.sum()

        return log_likeli_loss, events_loss, the_number_of_events
    
    def evaluate(self, time, events):
        '''
        Check if events data is present.
        Now, we assume that no event data is available.
        Args:
        1. time: the sequence containing events' timestamps. shape: [batch_size, seq_len + 1]
        2. events: the sequence containing information about events. shape: [batch_size, seq_len + 1]
        '''

        time_history, events_history = time[:, :-1], events[:, :-1]            # [batch_size, seq_len] * 2

        history, (events_pred, time_pred) = \
            self.model(time_history, events_history)                           # [batch_size, seq_len, num_types]

        return history, (events_pred, time_pred)

    '''
    Loss functions
    '''
    def log_likelihood(self, history, time, events, mask):
        """ Log-likelihood of sequence. """
    
        '''
        Currently, we assume no event data is available.
        '''
        non_pad_mask = get_non_pad_mask(time, self.num_events).squeeze(2)      # [batch_size, seq_len]
    
        if events is not None:
            type_mask = torch.zeros([*events.size(), self.num_events], device=history.device)
            for i in range(self.num_events):
                type_mask[:, :, i] = (events == i + 1).bool().to(history.device)
                                                                               # [batch_size, seq_len, num_types]
        else:
            '''
            All except the first events are included.
            '''
            type_mask = torch.ones_like(history, device = history.device)      # [batch_size, seq_len, num_types]


        '''
        Obtain values from intensity functions. But why they flitered out other intensity values?
        '''
        type_lambda = torch.sum(F.softplus(history + self.alpha * time.unsqueeze(dim = -1), self.beta) * type_mask, dim=2)
                                                                               # [batch_size, seq_len]
    
        # event log-likelihood
        event_ll = compute_event(type_lambda, non_pad_mask) * mask             # [batch_size, seq_len]
        event_ll = torch.sum(event_ll, dim=-1)                                 # [batch_size]
    
        # non-event log-likelihood, either numerical integration or MC integration
        # non_event_ll = compute_integral_biased(type_lambda, time, non_pad_mask)
        non_event_ll = self.compute_integral_unbiased(history, time, non_pad_mask, type_mask) * mask
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
        time, events, _, _, _ = input_data                                     # 3 * [batch_size, seq_len + 1]
        time_next, events_next = time[:, 1:], events[:, 1:]                    # 2 * [batch_size, seq_len]

        batch_size, seq_len = time.shape
        seq_len -= 1
        history, (event_pred, time_pred) = self.evaluate(time, events)         # [batch_size, seq_len, num_types]

        # intensity part
        if events is not None:
            type_mask = torch.zeros([batch_size, seq_len, self.num_events], device=history.device)
            for i in range(self.num_events):
                type_mask[:, :, i] = (events_next == i + 1).bool().to(history.device)
                                                                               # [batch_size, seq_len, num_types]
        else:
            '''
            All except the first events are included.
            '''
            type_mask = torch.ones_like(history, device = history.device)      # [batch_size, seq_len, num_types]

        time_multiplier = torch.linspace(0, 1, resolution, device = self.device)
        expanded_time = time_next.unsqueeze(-1) * time_multiplier              # [batch_size, seq_len, resolution]
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
        time, event, fact, mask = minibatch                                    # 3 * [batch_size, seq_len + 1, 1] & [batch_size, seq_len, 1]
        tpp_loss, mark_loss, the_number_of_events = model(time, event, mask)
        loss = tpp_loss + mark_loss
        loss.backward()

        tpp_loss, mark_loss = tpp_loss.item(), mark_loss.item()
        fact = fact.sum()
    
        return tpp_loss / the_number_of_events , mark_loss / the_number_of_events, fact / the_number_of_events
    
    def evaluation_step(model, minibatch, device):
        ''' Epoch operation in evaluation phase '''
    
        model.eval()

        time, event, fact, mask = minibatch                                    # 3 * [batch_size, seq_len + 1, 1] & [batch_size, seq_len, 1]
        tpp_loss, mark_loss, the_number_of_events = model(time, event, mask)

        tpp_loss, mark_loss = tpp_loss.item(), mark_loss.item()
        fact = fact.sum()

        return tpp_loss / the_number_of_events, mark_loss / the_number_of_events, fact / the_number_of_events

    def postprocess(input):
        return [input[0], input[0] - input[2], input[1]]


    def log_print_format(input):
        format_dict = {}
        format_dict['absolute_loss'] = input[0]
        format_dict['relative_loss'] = input[1]
        format_dict['events_loss'] = input[2]
        format_dict['num_format'] = {'absolute_loss': ':8.5f', 'relative_loss': ':8.5f', 'events_loss': ':8.5f'}
        return format_dict
        
    format_dict_length = 3
    

    logfile_format = {'step': '', 'absolute loss': ':8.5f', 'relative loss': ':8.5f', 'events loss': ':8.5f'}

    def logfile_print_format(input):
        format_dict = {}
        format_dict['absolute loss'] = input[0]
        format_dict['relative loss'] = input[1]
        format_dict['events loss'] = input[2]
        return format_dict
    
    def choose_metric(evaluation_report, test_report):
        '''
        [relative loss on evaluation dataset, relative loss on test dataset]
        '''
        return [evaluation_report[1].item() + evaluation_report[-1], test_report[1].item()+ test_report[-1]]
    
    metric_number = 2 # metric number is the length of the output of choose_metric