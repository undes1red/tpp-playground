from ..utils import BasicModule
from .ctlstm import CTLSTM

import torch

class CTLSTMwrapper(BasicModule):
    def __init__(self, hidden_dim, device, mc_sample_num = 1., event_num = 1, beta = 1):
        super(CTLSTMwrapper, self).__init__()
        self.device = device
        self.event_num = event_num
        self.model = CTLSTM(hidden_dim = hidden_dim, event_num = event_num, beta = beta, device = device, mc_sample_num = mc_sample_num)
    
    def forward(self, minibatch, eval_tag, intensity_instead):
        '''
        The shape of minibatch
        [
            [<event_tensor>],
            [<dtime_tensor>],
            [token_number_tensor],
            [duration_tensor]
        ]
        '''

        event_tensor, dtime_tensor, token_num_tensor, duration_tensor = minibatch

        return self.model(event_tensor = event_tensor, dtime_tensor = dtime_tensor, 
                          token_num_tensor = token_num_tensor, duration_tensor = duration_tensor, 
                          eval_tag = eval_tag, intensity_instead = intensity_instead)

    def train_step(model, minibatch, device):
        ''' Epoch operation in training phase'''
    
        model.train()
            
        log_likelihood = model(
            minibatch[0], eval_tag = False, intensity_instead = False
        )
    
        loss = loss_f(
            value = log_likelihood
        )
        loss.backward()
    
        loss = loss.item()
        fact = minibatch[1].sum()
    
        return loss, fact
    
    def evaluation_step(model, minibatch, device):
        ''' Epoch operation in evaluation phase '''
    
        model.eval()
        log_likelihood = model(
            minibatch[0], eval_tag = True, intensity_instead = False
        )
    
        loss = loss_f(
            value = log_likelihood
        )
    
        loss = loss.item()
        fact = minibatch[1].sum()
    
        return loss, fact

    def postprocess(input):
        return [input[0], input[0] - input[1]]

    def log_print_format(input):
        format_dict = {}
        format_dict['absolute_loss'] = input[0]
        format_dict['relative_loss'] = input[1]
        format_dict['num_format'] = {'absolute_loss': ':8.5f', 'relative_loss': ':8.5f'}
        return format_dict

    format_dict_length = 2
    
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
        return [torch.abs(evaluation_report[-1]).item(), torch.abs(test_report[-1]).item()]
    
    metric_number = 2 # metric number is the length of the output of choose_metric

    def function_prober(self, data, resolution):
        self.model.eval()
        integral, intensity, timestamp = self.model.intensity(data[0], resolution)
        return integral, intensity, timestamp
    
    def model_prober(self, data):
        self.model.eval()
        return self.forward(data[0], eval_tag = True, intensity_instead = True)

def loss_f(value):
    '''
    The definition of loss.
    '''
    loss = -value
    
    return loss