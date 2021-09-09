from .log_norm_mix import LogNormMix
from ..utils import BasicModule

import torch

class ifl(BasicModule):
    def __init__(self, num_marks: int, device, mean_log_inter_time: float = 0.0, std_log_inter_time: float = 1.0, 
                       context_size: int = 32, mark_embedding_size: int = 32, num_mix_components: int = 16, rnn_type: str = "GRU"):
        super(ifl, self).__init__()
        self.device = device
        self.model = LogNormMix(
            num_marks,
            mean_log_inter_time,
            std_log_inter_time,
            context_size,
            mark_embedding_size,
            num_mix_components,
            rnn_type,
        ).to(self.device)
    
    def forward(self, minibatch):
        '''
        The shape of minibatch
        [
            event_tensor,
            time_tensor,
            mask_tensor
        ]
        '''
        return self.model.log_prob(minibatch)

    def train_step(model, minibatch, device):
        ''' Epoch operation in training phase'''
    
        model.train()
            
        log_likelihood = model(minibatch[0])
    
        loss = loss_f(log_likelihood)
        loss.backward()
    
        loss = loss.item()
        fact = minibatch[1].sum()
    
        return loss, fact
    
    def evaluation_step(model, minibatch, device):
        ''' Epoch operation in evaluation phase '''
    
        model.eval()
        log_likelihood = model(minibatch[0])
    
        loss = loss_f(log_likelihood)
    
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

def loss_f(loglik):
    '''
    The definition of loss.
    '''
    return loglik.mul(-1.0).sum()
