import torch
import torch.nn as nn
from .ctlstm import CTLSTM

class CTLSTMwrapper(nn.Module):
    def __init__(self, hidden_dim, event_num = 1, beta = 1):
        super(CTLSTMwrapper, self).__init__()
        self.model = CTLSTM(hidden_dim = hidden_dim, event_num = event_num, beta = beta)
    
    def forward(self, minibatch, eval_tag):
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
                          eval_tag = eval_tag)

    @staticmethod
    def train_step(model, minibatch, optimizer, device):
        ''' Epoch operation in training phase'''
    
        model.train()
        optimizer.zero_grad()
        intensity_integral, intensity = model(
                minibatch[0].to(device), eval_tag = False)
    
        loss = loss_f(
            intensity=intensity, intensity_integral=intensity_integral
        )
        loss.backward()
        optimizer.step_and_update_lr()
    
        loss = loss.item()
        fact = minibatch[1].sum()
    
        return loss, fact
    
    @staticmethod
    def evaluation_step(model, minibatch, device):
        ''' Epoch operation in evaluation phase '''
    
        model.eval()
        intensity_integral, intensity = model(
            minibatch.to(device), eval_tag = True
        )
    
        loss = loss_f(
            intensity=intensity, intensity_integral=intensity_integral
        )
    
        loss = loss.item()
        fact = minibatch[1].sum()
    
        return loss, fact

def loss_f(value):
    '''
    The definition of loss.
    '''
    loss = -value
    loss = torch.clamp(loss, max=10)
    loss = loss.sum(axis = -1)
    
    return loss