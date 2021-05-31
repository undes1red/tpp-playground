import torch.nn as nn
from .ctlstm import CTLSTM

class CTLSTMwrapper(nn.Module):
    def __init__(self, hidden_dim, device, mc_sample_num = 1., event_num = 1, beta = 1):
        super(CTLSTMwrapper, self).__init__()
        self.device = device
        self.model = CTLSTM(hidden_dim = hidden_dim, event_num = event_num, beta = beta, device = device, mc_sample_num = mc_sample_num)
    
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

        return self.model(event_tensor = event_tensor.to(self.device), dtime_tensor = dtime_tensor.to(self.device), 
                          token_num_tensor = token_num_tensor.to(self.device), duration_tensor = duration_tensor.to(self.device), 
                          eval_tag = eval_tag)

    @staticmethod
    def train_step(model, minibatch, optimizer, device, update_or_not):
        ''' Epoch operation in training phase'''
    
        model.train()
        optimizer.zero_grad()
        log_likelihood = model(
            minibatch[0], eval_tag = False
        )
    
        loss = loss_f(
            value = log_likelihood
        )
        loss.backward()
        if update_or_not:
            optimizer.step_and_update_lr()
    
        loss = loss.item()
        fact = minibatch[1].sum()
    
        return loss, fact
    
    @staticmethod
    def evaluation_step(model, minibatch, device):
        ''' Epoch operation in evaluation phase '''
    
        model.eval()
        log_likelihood = model(
            minibatch[0], eval_tag = True
        )
    
        loss = loss_f(
            value = log_likelihood
        )
    
        loss = loss.item()
        fact = minibatch[1].sum()
    
        return loss, fact

def loss_f(value):
    '''
    The definition of loss.
    '''
    loss = -value
    loss = loss.sum(axis = -1)
    
    return loss