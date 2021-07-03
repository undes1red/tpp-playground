from .log_norm_mix import LogNormMix

import torch.nn as nn

class ifl(nn.Module):
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

    @staticmethod
    def train_step(model, minibatch, optimizer, device, update_or_not):
        ''' Epoch operation in training phase'''
    
        model.train()
            
        log_likelihood = model(minibatch[0])
    
        loss = loss_f(log_likelihood)
        loss.backward()
        if update_or_not:
            optimizer.step_and_update_lr()
            optimizer.zero_grad()
    
        loss = loss.item()
        fact = minibatch[1].sum()
    
        return loss, fact
    
    @staticmethod
    def evaluation_step(model, minibatch, device):
        ''' Epoch operation in evaluation phase '''
    
        model.eval()
        log_likelihood = model(minibatch[0])
    
        loss = loss_f(log_likelihood)
    
        loss = loss.item()
        fact = minibatch[1].sum()
    
        return loss, fact

    @staticmethod
    def postprocess(input):
        return [input[0], input[0] - input[1]]


def loss_f(loglik):
    '''
    The definition of loss.
    '''
    return loglik.mul(-1.0).sum()
