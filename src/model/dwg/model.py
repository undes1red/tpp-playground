from .submodel import DynamicMLP
import torch
import torch.nn as nn


def check_tensor(x):
    assert (x < 0).cpu().numpy().any() == False

class TemporalModel(nn.Module):
    def __init__(self, d_history,
                 d_intensity,
                 dropout,
                 rnn_layers,
                 mlp_layers,
                 device):
        super(TemporalModel, self).__init__()
        self.device = device

        self.model = DynamicMLP(d_history, d_intensity,
                                dropout, rnn_layers, mlp_layers, device = device)

    def forward(self, input_time, input_result):
        input_result.requires_grad = True

        integral = self.model(input_time, input_result)

        intensity = torch.autograd.grad(
            outputs=integral,
            inputs=input_result,
            grad_outputs=torch.ones_like(integral),
            create_graph=True,
        )[0]
        check_tensor(intensity)

        input_result.requires_grad = False

        assert intensity.shape == input_result.shape

        return integral, intensity
    
    @staticmethod
    def train_step(model, minibatch, optimizer, device):
        ''' Epoch operation in training phase'''
        model.train()
        optimizer.zero_grad()
        intensity_integral, intensity = model(
                minibatch[0].to(device), minibatch[1].to(device)
        )
    
        loss = loss_f(
            intensity=intensity, intensity_integral=intensity_integral
        )
        loss.backward()
        optimizer.step_and_update_lr()
    
        loss = loss.item()
        fact = minibatch[2].sum()
    
        return loss, fact
    
    @staticmethod
    def evaluation_step(model, minibatch, device):
        ''' Epoch operation in evaluation phase '''
    
        model.eval()
        intensity_integral, intensity = model(
            minibatch[0].to(device), minibatch[1].to(device)
        )
    
        loss = loss_f(
            intensity=intensity, intensity_integral=intensity_integral
        )
    
        loss = loss.item()
        fact = minibatch[2].sum()
    
        return loss, fact



def loss_f(intensity, intensity_integral):
    '''
    The definition of loss.
    '''
    intensity, intensity_integral = intensity.squeeze(), intensity_integral.squeeze()

    log_intensity = torch.log(intensity)
    log_p = log_intensity - intensity_integral
    
    loss = -log_p
    loss = torch.clamp(loss, max=10)
    loss = torch.sum(loss, dim=0)
    return loss