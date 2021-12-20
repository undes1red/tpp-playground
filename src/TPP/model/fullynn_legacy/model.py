from .submodel import FullyNN
from ..utils import BasicModule
import torch


def check_tensor(x):
    assert (x < 0).cpu().numpy().any() == False


class FullyNNModel_legacy(BasicModule):
    def __init__(self, d_history,
                 d_intensity,
                 dropout,
                 rnn_layers,
                 mlp_layers,
                 nonlinear,
                 device):
        super(FullyNNModel_legacy, self).__init__()
        self.device = device
        self.model = FullyNN(d_history = d_history, d_intensity = d_intensity,
                             dropout = dropout, rnn_layers = rnn_layers, mlp_layers = mlp_layers,
                             nonlinear = nonlinear, device = device)

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

    def train_step(model, minibatch, device):
        ''' Epoch operation in training phase'''
    
        model.train()
        intensity_integral, intensity = model(
                minibatch[0].to(device), minibatch[1].to(device)
        )
    
        loss = loss_f(
            intensity=intensity, intensity_integral=intensity_integral
        )
        loss.backward()
    
        loss = loss.item()
        fact = minibatch[2].sum()
        
        return loss, fact
    
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

    def postprocess(input):
        return [input[0], input[0] - input[1]]
    
    def log_print_format(input):
        format_dict = {}
        format_dict['absolute_loss'] = input[0]
        format_dict['relative_loss'] = input[1]
        format_dict['num_format'] = {'absolute_loss': ':8.5f', 'relative_loss': ':8.5f'}
        return format_dict
    
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
        return [evaluation_report[-1].item(), test_report[-1].item()]
    
    metric_number = 2 # metric number is the length of the output of choose_metric

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