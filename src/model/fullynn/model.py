from .submodel import FullyNN
from ..utils import BasicModule
import torch


def check_tensor(x):
    assert (x < 0).cpu().numpy().any() == False


class FullyNNModel(BasicModule):
    def __init__(self, d_history,
                 d_intensity,
                 dropout,
                 rnn_layers,
                 mlp_layers,
                 nonlinear,
                 device):
        super(FullyNNModel, self).__init__()
        self.device = device
        self.model = FullyNN(d_history = d_history, d_intensity = d_intensity,
                             dropout = dropout, rnn_layers = rnn_layers, mlp_layers = mlp_layers,
                             nonlinear = nonlinear, device = device)

    def forward(self, input_time):
        time_history, time_next = input_time.clone()[:, :-1], input_time.clone()[:, 1:]
        time_history = time_history.unsqueeze(-1)                              # [batch_size, seq_len, 1]
        time_next = time_next.unsqueeze(-1)                                    # [batch_size, seq_len, 1]
        time_next.requires_grad = True

        integral = self.model(time_history, time_next)                         # [batch_size, seq_len, 1]

        intensity = torch.autograd.grad(
            outputs=integral,
            inputs=time_next,
            grad_outputs=torch.ones_like(integral),
            create_graph=True,
        )[0]                                                                   # [batch_size, seq_len, 1]
        check_tensor(intensity)
        assert intensity.shape == integral.shape
        time_next.requires_grad = False

        return integral, intensity

    def train_step(model, minibatch, device):
        ''' 
        Epoch operation in training phase.
        The input minibatch comprise time sequences.

        Args:
            minibatch: [batch_size, seq_len]
        '''
    
        model.train()
        intensity_integral, intensity = model(         # [batch_size, seq_len, 1]
                minibatch[0]
        )
    
        loss = loss_f(
            intensity=intensity, intensity_integral=intensity_integral
        )
        loss.backward()
    
        loss = loss.item()
        fact = minibatch[1].sum()
        
        return loss, fact
    
    def evaluation_step(model, minibatch, device):
        ''' Epoch operation in evaluation phase '''
    
        model.eval()
        intensity_integral, intensity = model(         # [batch_size, seq_len, 1]
            minibatch[0]
        )
    
        loss = loss_f(
            intensity=intensity, intensity_integral=intensity_integral
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
        return [evaluation_report[-1].item(), test_report[-1].item()]
    
    metric_number = 2 # metric number is the length of the output of choose_metric

    # All methods not required by BasicModule are intensity plotter exclusive.
    def function_prober(self, input_time, resolution):
        '''
        Args:
        time: [batch_size(always 1), seq_len + 1]
              The original dataset records. 
        resolution: int
                    How many interpretive numbers we have between an event interval?
        '''
        batch_size, seq_len_plus_1 = input_time.shape
        self.model.eval()
        time_history, time_next = input_time.clone()[:, :-1], input_time.clone()[:, 1:]
        time_history = time_history.unsqueeze(-1)                              # [batch_size, seq_len, 1]
        time_next = time_next.unsqueeze(-1)                                    # [batch_size, seq_len, 1]

        expand_integral, expand_intensity, timestamp = self.model.integral_intensity(time_history, time_next, resolution)
                                                                               # [batch_size, seq_len * resolution, 1]
        check_tensor(expand_intensity)
        assert expand_intensity.shape == expand_integral.shape
        expand_intensity = expand_intensity.squeeze(-1)                        # [batch_size, seq_len * resolution]
        expand_integral = expand_integral.squeeze(-1)                          # [batch_size, seq_len * resolution]

        return expand_integral, expand_intensity, timestamp

    def probe_intensity(self, data):
        self.model.eval()
        return self.forward(data[0])
        

def loss_f(intensity, intensity_integral):
    '''
    The definition of loss.

    Args:
        intensity:          [batch_size, seq_len - 1, 1]
        intensity_integral: [batch_size, seq_len - 1, 1]
    '''
    intensity, intensity_integral = intensity.squeeze(), intensity_integral.squeeze()

    log_intensity = torch.log(intensity)
    log_p = log_intensity - intensity_integral

    loss = -log_p
    loss = torch.clamp(loss, max=10)
    loss = torch.sum(loss)
    return loss