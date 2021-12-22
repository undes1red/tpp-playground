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

    def forward(self, input_time, evaluate = False):
        time_history, time_next = self.divide_history_and_next(input_time)
        time_next.requires_grad = True

        integral = self.model(time_history, time_next)                         # [batch_size, seq_len, 1]
        mae = 0
        if evaluate:
            mae = self.mean_absolute_error(time_history, time_next)

        intensity = torch.autograd.grad(
            outputs=integral,
            inputs=time_next,
            grad_outputs=torch.ones_like(integral),
            create_graph=True,
        )[0]                                                                   # [batch_size, seq_len, 1]
        check_tensor(intensity)
        assert intensity.shape == integral.shape
        time_next.requires_grad = False

        return integral, intensity, mae

    def divide_history_and_next(self, input_time):
        time_history, time_next = input_time.clone()[:, :-1], input_time.clone()[:, 1:]
        time_history = time_history.unsqueeze(-1)                              # [batch_size, seq_len, 1]
        time_next = time_next.unsqueeze(-1)                                    # [batch_size, seq_len, 1]
        return time_history, time_next

    def mean_absolute_error(self, history, target):
        '''
        The input should be the original minibatch
        MAE evaluation part, dwg and fullynn exclusive
        '''
        def bisect_target(history, taus):
            return self.evaluate(history, taus)[0] - torch.log(torch.tensor(2, device = history.device))
        
        def median_prediction(history, l, r):
            for _ in range(30):
                c = (l + r)/2
                v = bisect_target(history, c)
                l = torch.where(v < 0, c, l)
                r = torch.where(v >= 0, c, r)

            return (l + r)/2
        
        l = 0.0001*torch.ones_like(history, dtype = torch.float32)             # [batch_size, seq_len, 1]
        r = 6500.0*torch.ones_like(history, dtype = torch.float32)             # [batch_size, seq_len, 1]
        tau_pred = median_prediction(history, l, r)
        gap = tau_pred - target
        return torch.mean(torch.abs(gap)).item()

    # All methods not required by BasicModule are intensity plotter exclusive.
    def function_prober(self, input_data, resolution):
        '''
        Args:
        time: [batch_size(always 1), seq_len + 1]
              The original dataset records. 
        resolution: int
                    How many interpretive numbers we have between an event interval?
        '''
        self.model.eval()
        input_time = input_data[0]
        time_history, time_next = input_time.clone()[:, :-1], input_time.clone()[:, 1:]
        time_history = time_history.unsqueeze(-1)                              # [batch_size, seq_len, 1]
        time_next = time_next.unsqueeze(-1)                                    # [batch_size, seq_len, 1]

        expand_integral, expand_intensity, timestamp = self.model.integral_intensity(time_history, time_next, resolution)
                                                                               # [batch_size, seq_len * resolution]
        check_tensor(expand_intensity)
        assert expand_intensity.shape == expand_integral.shape
        return expand_integral, expand_intensity, timestamp

    def model_prober(self, input_data, resolution):
        '''
        Args:
        time: [batch_size(always 1), seq_len + 1]
              The original dataset records. 
        resolution: int
                    How many interpretive numbers we have between an event interval?
        '''
        self.model.eval()
        input_time = input_data[0]
        time_history, time_next = input_time.clone()[:, :-1], input_time.clone()[:, 1:]
        time_history = time_history.unsqueeze(-1)                              # [batch_size, seq_len, 1]
        time_next = time_next.unsqueeze(-1)                                    # [batch_size, seq_len, 1]

        probed_results, timestamp = self.model.model_probe_function(time_history, time_next, resolution)
                                                                               # [batch_size, seq_len * resolution, 1] * n

        return probed_results, timestamp

    '''
    All static methods
    '''
    def train_step(model, minibatch, device):
        ''' 
        Epoch operation in training phase.
        The input minibatch comprise time sequences.

        Args:
            minibatch: [batch_size, seq_len]
        '''
    
        model.train()
        intensity_integral, intensity, mae = model(         # [batch_size, seq_len, 1]
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
        intensity_integral, intensity, mae = model(         # [batch_size, seq_len, 1]
            minibatch[0]
        )
    
        loss = loss_f(
            intensity=intensity, intensity_integral=intensity_integral
        )
    
        loss = loss.item()
        fact = minibatch[1].sum()
    
        return loss, fact, mae

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