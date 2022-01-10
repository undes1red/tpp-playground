from .submodel import DynamicMLP
from ..utils import BasicModule
import torch

def check_tensor(x):
    assert (x < 0).cpu().numpy().any() == False


class TemporalModel(BasicModule):
    def __init__(self, d_history,
                 d_intensity,
                 dropout,
                 rnn_layers,
                 mlp_layers,
                 time_activation,
                 no_time_weight,
                 no_scale,
                 device,
                 mae_threshold = 2,
                 num_events = 1,
                 weight_gen_min = None,
                 time_weight_min = None):
        super(TemporalModel, self).__init__()
        self.device = device
        self.num_events = num_events
        self.mae_threshold = mae_threshold
        self.model = DynamicMLP(d_history = d_history, d_intensity = d_intensity, dropout = dropout, weight_gen_min = weight_gen_min,
                                time_weight_min = time_weight_min,num_layers = rnn_layers, mlp_layers = mlp_layers, time_activation = time_activation,
                                no_time_weight = no_time_weight, no_scale = no_scale, num_events = num_events, device = device)

    def forward(self, time, events, evaluate = False):
        time_history, time_next = self.divide_history_and_next(time, unsqueeze = True)
                                                                               # [batch_size, seq_len, 1]
        events_history, events_next = self.divide_history_and_next(events, unsqueeze = False)
                                                                               # [batch_size, seq_len]
        time_next.requires_grad = True

        integral, events_probability = self.model(events_history, time_history, time_next)
                                                                               # [batch_size, seq_len] & [batch_size, seq_len, 10]

        mae = 0
        if evaluate:
            mae = self.mean_absolute_error(time_history = time_history, events_history = events_history, time_next = time_next)
        
        intensity = torch.autograd.grad(
            outputs=integral,
            inputs=time_next,
            grad_outputs=torch.ones_like(integral),
            create_graph=True,
        )[0].squeeze(-1)                                                       # [batch_size, seq_len]
        check_tensor(intensity)
        time_next.requires_grad = False

        time_loss = self.time_loss_f(intensity = intensity, intensity_integral = integral)
        event_loss = torch.nn.functional.cross_entropy(events_probability.reshape(-1, self.num_events), \
                                                       events_next.reshape(-1).long(), reduction = 'sum')

        return time_loss, mae, event_loss

    def evaluate(self, event_history, time_history, timestamp):
        integral, _ = self.model(event_history, time_history, timestamp)       # [batch_size, seq_len]

        return integral
    
    def divide_history_and_next(self, input, unsqueeze = False):
        input_history, input_next = input.clone()[:, :-1], input.clone()[:, 1:]
        if unsqueeze:
            input_history = input_history.unsqueeze(-1)                        # [batch_size, seq_len, 1]
            input_next = input_next.unsqueeze(-1)                              # [batch_size, seq_len, 1]

        return input_history, input_next
    
    def mean_absolute_error(self, events_history, time_history, time_next):
        '''
        The input should be the original minibatch
        MAE evaluation part, dwg and fullynn exclusive
        '''
        def bisect_target(event_history, time_history, taus):
            return self.evaluate(event_history, time_history, taus).unsqueeze(-1) - \
                   torch.log(torch.tensor(self.mae_threshold, device = time_history.device))
                                                                               # [batch_size, seq_len, 1]
        
        def median_prediction(events_history, time_history, l, r):
            for _ in range(30):
                c = (l + r)/2
                v = bisect_target(events_history, time_history, c)
                l = torch.where(v < 0, c, l)
                r = torch.where(v >= 0, c, r)

            return (l + r)/2
        
        l = 0.0001*torch.ones_like(time_history, dtype = torch.float32)        # [batch_size, seq_len, 1]
        r = 6500.0*torch.ones_like(time_history, dtype = torch.float32)        # [batch_size, seq_len, 1]
        tau_pred = median_prediction(events_history, time_history, l, r)
        gap = tau_pred - time_next
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
        input_time, input_events, _, _ = input_data
        time_history, time_next = self.divide_history_and_next(input_time, unsqueeze = True)
                                                                               # 2 * [batch_size, seq_len, 1]
        event_history, event_next = self.divide_history_and_next(input_events, unsqueeze = False)
                                                                               # 2 * [batch_size, seq_len]

        integral, intensity, timestamp = self.model.intensity(time_history = time_history,\
                                                              time_next = time_next, event_history = event_history,\
                                                              resolution = resolution)
                                                                               # 3 * [batch_size, seq_len * resolution, 1]
        check_tensor(intensity)
        assert intensity.shape == integral.shape
        intensity = intensity.squeeze(-1)                                      # [batch_size, seq_len * resolution]
        integral = integral.squeeze(-1)                                        # [batch_size, seq_len * resolution]

        return integral, intensity, timestamp
    
    def model_prober(self, input_data, resolution):
        '''
        Args:
        time: [batch_size(always 1), seq_len + 1]
              The original dataset records. 
        resolution: int
                    How many interpretive numbers we have between an event interval?
        device: conduct all computations on cpu, gpu, or other devices
        '''
        self.model.eval()
        input_time, input_events, _, _ = input_data
        time_history, time_next = self.divide_history_and_next(input_time, unsqueeze = True)
                                                                               # 2 * [batch_size, seq_len, 1]
        event_history, event_next = self.divide_history_and_next(input_events, unsqueeze = False)
                                                                               # 2 * [batch_size, seq_len]

        probed_results, timestamp = self.model.model_probe_function(time_history = time_history,\
                                                                    time_next = time_next, event_history = event_history,\
                                                                    resolution = resolution)
                                                                               # 3 * [batch_size, seq_len * resolution, 1]

        return probed_results, timestamp
    
    def time_loss_f(self, intensity, intensity_integral):
        '''
        The definition of loss.
        '''    
        log_intensity = torch.log(intensity)
        log_p = log_intensity - intensity_integral
        
        loss = -log_p
        loss = torch.clamp(loss, max=10)
        loss = torch.sum(loss)
        return loss
    
    '''
    Static methods
    '''
    def train_step(model, minibatch, device):
        ''' Epoch operation in training phase'''
        model.train()

        time_loss, mae, events_loss = model(
            time = minibatch[0], events = minibatch[1]
        )                                                                      # 2 * [batch_size, seq_len], int, [batch_size, seq_len]

        loss = time_loss + events_loss
        loss.backward()
    
        time_loss = time_loss.item()
        events_loss = events_loss.item()
        fact = minibatch[-1].sum()
    
        return time_loss, fact, events_loss
    
    def evaluation_step(model, minibatch, device):
        ''' Epoch operation in evaluation phase '''
    
        model.eval()
        time_loss, mae, events_loss = model(
            time = minibatch[0], events = minibatch[1], evaluate = True
        )                                                                      # [batch_size, seq_len, 1]
    
    
        time_loss = time_loss.item()
        events_loss = events_loss.item()
        fact = minibatch[-1].sum()

        return time_loss, fact, events_loss, mae

    def postprocess(input):
        def train_postprocess(input):
            '''
            Training process
            [absolute loss, relative loss]
            '''
            return [input[0], input[0] - input[1], input[2]]
        
        def test_postprocess(input):
            '''
            Evaluation process
            [absolute loss, relative loss, mae value]
            '''
            return [input[0], input[0] - input[1], input[2], input[3]]
        
        return (train_postprocess(input) if len(input) == 3 else test_postprocess(input))

    def log_print_format(input):
        def train_log_print_format(input):
            format_dict = {}
            format_dict['absolute_loss'] = input[0]
            format_dict['relative_loss'] = input[1]
            format_dict['event_loss'] = input[2]
            format_dict['num_format'] = {'absolute_loss': ':6.5f', 'relative_loss': ':6.5f', 'event_loss': ':6.5f'}
            return format_dict

        def test_log_print_format(input):
            format_dict = {}
            format_dict['absolute_loss'] = input[0]
            format_dict['relative_loss'] = input[1]
            format_dict['event_loss'] = input[2]
            format_dict['mae'] = input[3]
            format_dict['num_format'] = {'absolute_loss': ':6.5f', 'relative_loss': ':6.5f', 'event_loss': ':6.5f', 'mae': ':2.8f'}
            return format_dict
        
        return (train_log_print_format(input) if len(input) == 3 else test_log_print_format(input))

    format_dict_length = 4
    
    logfile_format = {'step': '', 'absolute loss': ':6.5f', 'relative loss': ':6.5f', 'event loss': ':6.5f'}

    def logfile_print_format(input):
        format_dict = {}
        format_dict['absolute loss'] = input[0]
        format_dict['relative loss'] = input[1]
        format_dict['event loss'] = input[2]
        return format_dict
    
    def choose_metric(evaluation_report, test_report):
        '''
        [relative loss on evaluation dataset, relative loss on test dataset]
        '''
        return [evaluation_report[1].item(), test_report[1].item()]
    
    metric_number = 2 # metric number is the length of the output of choose_metric
