from .submodel import DynamicMLP
from ..utils import BasicModule
from einops import rearrange, reduce, repeat, pack, unpack

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
                 num_events,
                 event_toggle = False,
                 mae_threshold = 2,
                 weight_gen_min = None,
                 time_weight_min = None):
        super(TemporalModel, self).__init__()
        self.device = device
        self.event_toggle = event_toggle
        self.num_events = num_events
        self.mae_threshold = mae_threshold

        '''
        Model created here.
        '''
        self.model = DynamicMLP(d_history = d_history, d_intensity = d_intensity, dropout = dropout, weight_gen_min = weight_gen_min,
                                time_weight_min = time_weight_min,num_layers = rnn_layers, mlp_layers = mlp_layers, time_activation = time_activation,
                                no_time_weight = no_time_weight, no_scale = no_scale, num_events = num_events, event_toggle = event_toggle, device = device)

    def forward(self, time, events, mask, mean, var, evaluate = False):
        time_history, time_next = self.divide_history_and_next(time)           # 2 * [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(events)     # 2 * [batch_size, seq_len]
        _, mask_next = self.divide_history_and_next(mask)                      # [batch_size, seq_len]

        mae = 0
        if evaluate:
            mae = self.mean_absolute_error(time_history = time_history, events_history = events_history,\
                                           time_next = time_next, mask = mask_next, mean = mean, var = var)
        
        if self.event_toggle:
            time_next = repeat(time_next, 'b s -> b s ne', ne = self.num_events)
                                                                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]

        time_next.requires_grad = True
        integral = self.model(events_history, time_history, time_next, mean, var)
                                                                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
        intensity = torch.autograd.grad(
            outputs=integral,
            inputs=time_next,
            grad_outputs=torch.ones_like(integral),
            create_graph=True
        )[0]                                                                   # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
        check_tensor(intensity)
        time_next.requires_grad = False

        if self.event_toggle:
            log_intensity = torch.log(intensity)                               # [batch_size, seq_len, num_events]
            events_probability = torch.nn.functional.softmax(log_intensity, dim = -1)
                                                                               # [batch_size, seq_len, num_events]
            events_loss = torch.nn.functional.cross_entropy(rearrange(events_probability, 'b s ne -> b ne s'), \
                                                           events_next.long(), reduction = 'none')
                                                                               # [batch_size, seq_len]
            events_loss = events_loss * mask_next
            events_loss = events_loss.sum()

            time_loss = self.time_loss_f(intensity = intensity, intensity_integral = integral, 
                                         events_next = events_next, mask = mask_next)
        else:
            events_loss = torch.tensor(0., dtype = torch.float32, device = self.device)
            time_loss = self.time_loss_f(intensity = intensity, intensity_integral = integral, mask = mask_next)

        the_number_of_events = mask_next.sum()
        return time_loss, mae, events_loss, the_number_of_events

    def evaluate(self, events_history, time_history, timestamp, mean, var):
        if self.event_toggle:
            timestamp = repeat(timestamp, ' ... -> ... ne', ne = self.num_events)
        integral = self.model(events_history, time_history, timestamp, mean, var)
                                                                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
        if self.event_toggle:
            integral = reduce(integral, 'b s ne -> b s', 'sum')                # [batch_size, seq_len]
        return integral
    
    def divide_history_and_next(self, input):
        input_history, input_next = input[:, :-1].clone(), input[:, 1:].clone()
        return input_history, input_next
    
    def mean_absolute_error(self, events_history, time_history, time_next, mask, mean, var):
        '''
        The input should be the original minibatch
        MAE evaluation part, dwg and fullynn exclusive
        '''
        def bisect_target(events_history, time_history, taus, mean, var):
            return self.evaluate(events_history, time_history, taus, mean, var) - \
                   torch.log(torch.tensor(self.mae_threshold, device = self.device))
                                                                               # [batch_size, seq_len]
        
        def median_prediction(events_history, time_history, l, r, mean, var):
            for _ in range(50):
                c = (l + r)/2
                v = bisect_target(events_history, time_history, c, mean, var)
                l = torch.where(v < 0, c, l)
                r = torch.where(v >= 0, c, r)

            return (l + r)/2
        
        l = 0.0001*torch.ones_like(time_history, dtype = torch.float32)        # [batch_size, seq_len]
        r = 1e6*torch.ones_like(time_history, dtype = torch.float32)           # [batch_size, seq_len]
        tau_pred = median_prediction(events_history, time_history, l, r, mean, var)
                                                                               # [batch_size, seq_len]
        gap = (tau_pred - time_next) * mask                                    # [batch_size, seq_len]
        gap_mean = torch.sum(torch.abs(gap)) / mask.sum()
        return gap_mean.item()

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
        [input_time, input_events, _, _, _] , (mean, var) = input_data
        time_history, time_next = self.divide_history_and_next(input_time)     # 2 * [batch_size, seq_len]
        events_history, _ = self.divide_history_and_next(input_events)         # 2 * [batch_size, seq_len]

        integral, intensity, timestamp = self.model.intensity_integral(
                                                time_history = time_history,\
                                                time_next = time_next, events_history = events_history,\
                                                resolution = resolution, mean = mean, var = var
                                        )                                      # 2 * [batch_size, seq_len * resolution, num_events] + [batch_size, seq_len * resolution] if we need events else 3 * [batch_size, seq_len * resolution]
        check_tensor(intensity)
        assert intensity.shape == integral.shape

        '''
        Compatibility with the ploter
        '''
        if self.event_toggle:
            integral = integral.sum(dim = -1)                                  # [batch_size, seq_len * resolution]
            intensity = intensity.sum(dim = -1)                                # [batch_size, seq_len * resolution]

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
        [input_time, input_events, _, _, _], (mean, var) = input_data
        time_history, time_next = self.divide_history_and_next(input_time)     # 2 * [batch_size, seq_len]
        events_history, _ = self.divide_history_and_next(input_events)         # 2 * [batch_size, seq_len]

        probed_results, timestamp = self.model.model_probe_function(time_history = time_history,\
                                                                    time_next = time_next, events_history = events_history,\
                                                                    resolution = resolution, mean = mean, var = var)
                                                                               # 2 * [batch_size, seq_len * resolution, num_events] + [batch_size, seq_len * resolution] if we need events else 3 * [batch_size, seq_len * resolution]

        return (probed_results,), timestamp
    
    def time_loss_f(self, intensity, intensity_integral, mask, events_next = None):
        '''
        The definition of loss.
        '''
        if events_next is not None:
            '''
            self.event_toggle = True
            '''
            event_mask = torch.nn.functional.one_hot(events_next.long(), num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]
            intensity = intensity * event_mask                                 # [batch_size, seq_len, num_events]
            intensity = reduce(intensity, 'b s ne -> b s', 'sum')              # [batch_size, seq_len]
            intensity_integral = reduce(intensity_integral, 'b s ne -> b s', 'sum')
                                                                               # [batch_size, seq_len]

        log_intensity = torch.log(intensity + 1e-6)                            # [batch_size, seq_len]
        log_p = log_intensity - intensity_integral                             # [batch_size, seq_len]
        
        loss = -log_p
        loss = torch.clamp(loss * mask, max = 15)                              # [batch_size, seq_len]
        loss = torch.sum(loss)
        return loss
    
    '''
    Static methods
    '''
    def train_step(model, minibatch, device):
        '''
        Epoch operation in training phase

        minibatch: [time_seq, event_seq, score, mask]
        '''
        model.train()
        [time_seq, event_seq, score, mask], (mean, var) = minibatch

        time_loss, mae, events_loss, the_number_of_events = model(
            time = time_seq, events = event_seq, mask = mask, mean = mean, var = var
        )                                                                      # 2 * [batch_size, seq_len], int, [batch_size, seq_len]

        loss = time_loss
        loss.backward()
    
        time_loss = time_loss.item() / the_number_of_events
        events_loss = events_loss.item() / the_number_of_events
        fact = score.sum() / the_number_of_events
    
        return time_loss, fact, events_loss
    
    def evaluation_step(model, minibatch, device):
        ''' Epoch operation in evaluation phase '''
    
        model.eval()
        [time_seq, event_seq, score, mask], (mean, var) = minibatch

        time_loss, mae, events_loss, the_number_of_events = model(
            time = time_seq, events = event_seq, mask = mask, evaluate = True, mean = mean, var = var
        )                                                                      # [batch_size, seq_len, 1]
    
    
        time_loss = time_loss.item() / the_number_of_events
        events_loss = events_loss.item() / the_number_of_events
        fact = score.sum() / the_number_of_events

        return time_loss, fact, events_loss, mae

    def postprocess(input, procedure):
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
        
        return (train_postprocess(input) if procedure == 'Training' else test_postprocess(input))

    def log_print_format(input, procedure):
        def train_log_print_format(input):
            format_dict = {}
            format_dict['absolute_loss'] = input[0]
            format_dict['relative_loss'] = input[1]
            format_dict['events_loss'] = input[2]
            format_dict['num_format'] = {'absolute_loss': ':6.5f', 'relative_loss': ':6.5f', 'events_loss': ':6.5f'}
            return format_dict

        def test_log_print_format(input):
            format_dict = {}
            format_dict['absolute_loss'] = input[0]
            format_dict['relative_loss'] = input[1]
            format_dict['events_loss'] = input[2]
            format_dict['mae'] = input[3]
            format_dict['num_format'] = {'absolute_loss': ':6.5f', 'relative_loss': ':6.5f', 'events_loss': ':6.5f', 'mae': ':2.8f'}
            return format_dict
        
        return (train_log_print_format(input) if procedure == 'Training' else test_log_print_format(input))

    format_dict_length = 4
    
    logfile_format = {'step': '', 'absolute loss': ':6.5f', 'relative loss': ':6.5f', 'events loss': ':6.5f'}

    def logfile_print_format(input):
        format_dict = {}
        format_dict['absolute loss'] = input[0]
        format_dict['relative loss'] = input[1]
        format_dict['events loss'] = input[2]
        return format_dict
    
    def choose_metric(evaluation_report, test_report):
        '''
        [relative loss on evaluation dataset, relative loss on test dataset]
        '''
        return [evaluation_report[1].item(), test_report[1].item()]
    
    metric_number = 2 # metric number is the length of the output of choose_metric
