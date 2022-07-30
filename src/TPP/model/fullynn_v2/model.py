from .submodel import FullyNN_v2
from ..utils import BasicModule
import torch


def check_tensor(x):
    assert (x < 0).cpu().numpy().any() == False

'''
Q1: why without bottleneck, the intensity function for each type of event fails to learn?
A: The reason might still be the activation, because we detect that although the norms of gradients are similar, the variances
are significantly different, which is over 100 times larger when a bottleneck layer is applied.
'''

class FullyNN2Model(BasicModule):
    def __init__(self, d_history,
                 num_events,
                 d_intensity,
                 dropout,
                 history_module_layers,
                 integral_module_layers,
                 mlp_layers,
                 nonlinear,
                 device,
                 history_module = 'LSTM',
                 n_head = 0,
                 mae_threshold = 2,
                 event_toggle = False,
                 reverse_bottleneck = True,
                 no_bottleneck = False, no_norm = False, no_activate = False,
                 wq_nonneg = False, wk_nonneg = False, wv_nonneg = False):
        super(FullyNN2Model, self).__init__()
        self.device = device
        self.mae_threshold = mae_threshold
        self.num_events = num_events
        self.event_toggle = event_toggle
        self.model = FullyNN_v2(d_history = d_history, d_intensity = d_intensity, num_events = num_events, 
                                integral_module_layers = integral_module_layers, dropout = dropout, history_module = history_module,
                                history_module_layers = history_module_layers, mlp_layers = mlp_layers, nonlinear = nonlinear,
                                event_toggle = event_toggle, n_head = n_head, wq_nonneg = wq_nonneg, wk_nonneg = wk_nonneg,
                                wv_nonneg = wv_nonneg, device = device)

    def forward(self, history_time, history_event, result, score, event, mask, mean, var, evaluate = False):
        '''
        Inputs:
        1. history_time: [batch_size, seq_len, history_length]
           history time sequences for history encoder
        2. history_event:[batch_size, seq_len, history_length]
           history event sequences for history encoder (model can decide if it should use it by event_toggle)
        3. result:       [batch_size, seq_len]
           the value of t-t_l
        4. score:        [batch_size, seq_len]
           ideal loss value
        5. event:        [batch_size, seq_len]
           target event
        6. mask:         [batch_size, seq_len, history_length]
           mask matrix to filter out padding events from the original sequences. 0 means should be masked.
        7. mean:         int
        8. var:          int
           For data normalization.

        First, this model doesn't use intensity function to predict the event. Maybe I could find a better way to do this, but not now.
        '''
        batch_size, seq_len, history_length = history_time.shape
        number_of_event = batch_size * seq_len

        mae = 0
        if evaluate:
            mae = self.mean_absolute_error(history_time = history_time, history_event = history_event,
                                           result = result, mask = mask, mean = mean, var = var)

        # preparing for multi-event training when needed
        result.requires_grad = True
        
        if self.event_toggle:
            integral, event_pred = self.model(history_time, history_event, result, mean = mean, var = var, mask = mask)
                                                                               # [batch_size, seq_len] + [batch_size, seq_len, num_events]
        else:
            integral = self.model(history_time, history_event, result, mean = mean, var = var, mask = mask)
                                                                               # [batch_size, seq_len]
        
        # Intensity values and their sum.
        intensity = torch.autograd.grad(
            outputs = integral,
            inputs = result,
            grad_outputs = torch.ones_like(integral),
            create_graph = True,
        )[0]
        # Check whether the negative gradient only occurs at index 0, or it's an architecture-level bug.
        check_tensor(intensity)
        
        '''
        This part is only available when evnet_toggle = True
        '''
        if self.event_toggle:
            event_loss = torch.nn.functional.cross_entropy(event_pred.reshape(-1, self.num_events), \
                                                        event.long().flatten(), reduction = 'sum')
        else:
            event_loss = torch.tensor(0., dtype = torch.float32)
    
        assert intensity.shape == integral.shape
        result.requires_grad = False
        time_loss = self.time_loss_f(intensity = intensity, intensity_integral = integral)

        return time_loss, event_loss, mae, number_of_event

    def evaluate(self, history_time, history_event, taus, mean, var, mask):
        if self.event_toggle:
            integral, _ = self.model(history_time, history_event, taus, mean = mean, var = var, mask = mask)
                                                                               # [batch_size, seq_len]
        else:
            integral = self.model(history_time, history_event, taus, mean = mean, var = var, mask = mask)
                                                                               # [batch_size, seq_len]
        return integral

    def divide_history_and_next(self, input, unsqueeze = False):
        input_history, input_next = input.clone()[:, :-1], input.clone()[:, 1:]
        if unsqueeze:
            input_history = input_history.unsqueeze(-1)                        # [batch_size, seq_len, 1]
            input_next = input_next.unsqueeze(-1)                              # [batch_size, seq_len, 1]
        return input_history, input_next

    def mean_absolute_error(self, history_time, history_event, result, mean, var, mask):
        '''
        The input should be the original minibatch
        MAE evaluation part, dwg and fullynn exclusive

        Inputs:
        1. history_time: [batch_size, seq_len, history_length]
           history time sequences for history encoder
        2. history_event:[batch_size, seq_len, history_length]
           history event sequences for history encoder (model can decide if it should use it by event_toggle)
        3. result:       [batch_size, seq_len]
           the value of t-t_l
        4. mask:         [batch_size, seq_len, history_length]
           mask matrix to filter out padding events from the original sequences. 0 means should be masked.
        5. mean:         int
        6. var:          int
           For data normalization.
        '''
        def bisect_target(history_time, history_event, taus, mean, var):
            return self.evaluate(history_time, history_event, taus, mean, var, mask) - \
                   torch.log(torch.tensor(self.mae_threshold, device = history_time.device))
            
        def median_prediction(events_history, time_history, l, r, mean, var):
            for _ in range(50):
                c = (l + r)/2
                v = bisect_target(events_history, time_history, c, mean, var)
                l = torch.where(v < 0, c, l)
                r = torch.where(v >= 0, c, r)

            return (l + r)/2
        
        l = 0.0001*torch.ones_like(result, dtype = torch.float32, device = self.device)
                                                                               # [batch_size, seq_len]
        r = 1e6*torch.ones_like(result, dtype = torch.float32, device = self.device)
                                                                               # [batch_size, seq_len]
        tau_pred = median_prediction(history_time, history_event, l, r, mean, var)
        gap = tau_pred - result
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
        (history_time, history_event, result, _, _, _, mask), (mean, var)  = input_data

        expand_integral, expand_intensity, timestamp = \
                        self.model.integral_intensity(history_time = history_time, \
                                                      history_event = history_event, \
                                                      result = result, resolution = resolution, \
                                                      mean = mean, var = var, mask = mask)
                                                                               # 3 * [batch_size, seq_len * resolution]

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
        input_time, input_events, _, mask = input_data[0][:4]
        mean, var = input_data[1]

        time_history, time_next = self.divide_history_and_next(input_time, unsqueeze = True)
                                                                               # [batch_size, seq_len, 1]
        events_history, events_next = self.divide_history_and_next(input_events, unsqueeze = False)
                                                                               # [batch_size, seq_len]
        _, mask_next = self.divide_history_and_next(mask, unsqueeze = False)   # [batch_size, seq_len]


        probed_results, timestamp = self.model.model_probe_function(events_history, time_history, \
                                                                    time_next, resolution, mean, var, mask_next)
                                                                               # [batch_size, seq_len * resolution, 1] * n

        return probed_results, timestamp
    
    def time_loss_f(self, intensity, intensity_integral):
        '''
        The definition of loss.
    
        Args:
            intensity:          [batch_size, seq_len]
            intensity_integral: [batch_size, seq_len]
        '''
    
        log_intensity = torch.log(intensity + 1e-6)
        log_p = log_intensity - intensity_integral
    
        loss = -log_p
        loss = torch.clamp(loss, max=15)
        loss = torch.sum(loss)
        return loss

    '''
    All static methods
    '''
    def train_step(model, minibatch, device):
        ''' 
        Epoch operation in training phase.
        The input minibatch comprise time sequences.

        Args:
            minibatch: [batch_size, seq_len, *]
                       contains [history_time, history_event, result, score, event, mask], (mean, var)
        '''
    
        model.train()
        [history_time, history_event, result, score, event, mask], (mean, var) = minibatch
        time_loss, events_loss, mae, the_number_of_events = model(         
                history_time = history_time, history_event = history_event, result = result, 
                score = score, event = event, mask = mask, mean = mean, var = var
        )

        loss = time_loss + events_loss
        loss.backward()
    
        time_loss = time_loss.item() / the_number_of_events
        events_loss = events_loss.item() / the_number_of_events
        fact = score.sum().item() / the_number_of_events
        
        return [time_loss, fact, events_loss]
    
    def evaluation_step(model, minibatch, device):
        ''' Epoch operation in evaluation phase '''
    
        model.eval()
        [history_time, history_event, result, score, event, mask], (mean, var) = minibatch
        time_loss, events_loss, mae, the_number_of_events = model(
                history_time = history_time, history_event = history_event, result = result, 
                score = score, event = event, mask = mask, mean = mean, var = var, evaluate = True
        )
    
        time_loss = time_loss.item() / the_number_of_events
        events_loss = events_loss.item() / the_number_of_events
        fact = score.sum().item() / the_number_of_events
        
        return [time_loss, fact, events_loss, mae]

    def postprocess(input, procedure):
        def train_postprocess(input):
            '''
            Training process
            [absolute loss, relative loss, events loss]
            '''
            return [input[0], input[0] - input[1], input[2]]
        
        def test_postprocess(input):
            '''
            Evaluation process
            [absolute loss, relative loss, events loss, mae value]
            '''
            return [input[0], input[0] - input[1], input[2], input[3]]
        
        return (train_postprocess(input) if procedure == 'Training' else test_postprocess(input))
    
    def log_print_format(input, procedure):
        def train_log_print_format(input):
            format_dict = {}
            format_dict['absolute_loss'] = input[0]
            format_dict['relative_loss'] = input[1]
            format_dict['events_loss'] = input[2]
            format_dict['num_format'] = {'absolute_loss': ':6.5f', 'relative_loss': ':6.5f', \
                                         'events_loss': ':6.5f'}
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
        [relative loss on evaluation dataset, relative loss on test dataset, event loss on test dataset]
        '''
        return [evaluation_report[1], test_report[1], test_report[2]]
    
    metric_number = 3 # metric number is the length of the output of choose_metric