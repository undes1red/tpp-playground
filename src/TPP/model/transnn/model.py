from .submodel import TransNN
from ..utils import BasicModule

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from einops import rearrange, reduce, repeat, einsum, pack, unpack

# Multi-head for multi-events?

def check_tensor(x):
    assert (x < 0).cpu().numpy().any() == False

'''
Q1: why without bottleneck, the intensity function for each type of event fails to learn?
A: The reason might still be the activation, because we detect that although the norms of gradients are similar, the variances
are significantly different, which is over 100 times larger when a bottleneck layer is applied.
'''

class TransNNModel(BasicModule):
    def __init__(self, d_history,
                 num_events,
                 d_intensity,
                 dropout,
                 history_module_layers,
                 integral_module_layers,
                 mlp_layers,
                 nonlinear,
                 d_qk,
                 device,
                 history_module = 'LSTM',
                 n_head = 0,
                 mae_threshold = 2,
                 event_toggle = False,
                 wq_nonneg = False, wk_nonneg = False, wv_nonneg = False,
                 negative_loss = False):
        super(TransNNModel, self).__init__()
        self.device = device
        self.mae_threshold = mae_threshold
        self.num_events = num_events
        self.event_toggle = event_toggle
        self.negative_loss = negative_loss

        self.model = TransNN(d_history = d_history, d_intensity = d_intensity, d_qk = d_qk, num_events = num_events, 
                             integral_module_layers = integral_module_layers, dropout = dropout, history_module = history_module,
                             history_module_layers = history_module_layers, mlp_layers = mlp_layers, nonlinear = nonlinear,
                             event_toggle = event_toggle, n_head = n_head, wq_nonneg = wq_nonneg, wk_nonneg = wk_nonneg,
                             wv_nonneg = wv_nonneg, device = device)
        self.event_predictor = nn.Softmax(dim = -1)

    def forward(self, time_seqence, events_seqence, mask, mean, var evaluate = False):
        '''
        Inputs:
        1. time_seqence: [batch_size, seq_len]
           Event time sequences t_i.
        2. event_seqence:[batch_size, seq_len]
           The marker sequences m_i.
        3. mask:         [batch_size, seq_len]
           Mask vectors to filter out padding events from the original sequences. 0 means should be masked.
        7. mean:         int
        8. var:          int
           For data normalization.

        '''
        number_of_event = (mask.sum(dim = -1) > 0).int().sum()

        time_history, time_next = self.divide_history_and_next(time_seqence)
        events_history, events_next = self.divide_history_and_next(events_seqence)
        mask_history, mask_next = self.divide_history_and_next(mask)

        mae = 0
        if evaluate:
            mae = self.mean_absolute_error(history_time = history_time, history_event = history_event,
                                           result = result, mask = mask, mean = mean, var = var)
        
        integral, intensity = self.model(history_time, history_event, result, mean = mean, var = var, mask = mask)
                                                                               # [batch_size, seq_len, num_events] * 2

        check_tensor(intensity)
        if self.event_toggle:
            event_pred = self.event_predictor(intensity)
        
        '''
        This part is only available when evnet_toggle = True
        '''
        if self.event_toggle:
            event_loss = torch.nn.functional.cross_entropy(event_pred.reshape(-1, self.num_events), \
                                                        event.long().flatten(), ignore_index = self.num_events + 1, reduction = 'sum')

            event_mask = ((mask.sum(dim = -1) > 0).int()).reshape(-1)
            event_pred_index = torch.argmax(event_pred.reshape(-1, self.num_events), dim = -1)[event_mask == 1].detach().cpu().numpy()
            event_true = event.long().flatten()[event_mask == 1].detach().cpu().numpy()
            f1 = f1_score(y_true = event_true, y_pred = event_pred_index, average = 'macro')
        else:
            event_loss = torch.tensor(0., dtype = torch.float32)
            f1 = 0

        assert intensity.shape == integral.shape

        time_loss = self.time_loss_f(intensity = intensity, intensity_integral = integral, mask = mask,\
                                     event = event, negative_loss = self.negative_loss)

        return time_loss, event_loss, mae, f1, number_of_event

    def evaluate(self, history_time, history_event, taus, mean, var, mask):
        integral, _ = self.model(history_time, history_event, taus, mean = mean, var = var, mask = mask)
                                                                               # [batch_size, seq_len, num_events]
        return integral.sum(dim = -1)

    def divide_history_and_next(self, input):
        input_history, input_next = input.clone()[:, :-1], input.clone()[:, 1:]
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
        gap_mask = (mask.sum(dim = -1) > 0).int()                              # [batch_size, seq_len]
        gap = (tau_pred - result) * gap_mask
        gap_mean = torch.sum(torch.abs(gap)) / torch.sum(gap_mask)
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
        (history_time, history_event, result, _, _, _, mask), (mean, var)  = input_data

        probed_results, additional_plot, timestamp  = \
                        self.model.model_probe_function(history_time = history_time, \
                                                      history_event = history_event, \
                                                      result = result, resolution = resolution, \
                                                      mean = mean, var = var, mask = mask)
                                                                               # n * [batch_size, seq_len * resolution]
    

        return (probed_results, additional_plot), timestamp
    
    def time_loss_f(self, intensity, intensity_integral, mask, event, negative_loss):
        '''
        The definition of loss.
    
        Args:
            intensity:          [batch_size, seq_len, num_event]
            intensity_integral: [batch_size, seq_len, num_event]
            event:              [batch_size, seq_len]
        '''
        loss_mask = (mask.sum(dim = -1) > 0).int()

        # give an small offset to the intensity function to elude nan output.
        intensity += 1e-9

        # Normal TPP loss
        intensity_mask = nn.functional.one_hot((event * loss_mask).long(), num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_event]

        log_intensity = torch.log(intensity)                                   # [batch_size, seq_len, num_event]
        log_intensity = (log_intensity * intensity_mask).sum(dim = -1)         # [batch_size, seq_len]
        log_p = -log_intensity + intensity_integral.sum(dim = -1)              # [batch_size, seq_len]
    
        loss = torch.sum(log_p * loss_mask)

        # Negative-Contrast Learning Loss
        neg_loss = 0
        if negative_loss:
            sum_of_intensity_and_neg = intensity.sum(dim = -1, keepdim = True) # [batch_size, seq_len, 1]
            log_posterior = - torch.log(intensity / sum_of_intensity_and_neg)
                                                                               # [batch_size, seq_len, num_event]
            log_posterior *= intensity_mask                                    # [batch_size, seq_len, num_event]
            neg_loss = log_posterior.sum(dim = -1) * loss_mask                 # [batch_size, seq_len]
            neg_loss = torch.sum(neg_loss)

        return loss + neg_loss

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
        (time_seq, event, score, mask), (mean, var) = minibatch
        time_loss, events_loss, mae, f1, the_number_of_events = model(         
                time_seqence = time_seq, event_seqence = event, mask = mask, mean = mean, var = var
        )

        # loss = time_loss + events_loss
        loss = time_loss
        loss.backward()
    
        time_loss = time_loss.item() / the_number_of_events
        events_loss = events_loss.item() / the_number_of_events
        fact = score.sum().item() / the_number_of_events
        
        return [time_loss, fact, events_loss]
    
    def evaluation_step(model, minibatch, device):
        ''' Epoch operation in evaluation phase '''
    
        model.eval()
        [history_time, history_event, result, score, event, mask], (mean, var) = minibatch
        time_loss, events_loss, mae, f1, the_number_of_events = model(
                history_time = history_time, history_event = history_event, result = result, 
                event = event, mask = mask, mean = mean, var = var, evaluate = True
        )
    
        time_loss = time_loss.item() / the_number_of_events
        events_loss = events_loss.item() / the_number_of_events
        fact = score.sum().item() / the_number_of_events
        
        return [time_loss, fact, events_loss, mae, f1]

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
            [absolute loss, relative loss, events loss, mae value, f1 value]
            '''
            return [input[0], input[0] - input[1], input[2], input[3], input[4]]
        
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
            format_dict['f1_value'] = input[4]
            format_dict['num_format'] = {'absolute_loss': ':6.5f', 'relative_loss': ':6.5f', \
                                         'events_loss': ':6.5f', 'mae': ':2.8f', 'f1_value': ':2.8f'}
            return format_dict
        
        return (train_log_print_format(input) if procedure == 'Training' else test_log_print_format(input))

    format_dict_length = 5
    
    logfile_format = {'step': '', 'absolute loss': ':6.5f', 'relative loss': ':6.5f', 'events loss': ':6.5f', 'mae': ':2.8f', 'f1_value': ':2.8f'}

    def logfile_print_format(input):
        if len(input) == 3:
            format_dict = {}
            format_dict['absolute loss'] = input[0]
            format_dict['relative loss'] = input[1]
            format_dict['events loss'] = input[2]
            format_dict['mae'] = 0
            format_dict['f1_value'] = 0
        else:
            format_dict = {}
            format_dict['absolute loss'] = input[0]
            format_dict['relative loss'] = input[1]
            format_dict['events loss'] = input[2]
            format_dict['mae'] = input[3]
            format_dict['f1_value'] = input[4]
        return format_dict
    
    def choose_metric(evaluation_report, test_report):
        '''
        [relative loss on evaluation dataset, relative loss on test dataset, event loss on test dataset]
        '''
        return [evaluation_report[3], test_report[2], test_report[3]]
    
    metric_number = 3 # metric number is the length of the output of choose_metric