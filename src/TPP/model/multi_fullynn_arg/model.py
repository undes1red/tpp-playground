from .submodel import FullyNN
from ..utils import BasicModule

from scipy.stats import spearmanr
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score


def check_tensor(x):
    assert (x < 0).cpu().numpy().any() == False

'''
Q1: why without bottleneck, the intensity function for each type of event fails to learn?
A: The reason might still be the activation, because we detect that although the norms of gradients are similar, the variances
are significantly different, which is over 100 times larger when a bottleneck layer is applied.
'''

class MultiFullyNNModel(BasicModule):
    def __init__(self, d_history,
                 d_intensity,
                 dropout,
                 history_module_layers,
                 mlp_layers,
                 nonlinear,
                 mae_threshold,
                 num_events,
                 device,
                 history_module = 'LSTM',
                 n_head = 0,
                 event_toggle = False,
                 wq_nonneg = False, wk_nonneg = False, wv_nonneg = False,
                 negative_loss = False, zero_shift = True,
                 zero_detach = True):
        super(MultiFullyNNModel, self).__init__()
        self.device = device
        self.mae_threshold = mae_threshold
        self.num_events = num_events
        self.event_toggle = event_toggle
        self.negative_loss = negative_loss
        self.zero_shift = zero_shift
        self.zero_detach = zero_detach

        if self.event_toggle is False:
            self.num_events = 1
        
        self.model = nn.ModuleList([
            FullyNN(d_history = d_history, d_intensity = d_intensity, num_events = num_events,
                    dropout = dropout, history_module = history_module, history_module_layers = history_module_layers,
                    mlp_layers = mlp_layers, nonlinear = nonlinear, event_toggle = event_toggle, n_head = n_head,
                    wq_nonneg = wq_nonneg, wk_nonneg = wk_nonneg, wv_nonneg = wv_nonneg, zero_shift = zero_shift, 
                    zero_detach = zero_detach, device = device)
                    for _ in range(self.num_events)])

    def forward(self, history_time, history_event, result, event, mask, mean, var, evaluate = False):
        '''
        Fit the new synthetic_autoregressive dataloader.
        1. history_time: history timestamp sequences
                         [batch_size, seq_len, history_length]
        2. history_event:history event sequences
                         [batch_size, seq_len, history_length]
        3. result:       when the next event occurs
                         [batch_size, seq_len]
        4. event:        the next event mark
                         [batch_size, seq_len]
        5. mask:         mask for history sequences
                         [batch_size, seq_len, history_length]
        '''

        mask_event = (mask.sum(dim = -1) > 0).int()
        number_of_events = mask_event.sum()
        # mae_mean_of_all_event, mae_mean_of_ded_event = 0, 0
        mae_mean_of_all_event = 0
        if evaluate:
            mae_mean_of_all_event = \
                        self.mean_absolute_error(events_history = history_event, events_next = event, 
                                                 time_history = history_time, time_next = result, 
                                                 mask = mask, mask_event = mask_event, mean = mean, var = var)

        # preparing for multi-event training when needed
        result.requires_grad = True
        
        integral = []
        intensity = []
        # Train k FullyNN models for k different event types.
        for item in self.model:
            sub_integral = item(history_event, history_time, result, mean = mean, var = var, mask = mask)
                                                                               # [batch_size, seq_len]
    
            # Intensity values and their sum.
            sub_intensity = torch.autograd.grad(
                outputs = sub_integral,
                inputs = result,
                grad_outputs = torch.ones_like(sub_integral),
                create_graph = True,
            )[0]
            sub_intensity = sub_intensity.squeeze(dim = -1)
            check_tensor(sub_intensity)                                        # [batch_size, seq_len]
            assert sub_intensity.shape == sub_integral.shape

            integral.append(sub_integral)
            intensity.append(sub_intensity)
        
        result.requires_grad = False

        integral = torch.stack(integral, dim = -1)                             # [batch_size, seq_len, num_events]
        intensity = torch.stack(intensity, dim = -1)                           # [batch_size, seq_len, num_events]

        '''
        This part is only available when evnet_toggle = True
        TODO: fix the loss calculation error when self.reverse_bottleneck = False and evnet_toggle = True
        '''
        if self.event_toggle:
            log_intensity = torch.log(intensity + 1e-9)                        # [batch_size, seq_len, num_events]
            event_probability = torch.nn.functional.softmax(log_intensity, dim = -1)
                                                                               # [batch_size, seq_len, num_events]
            event_loss = torch.nn.functional.cross_entropy(event_probability.reshape(-1, self.num_events), \
                                                        event.flatten().long(), reduction = 'none', ignore_index= self.num_events + 1)
            event_loss *= mask_event.reshape(-1)
            event_loss = event_loss.sum()

            event_pred_index = torch.argmax(event_probability.reshape(-1, self.num_events), dim = -1)[mask_event.reshape(-1) == 1].detach().cpu().numpy()
            event_true = event.long().flatten()[mask_event.reshape(-1) == 1].detach().cpu().numpy()
            f1 = f1_score(y_true = event_true, y_pred = event_pred_index, average = 'macro')
        else:
            event_loss = torch.tensor(0., dtype = torch.float32)
            f1 = 0

        time_loss = self.time_loss_f(intensity = intensity, events_next = event, \
                                     intensity_integral = integral, mask = mask_event, negative_loss = self.negative_loss)

        return time_loss, event_loss, mae_mean_of_all_event, f1, number_of_events

    def evaluate(self, events_history, time_history, taus, mean, var, mask):
        integral = []
        # Train k FullyNN models for k different event types.
        for item in self.model:
            sub_integral = item(events_history, time_history, taus, mean = mean, var = var, mask = mask)
                                                                               # [batch_size, seq_len] * num_events

            integral.append(sub_integral)
        
        integral = torch.stack(integral, dim = -1)                             # [batch_size, seq_len, num_events]
        integral = integral.sum(dim = -1)                                      # [batch_size, seq_len]
        return integral

    def divide_history_and_next(self, input, unsqueeze = False):
        input_history, input_next = input.clone()[:, :-1], input.clone()[:, 1:]
        if unsqueeze:
            input_history = input_history.unsqueeze(-1)                        # [batch_size, seq_len, 1]
            input_next = input_next.unsqueeze(-1)                              # [batch_size, seq_len, 1]
        return input_history, input_next

    def mean_absolute_error(self, events_history, events_next, time_history, time_next, mask, mask_event, mean, var):
        '''
        The input should be the original minibatch
        MAE evaluation part, dwg and fullynn exclusive

        Update: 2022-09-23
        Add event-wise MAE support.
        '''
        def bisect_target(events_history, time_history, taus, mean, var):
            return self.evaluate(events_history, time_history, taus, mean, var, mask) - \
                   torch.log(torch.tensor(self.mae_threshold, device = time_history.device))
            
        def median_prediction(events_history, time_history, l, r, mean, var):
            for _ in range(50):
                c = (l + r)/2
                v = bisect_target(events_history, time_history, c, mean, var)
                l = torch.where(v < 0, c, l)
                r = torch.where(v >= 0, c, r)

            return (l + r)/2
        
        l = 0.0001*torch.ones_like(time_next, dtype = torch.float32)           # [batch_size, seq_len]
        r = 1e6*torch.ones_like(time_next, dtype = torch.float32)              # [batch_size, seq_len]
        tau_pred = median_prediction(events_history, time_history, l, r, mean, var)
        gap = (tau_pred - time_next) * mask_event
        mae_mean_of_all_event = torch.sum(torch.abs(gap)) / mask_event.sum()

        return mae_mean_of_all_event.item()

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
        input_time, input_events, _, mask = input_data[0][:4]
        mean, var = input_data[1]
        
        time_history, time_next = self.divide_history_and_next(input_time, unsqueeze = True)
                                                                               # [batch_size, seq_len, 1]
        events_history, events_next = self.divide_history_and_next(input_events, unsqueeze = False)
                                                                               # [batch_size, seq_len]
        _, mask_next = self.divide_history_and_next(mask, unsqueeze = False)   # [batch_size, seq_len]

        expand_integral = []
        expand_intensity = []
        for item in self.model:
            expand_integral_item, expand_intensity_item, timestamp = item.integral_intensity(events_history, time_history, \
                                                                time_next, resolution, mean, var, mask_next)
            expand_integral.append(expand_integral_item)
            expand_intensity.append(expand_intensity_item)
        
        expand_integral = torch.stack(expand_integral, dim = -1).sum(dim = -1) # [batch_size, seq_len * resolution]
        expand_intensity = torch.stack(expand_intensity, dim = -1).sum(dim = -1)
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
        input_time, input_events, _, mask,  = input_data[0][:4]
        mean, var = input_data[1]

        time_history, time_next = self.divide_history_and_next(input_time, unsqueeze = True)
                                                                               # [batch_size, seq_len, 1]
        events_history, events_next = self.divide_history_and_next(input_events, unsqueeze = False)
                                                                               # [batch_size, seq_len]
        _, mask_next = self.divide_history_and_next(mask, unsqueeze = False)   # [batch_size, seq_len]

        probed_results = {}
        expand_integral = []
        expand_intensity = []
        for idx, item in enumerate(self.model):
            expand_integral_item, expand_intensity_item, timestamp = item.integral_intensity(events_history, time_history, \
                                                                time_next, resolution, mean, var, mask_next)
            probed_results[f'event_intensity_{idx}'] = expand_intensity_item
            probed_results[f'event_integral_{idx}'] = expand_integral_item
            expand_integral.append(expand_integral_item)
            expand_intensity.append(expand_intensity_item)

        expand_sum_integral = torch.stack(expand_integral, dim = -1).sum(dim = -1)
                                                                               # [batch_size, seq_len * resolution]
        expand_sum_intensity = torch.stack(expand_intensity, dim = -1).sum(dim = -1)
                                                                               # [batch_size, seq_len * resolution]
        probed_results['intensity'] = expand_sum_integral
        probed_results['integral'] = expand_sum_intensity

        # additional plot, measure the spearman correlation across available events.
        additional_plot = {
            'heatmap': []
        }

        expand_intensity = torch.stack(expand_intensity, dim = -1).detach().cpu().numpy()
                                                                               # [batch_size, seq_len * resolution, num_event]

        for idx, item in enumerate(expand_intensity):
            heatmap_data = {}
            # rho: spearman coefficient
            heatmap_data['spearman'] = spearmanr(item)[0]
            # r: pearson coefficient
            heatmap_data['pearson'] = np.corrcoef(item, rowvar = False)
            # L^1 metric
            heatmap_data['L1'] = L1_distance(item, resolution = resolution, num_events = self.num_events,
                                             time_next = time_next[idx].detach().cpu().numpy())

            # add plots
            for key, value in heatmap_data.items():
                idx = 0
                additional_plot['heatmap'].append(
                [
                    f'{key}_{idx}',
                    {
                        'data': value,
                        'cmap': "YlGnBu",
                        'vmin': 0
                    }
                ])
                idx += 1

        return (probed_results, additional_plot), timestamp
    
    def time_loss_f(self, intensity, intensity_integral, mask, events_next, negative_loss):
        '''
        The definition of loss.
    
        Args:
            intensity:          [batch_size, seq_len, num_event]
            intensity_integral: [batch_size, seq_len, num_event]
            events_next:        [batch_size, seq_len]
            mask:               [batch_size, seq_len]
        '''
        neg_loss = 0
        if negative_loss:
            intensity_mask = nn.functional.one_hot(events_next.long(), num_classes = self.num_events + 2)[:, :, :self.num_events]
                                                                               # [batch_size, seq_len, num_event]
            # elude the nan loss caused by 0 intensity.
            intensity += 1e-9                                                  # [batch_size, seq_len, num_event]
            sum_of_intensity_and_neg = intensity.sum(dim = -1, keepdim = True) # [batch_size, seq_len, 1]
            log_posterior = - torch.log(intensity / sum_of_intensity_and_neg)
                                                                               # [batch_size, seq_len, num_event]
            log_posterior *= intensity_mask                                    # [batch_size, seq_len, num_event]
            neg_loss = (log_posterior.sum(dim = -1).clamp(max = 15)) * mask    # [batch_size, seq_len]
            neg_loss = torch.sum(neg_loss)

        intensity_mask = nn.functional.one_hot(events_next.long(), num_classes = self.num_events + 2)[:, :, :self.num_events]
                                                                               # [batch_size, seq_len, num_event]

        log_intensity = torch.log(intensity + 1e-9)
        log_intensity = (log_intensity * intensity_mask).sum(dim = -1)         # [batch_size, seq_len]
        loss = -log_intensity + intensity_integral.sum(dim = -1)               # [batch_size, seq_len]
    
        loss = torch.clamp(loss, max = 15) * mask
        loss = torch.sum(loss) + neg_loss

        return loss

    '''
    All static methods
    '''
    def train_step(model, minibatch, device):
        ''' 
        Epoch operation in training phase.
        The input minibatch comprise time sequences.

        Args:
            minibatch: [batch_size, seq_len]
                       contains [time_seq, event_seq, score, mask]
        '''
    
        model.train()
        [history_time, history_event, result, score, event, mask], (mean, var) = minibatch
        time_loss, events_loss, mae, f1, the_number_of_events = model(         
                history_time = history_time, history_event = history_event, result = result, event = event, \
                mask = mask, mean = mean, var = var
        )


        loss = time_loss
        # loss = time_loss + events_loss
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
                history_time = history_time, history_event = history_event, result = result, event = event, \
                mask = mask, mean = mean, var = var, evaluate = True
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
            [absolute loss, relative loss, events loss, mae value, f1_value]
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
            format_dict['num_format'] = {'absolute_loss': ':6.5f', 'relative_loss': ':6.5f',\
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


def L1_distance(input, resolution, num_events, time_next):
    '''
    This function calculates the L^1 distance between two functions.
    Input:
    1. input:      function values
                   [seq_len * resolution, num_events]
    2. resolution: int
                   the number of points from [t_{i - 1}, t_i]
    3. num_event:  int
                   the number of event types
    4. time_next:  [seq_len, num_events]
                   the length of all intervals with interpolations.
    '''

    input = input.reshape(-1, resolution, num_events)                          # [seq_len, resolution, num_events]
    input = np.transpose(input, (2, 0, 1))                                     # [num_events, seq_len, resolution]
    intensity_1 = np.expand_dims(input, axis = 1).repeat(num_events, axis = 1) # [num_events, num_events, seq_len, resolution]
    intensity_2 = np.expand_dims(input, axis = 0).repeat(num_events, axis = 0) # [num_events, num_events, seq_len, resolution]
    delta_intensity = np.abs(intensity_1 - intensity_2)                        # [num_events, num_events, seq_len, resolution]

    gap = np.expand_dims(time_next, axis = 1)                                  # [num_events, 1, seq_len]
    gap = gap / (resolution - 1)
    gap = np.transpose(gap, (2, 0, 1))                                         # [num_events, seq_len, 1]
    gap = np.expand_dims(gap, axis = 1)                                        # [num_events, 1, seq_len, 1]

    L1 = (delta_intensity * gap)[:, :, :, :-1].sum(axis = -1).sum(axis = -1)   # [num_events, num_events]

    # round up the value smaller than 1e-6
    L1[L1 < 1e-6] = 0

    return L1