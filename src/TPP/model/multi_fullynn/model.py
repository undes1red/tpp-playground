from .submodel import FullyNN
from ..utils import BasicModule
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
                 negative_loss = False):
        super(MultiFullyNNModel, self).__init__()
        self.device = device
        self.mae_threshold = mae_threshold
        self.num_events = num_events
        self.event_toggle = event_toggle
        self.negative_loss = negative_loss
        
        self.model = nn.ModuleList([
            FullyNN(d_history = d_history, d_intensity = d_intensity, num_events = num_events,
                    dropout = dropout, history_module = history_module, history_module_layers = history_module_layers,
                    mlp_layers = mlp_layers, nonlinear = nonlinear, event_toggle = event_toggle, n_head = n_head,
                    wq_nonneg = wq_nonneg, wk_nonneg = wk_nonneg, wv_nonneg = wv_nonneg, device = device)
                    for _ in range(self.num_events)])

    def forward(self, input_time, input_events, mask, mean, var, evaluate = False):
        time_history, time_next = self.divide_history_and_next(input_time, unsqueeze = True)
                                                                               # 2 * [batch_size, seq_len, 1]
        events_history, events_next = self.divide_history_and_next(input_events, unsqueeze = False)
                                                                               # 2 * [batch_size, seq_len]
        _, mask_next = self.divide_history_and_next(mask, unsqueeze = False)   # [batch_size, seq_len]

        mae = 0
        if evaluate:
            mae = self.mean_absolute_error(events_history = events_history, time_history = time_history,\
                                           time_next = time_next, mask = mask_next, mean = mean, var = var)

        # preparing for multi-event training when needed
        time_next.requires_grad = True
        
        integral = []
        intensity = []
        # Train k FullyNN models for k different event types.
        for item in self.model:
            sub_integral = item(events_history, time_history, time_next, mean = mean, var = var, mask = mask_next)
                                                                               # [batch_size, seq_len]
    
            # Intensity values and their sum.
            sub_intensity = torch.autograd.grad(
                outputs = sub_integral,
                inputs = time_next,
                grad_outputs = torch.ones_like(sub_integral),
                create_graph = True,
            )[0]
            sub_intensity = sub_intensity.squeeze(dim = -1)
            check_tensor(sub_intensity)                                        # [batch_size, seq_len, 1]
            assert sub_intensity.shape == sub_integral.shape

            integral.append(sub_integral)
            intensity.append(sub_intensity)
        
        time_next.requires_grad = False

        integral = torch.stack(integral, dim = -1)                             # [batch_size, seq_len, num_events]
        intensity = torch.stack(intensity, dim = -1)                           # [batch_size, seq_len, num_events]

        '''
        This part is only available when evnet_toggle = True
        TODO: fix the loss calculation error when self.reverse_bottleneck = False and evnet_toggle = True
        '''
        if self.event_toggle:
            event_probability = torch.nn.functional.softmax(intensity, dim = -1)
                                                                               # [batch_size, seq_len, num_events]
            event_loss = torch.nn.functional.cross_entropy(event_probability.reshape(-1, self.num_events), \
                                                        events_next.flatten().long(), reduction = 'none')
            event_loss *= mask_next.reshape(-1)
            event_loss = event_loss.sum()

            event_pred_index = torch.argmax(event_probability.reshape(-1, self.num_events), dim = -1)[mask_next.reshape(-1) == 1].detach().cpu().numpy()
            event_true = events_next.long().flatten()[mask_next.reshape(-1) == 1].detach().cpu().numpy()
            f1 = f1_score(y_true = event_true, y_pred = event_pred_index, average = 'macro')
        else:
            event_loss = torch.tensor(0., dtype = torch.float32)
            f1 = 0

    
        time_loss = self.time_loss_f(intensity = intensity, events_next = events_next, \
                                     intensity_integral = integral, mask = mask_next, negative_loss = self.negative_loss)
        the_number_of_events = mask_next.sum()

        return time_loss, event_loss, mae, f1, the_number_of_events


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

    def mean_absolute_error(self, events_history, time_history, time_next, mask, mean, var):
        '''
        The input should be the original minibatch
        MAE evaluation part, dwg and fullynn exclusive
        '''
        def bisect_target(events_history, time_history, taus, mean, var):
            return self.evaluate(events_history, time_history, taus, mean, var, mask).unsqueeze(-1) - \
                   torch.log(torch.tensor(self.mae_threshold, device = time_history.device))
            
        def median_prediction(events_history, time_history, l, r, mean, var):
            for _ in range(50):
                c = (l + r)/2
                v = bisect_target(events_history, time_history, c, mean, var)
                l = torch.where(v < 0, c, l)
                r = torch.where(v >= 0, c, r)

            return (l + r)/2
        
        l = 0.0001*torch.ones_like(time_history, dtype = torch.float32)        # [batch_size, seq_len, 1]
        r = 1e6*torch.ones_like(time_history, dtype = torch.float32)           # [batch_size, seq_len, 1]
        tau_pred = median_prediction(events_history, time_history, l, r, mean, var)
        gap = (tau_pred - time_next).squeeze(-1) * mask
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
        
        expand_integral = torch.stack(expand_integral, dim = -1).sum(dim = -1) # [batch_size, seq_len]
        expand_intensity = torch.stack(expand_intensity, dim = -1).sum(dim = -1)
                                                                               # [batch_size, seq_len]

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

        expand_integral = torch.stack(expand_integral, dim = -1).sum(dim = -1) # [batch_size, seq_len]
        expand_intensity = torch.stack(expand_intensity, dim = -1).sum(dim = -1)
                                                                               # [batch_size, seq_len]
        probed_results['intensity'] = expand_intensity
        probed_results['integral'] = expand_integral

        return (probed_results,), timestamp
    
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
            intensity_mask = nn.functional.one_hot(events_next.long(), num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_event]
            # elude the nan loss caused by 0 intensity.
            intensity += 1e-9                                                  # [batch_size, seq_len, num_event]
            sum_of_intensity_and_neg = intensity.sum(dim = -1, keepdim = True) # [batch_size, seq_len, 1]
            log_posterior = - torch.log(intensity / sum_of_intensity_and_neg)
                                                                               # [batch_size, seq_len, num_event]
            log_posterior *= intensity_mask                                    # [batch_size, seq_len, num_event]
            neg_loss = (log_posterior.sum(dim = -1).clamp(max = 15)) * mask    # [batch_size, seq_len]
            neg_loss = torch.sum(neg_loss)

        intensity_mask = nn.functional.one_hot(events_next.long(), num_classes = self.num_events)
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
        [time_seq, event_seq, score, mask], (mean, var) = minibatch
        time_loss, events_loss, mae, f1, the_number_of_events = model(         
                input_time = time_seq, input_events = event_seq, mask = mask, mean = mean,\
                var = var
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
        [time_seq, event_seq, score, mask], (mean, var) = minibatch
        time_loss, events_loss, mae, f1, the_number_of_events = model(
                input_time = time_seq, input_events = event_seq, mask = mask, evaluate = True,\
                mean = mean, var = var
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