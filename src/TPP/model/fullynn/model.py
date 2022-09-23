from .submodel import FullyNN, InvertedBottleneck
from ..utils import BasicModule
import torch
from sklearn.metrics import f1_score


def check_tensor(x):
    assert (x < 0).cpu().numpy().any() == False

'''
Q1: why without bottleneck, the intensity function for each type of event fails to learn?
A: The reason might still be the activation, because we detect that although the norms of gradients are similar, the variances
are significantly different, which is over 100 times larger when a bottleneck layer is applied.
'''

class FullyNNModel(BasicModule):
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
                 reverse_bottleneck = True,
                 no_bottleneck = False, no_norm = False, no_activate = False,
                 wq_nonneg = False, wk_nonneg = False, wv_nonneg = False,
                 split_comp_graph = True):
        super(FullyNNModel, self).__init__()
        self.device = device
        self.mae_threshold = mae_threshold
        self.num_events = num_events
        self.event_toggle = event_toggle
        self.reverse_bottleneck = reverse_bottleneck if split_comp_graph else False
        self.split_comp_graph = split_comp_graph

        self.model = FullyNN(d_history = d_history, d_intensity = d_intensity, num_events = num_events,
                             dropout = dropout, history_module = history_module, history_module_layers = history_module_layers,
                             mlp_layers = mlp_layers, nonlinear = nonlinear, event_toggle = event_toggle, n_head = n_head,
                             wq_nonneg = wq_nonneg, wk_nonneg = wk_nonneg, wv_nonneg = wv_nonneg, split_comp_graph = split_comp_graph, 
                             device = device)
        if reverse_bottleneck and event_toggle:
            self.inv_neck_1 = InvertedBottleneck(self.num_events, self.num_events * 4, device = device, \
                                                 no_bottleneck = no_bottleneck, no_norm = no_norm, no_activate = no_activate)
            self.inv_neck_2 = InvertedBottleneck(self.num_events, self.num_events * 4, device = device, \
                                                 no_bottleneck = no_bottleneck, no_norm = no_norm, no_activate = no_activate)

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
        if self.event_toggle:
            time_next = time_next.repeat(1, 1, self.num_events)                # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len, 1]
        time_next.requires_grad = True
        integral = self.model(events_history, time_history, time_next, mean = mean, var = var, mask = mask_next)
                                                                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len, 1]
        
        # Intensity values and their sum.
        intensity_for_each_event = torch.autograd.grad(
            outputs = integral,
            inputs = time_next,
            grad_outputs = torch.ones_like(integral),
            create_graph = True,
        )[0]
        check_tensor(intensity_for_each_event)                                 # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len, 1]
        assert intensity_for_each_event.shape == integral.shape
        time_next.requires_grad = False

        '''
        This part is only available when evnet_toggle = True
        TODO: fix the loss calculation error when self.reverse_bottleneck = False and evnet_toggle = True
        '''
        if self.event_toggle:
            if self.reverse_bottleneck:
                intensity_for_each_event = self.inv_neck_1(intensity_for_each_event)
                                                                               # [batch_size, seq_len, num_events]
                intensity_for_each_event = self.inv_neck_2(intensity_for_each_event)
                                                                               # [batch_size, seq_len, num_events]
            else:
                '''
                Or, output the original intensity value directly.
                '''
                intensity_for_each_event = intensity_for_each_event
                                                                               # [batch_size, seq_len, num_events]
            event_probability = torch.nn.functional.softmax(intensity_for_each_event, dim = -1)
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
        
        if not evaluate and self.event_toggle and not self.reverse_bottleneck:
            event_loss = torch.tensor(0., dtype = torch.float32)
    
        time_loss = self.time_loss_f(intensity = intensity_for_each_event, events_next = events_next, \
                                     intensity_integral = integral, mask = mask_next)
        the_number_of_events = mask_next.sum()

        return time_loss, event_loss, mae, f1, the_number_of_events


    def evaluate(self, events_history, time_history, taus, mean, var, mask):
        if self.event_toggle:
            taus = taus.repeat(1, 1, self.num_events)
        integral = self.model(events_history, time_history, taus, mean, var, mask)
                                                                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len, 1]
        integral = integral.sum(dim = -1)

        return integral                                                        # [batch_size, seq_len]

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


        expand_integral, expand_intensity, timestamp = \
                        self.model.integral_intensity(events_history, time_history, \
                                                      time_next, resolution, mean, var, mask_next)

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


        probed_results, additional_plot, timestamp = self.model.model_probe_function(events_history, time_history, \
                                                                    time_next, resolution, mean, var, mask_next)
                                                                               # [batch_size, seq_len * resolution] * n

        return (probed_results, additional_plot), timestamp
    
    def time_loss_f(self, intensity, intensity_integral, mask, events_next):
        '''
        The definition of loss.
    
        Args:
            intensity:          [batch_size, seq_len, num_event] if we need event else [batch_size, seq_len, 1]
            intensity_integral: [batch_size, seq_len, num_event] if we need event else [batch_size, seq_len, 1]
            mask:               [batch_size, seq_len]
            events_next:        [batch_size, seq_len]
        '''

        if self.reverse_bottleneck:
            if self.event_toggle:
                intensity = intensity.sum(dim = -1)
                intensity_integral = intensity_integral.sum(dim = -1)
                log_intensity = torch.log(intensity + 1e-9)
                log_p = -log_intensity + intensity_integral
            else:
                log_intensity = torch.log(intensity + 1e-9)
                log_p = -log_intensity.sum(dim = -1) + intensity_integral.sum(dim = -1)
        else:
            if self.event_toggle:
                intensity_mask = torch.nn.functional.one_hot(events_next.long(), num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_event]
                log_intensity = torch.log(intensity + 1e-9)
                log_intensity = (log_intensity * intensity_mask).sum(dim = -1) # [batch_size, seq_len]
                log_p = -log_intensity + intensity_integral.sum(dim = -1)      # [batch_size, seq_len]
            else:
                log_intensity = torch.log(intensity + 1e-9)
                log_p = -log_intensity.sum(dim = -1) + intensity_integral.sum(dim = -1)
    
        loss = log_p
        loss = torch.clamp(loss, max=15) * mask
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
            minibatch: [batch_size, seq_len]
                       contains [time_seq, event_seq, score, mask]
        '''
    
        model.train()
        [time_seq, event_seq, score, mask], (mean, var) = minibatch
        time_loss, events_loss, mae, f1, the_number_of_events = model(         
                input_time = time_seq, input_events = event_seq, mask = mask, mean = mean,\
                var = var
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
            [absolute loss, relative loss, events loss, mae value]
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
            format_dict['num_format'] = {'absolute_loss': ':6.5f', 'relative_loss': ':6.5f',
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