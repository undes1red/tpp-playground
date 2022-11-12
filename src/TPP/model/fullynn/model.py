from .submodel import FullyNN, InvertedBottleneck
from ..utils import BasicModule
import torch
from sklearn.metrics import f1_score, top_k_accuracy_score, accuracy_score
from einops import rearrange, repeat, reduce
import numpy as np


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
                 split_comp_graph = True, zero_shift = False, negative_loss = False):
        super(FullyNNModel, self).__init__()
        self.device = device
        self.mae_threshold = mae_threshold
        self.num_events = num_events
        self.event_toggle = event_toggle
        self.reverse_bottleneck = reverse_bottleneck if split_comp_graph else False
        self.split_comp_graph = split_comp_graph
        self.negative_loss = negative_loss

        self.model = FullyNN(d_history = d_history, d_intensity = d_intensity, num_events = num_events,
                             dropout = dropout, history_module = history_module, history_module_layers = history_module_layers,
                             mlp_layers = mlp_layers, nonlinear = nonlinear, event_toggle = event_toggle, n_head = n_head,
                             wq_nonneg = wq_nonneg, wk_nonneg = wk_nonneg, wv_nonneg = wv_nonneg, split_comp_graph = split_comp_graph, 
                             zero_shift = zero_shift, device = device)
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
                                     intensity_integral = integral, mask = mask_next, negative_loss = self.negative_loss)
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

    def mean_absolute_error_per_event(self, input_time, input_events, mask, mean, var, fast = False):
        '''
        Well...We will do something totally different by performing event-wise MAE.
        First, predict the event types by \int_{t_i}^{+\infty}{\lambda^*_i(t)\exp(-\int_{t_0}^{\tau}{\lambda^*_i(t)dt})d\tau}
        Next, given time predictions. (Expectation? or probability bigger than 0.5?)
        
        Monte-Carlo estiamtion are required.
        '''

        # might be a good idea to utilise function_prober.
        # Now we need to build the input_data by ourselves.
        time_history, time_next = self.divide_history_and_next(input_time, unsqueeze = True)
                                                                               # [batch_size, seq_len, 1]
        events_history, events_next = self.divide_history_and_next(input_events, unsqueeze = False)
                                                                               # [batch_size, seq_len]
        _, mask_next = self.divide_history_and_next(mask, unsqueeze = False)   # [batch_size, seq_len]

        memory_ceiling = 7e6
        _, seq_len = events_next.shape

        # set a relatively large number as the infinity
        if mean == 0 and var == 1:
            max_ = input_time.mean() + 10 * input_time.var()
            time_next_inf = torch.ones_like(time_history, device = self.device) * max_
                                                                               # [batch_size, seq_len, 1]
        else:
            max_ = mean + 10 * var
            time_next_inf = torch.ones_like(time_history, device = self.device) * max_
                                                                               # [batch_size, seq_len, 1]
        resolution = min(max(int(torch.max(time_next_inf).item() // 0.005), 100), 5000)
        # resolution = min(int(torch.max(time_next_inf).item() // 0.01), 100)
        if seq_len * resolution * self.num_events > memory_ceiling:
            resolution = int(memory_ceiling // (seq_len * self.num_events))

        expand_integral_to_inf, expand_intensity_to_inf, timestamp \
                = self.model.integral_intensity(events_history, time_history, time_next_inf, resolution, mean, var, mask_next,
                                                sum = False)
        
        expand_integral_to_inf = rearrange(expand_integral_to_inf, 'b (s r) n -> b s r n', r = resolution)
                                                                               # [batch_size, seq_len, resolution, num_event]
        expand_intensity_to_inf = rearrange(expand_intensity_to_inf, 'b (s r) n -> b s r n', r = resolution)
                                                                               # [batch_size, seq_len, resolution, num_event]
        timestamp = rearrange(timestamp, 'b (s r) -> b s r', r = resolution)   # [batch_size, seq_len, resolution]

        # step 1: find the event
        expand_probability_per_event = expand_intensity_to_inf * torch.exp(-expand_integral_to_inf.sum(dim = -1, keepdim = True))
                                                                               # [batch_size, seq_len, resolution, num_event]
        probability_integral = expand_probability_per_event[:, :, :-1, :] * (timestamp[:, :, 1:].unsqueeze(dim = -1))
                                                                               # [batch_size, seq_len, resolution, num_event]
        probability_integral = probability_integral.sum(dim = -2)              # [batch_size, seq_len, num_event]
        predict_index = torch.argmax(probability_integral, dim = -1)           # [batch_size, seq_len]
        probability_integral_sum = probability_integral.sum(dim = -1)          # [batch_size, seq_len]

        # Only available when batch_size = 1
        f1 = f1_score(y_true = events_next.squeeze().detach().cpu(),
                      y_pred = predict_index.squeeze().detach().cpu(), average = 'macro')

        # Only available when batch_size = 1
        top_k_acc = []
        if not fast:
            if self.num_events > 2:
                for k in range(1, self.num_events + 1):
                    top_k_acc.append(
                        top_k_accuracy_score(y_true = events_next.squeeze().detach().cpu(),
                                             y_score = probability_integral.reshape(-1, self.num_events).detach().cpu(),
                                             k = k,
                                             labels = np.arange(self.num_events))
                    )
            else:
                top_k_acc.append(
                    accuracy_score(
                        y_true = events_next.squeeze().detach().cpu(),
                        y_pred = predict_index.squeeze().detach().cpu()
                    )
                )
                top_k_acc.append(1.0)

        del expand_probability_per_event, timestamp, expand_intensity_to_inf, expand_integral_to_inf

        # step 2: get the time prediction for that kind of event
        if mean == 0:
            resolution = max(min(int(input_time.mean().item() // 0.005), 500), 1)
        else:
            resolution = max(min(int(mean // 0.005), 500), 1)

        mae_per_event_pure_predict = self.mean_absolute_error_per_event_worker(events_history, predict_index, time_history, time_next,
                                                                               probability_integral, resolution, mask_next, mean, var, max_)
        mae_per_event = self.mean_absolute_error_per_event_worker(events_history, events_next, time_history, time_next, 
                                                                  probability_integral, resolution, mask_next, mean, var, max_)
        
        mae_per_event_pure_predict_avg = torch.sum(mae_per_event_pure_predict) / mask_next.sum()
        mae_per_event_avg = torch.sum(mae_per_event) / mask_next.sum()

        return f1, top_k_acc, probability_integral_sum, (mae_per_event_pure_predict_avg.item(), mae_per_event_avg.item()), \
               (mae_per_event_pure_predict, mae_per_event)

    def evaluate_per_event(self, events_history, event_next, time_history, taus, resolution, mean, var, mask):
        # Train k FullyNN models for k different event types.
        integral_all_event, intensity_all_event, timestamp = self.model.integral_intensity(events_history, time_history, \
                                                 taus, resolution, mean, var, mask, sum = False)
                                                                               # [batch_size, seq_len * resolution] * n

        intensity_all_event = rearrange(intensity_all_event, 'b (s r) n -> b s r n', r = resolution)
                                                                               # [batch_size, seq_len, resolution, num_events]
        integral_all_event = rearrange(integral_all_event, 'b (s r) n -> b s r n', r = resolution)
                                                                               # [batch_size, seq_len, resolution, num_events]
        timestamp = rearrange(timestamp, 'b (s r) -> b s r', r = resolution)   # [batch_size, seq_len, resolution, num_events]

        event_next_index = torch.nn.functional.one_hot(event_next.long(), num_classes = self.num_events).unsqueeze(dim = -2)
                                                                               # [batch_size, seq_len, 1, num_events]
        intensity_i = (intensity_all_event * event_next_index).sum(dim = -1)   # [batch_size, seq_len, resolution]

        p_dist = intensity_i * torch.exp(-integral_all_event.sum(dim = -1))    # [batch_size, seq_len, resolution]
        probability = torch.sum(p_dist[:, :, :-1] * timestamp[:, :, 1:], dim = -1)
                                                                               # [batch_size, seq_len]

        return probability

    def mean_absolute_error_per_event_worker(self, events_history, events_next,
        time_history, time_next, probability_integral, resolution, mask, mean, var, max_val):
        '''
        The input should be the original minibatch
        MAE evaluation part, dwg and fullynn exclusive

        '''
        def bisect_target(events_history, time_history, taus, mean, var):
            p_xt = self.evaluate_per_event(events_history, events_next, time_history, taus,
                                           resolution, mean, var, mask)        # [batch_size, seq_len]
            event_next_one_hot = torch.nn.functional.one_hot(events_next.long(), num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_event]
            p_x = torch.sum(probability_integral * event_next_one_hot, dim = -1)
                                                                               # [batch_size, seq_len]
            p_t_x = p_xt / p_x                                                 # [batch_size, seq_len]
            p_gap = p_t_x - (1 / self.mae_threshold)                           # [batch_size, seq_len]

            return p_gap.unsqueeze(dim = -1)
            
        def median_prediction(events_history, time_history, l, r, mean, var):
            for _ in range(50):
                c = (l + r)/2
                v = bisect_target(events_history, time_history, c, mean, var)
                l = torch.where(v < 0, c, l)
                r = torch.where(v >= 0, c, r)

            return (l + r)/2
        
        l = 0.0001*torch.ones_like(time_history, dtype = torch.float32)        # [batch_size, seq_len, 1]
        r = max_val*torch.ones_like(time_history, dtype = torch.float32)       # [batch_size, seq_len, 1]
        tau_pred = median_prediction(events_history, time_history, l, r, mean, var)
        gap = (tau_pred - time_next).squeeze(-1) * mask
        gap = torch.abs(gap)

        return gap

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
        events_history, _ = self.divide_history_and_next(input_events, unsqueeze = False)
                                                                               # [batch_size, seq_len]
        _, mask_next = self.divide_history_and_next(mask, unsqueeze = False)   # [batch_size, seq_len]


        probed_results, additional_plot, timestamp = self.model.model_probe_function(events_history, time_history, \
                                                                    time_next, resolution, mean, var, mask_next)

                                                                               # [batch_size, seq_len * resolution] * n
        f1, top_k, probability_sum, maes_avg, maes = self.mean_absolute_error_per_event(input_time, input_events, mask, mean, var)
        maes_avg = np.array(maes_avg)

        data_mae_avg = {
            'x': np.ones_like(maes_avg) * f1,
            'y': maes_avg,
            'marks': ['Predicted labels', 'True labels']
        }

        data_top_k = {
            'x': np.arange(1, self.num_events + 1),
            'y': top_k,
            'marks': 'Top-K accuracy'
        }

        data_maes = {
            'x': list(range(len(maes[0][0]))) * 2,
            'y': np.concatenate(
                (torch.log(1 + maes[0]).detach().cpu().numpy().squeeze(), torch.log(1 + maes[1]).detach().cpu().numpy().squeeze())
            ),
            'marks': ['MAE_k against prediction'] *len(maes[0][0]) +  ['MAE_k against real events'] * len(maes[0][0])
        }

        data_probability_sum = {
            'x': torch.arange(probability_sum.shape[-1]),
            'y': probability_sum.detach().squeeze().cpu().numpy()
        }

        # Point plot
        additional_plot['pointplot'] = [[
            'mae_per_event',
            {
                'x': 'x',
                'y': 'y',
                'data': data_mae_avg,
                'hue': 'marks'
            },
            {
                'horizontalalignment': 'center',
                'color': 'black',
                'weight': 'light'
            }
        ],
        ]

        # Line plot
        additional_plot['lineplot'] = [[
            'top_k_accuracy',
            {
                'x': 'x',
                'y': 'y',
                'hue': 'marks',
                'data': data_top_k,
                'markers': True
            }
        ],
        [
            'log_mae_k',
            {
                'x': 'x',
                'y': 'y',
                'hue': 'marks',
                'data': data_maes,
                'markers': True
            }
        ],
        [
            'probability_sum',
            {
                'x': 'x',
                'y': 'y',
                'data': data_probability_sum,
                'markers': True
            }
        ]]

        return (probed_results, additional_plot), timestamp
    
    def time_loss_f(self, intensity, intensity_integral, mask, events_next, negative_loss):
        '''
        The definition of loss.
    
        Args:
            intensity:          [batch_size, seq_len, num_event] if we need event else [batch_size, seq_len, 1]
            intensity_integral: [batch_size, seq_len, num_event] if we need event else [batch_size, seq_len, 1]
            mask:               [batch_size, seq_len]
            events_next:        [batch_size, seq_len]
        '''
        neg_loss = 0

        if self.reverse_bottleneck:
            if self.event_toggle:
                if negative_loss:
                    intensity_mask = torch.nn.functional.one_hot(events_next.long(), num_classes = self.num_events)
                                                                                       # [batch_size, seq_len, num_event]
                    # elude the nan loss caused by 0 intensity.
                    intensity += 1e-9                                                  # [batch_size, seq_len, num_event]
                    sum_of_intensity_and_neg = intensity.sum(dim = -1, keepdim = True) # [batch_size, seq_len, 1]
                    log_posterior = - torch.log(intensity / sum_of_intensity_and_neg)
                                                                                       # [batch_size, seq_len, num_event]
                    log_posterior *= intensity_mask                                    # [batch_size, seq_len, num_event]
                    neg_loss = (log_posterior.sum(dim = -1).clamp(max = 15)) * mask    # [batch_size, seq_len]
                    neg_loss = torch.sum(neg_loss)

                intensity = intensity.sum(dim = -1)
                intensity_integral = intensity_integral.sum(dim = -1)
                log_intensity = torch.log(intensity + 1e-9)
                log_p = -log_intensity + intensity_integral
            else:
                log_intensity = torch.log(intensity + 1e-9)
                log_p = -log_intensity.sum(dim = -1) + intensity_integral.sum(dim = -1)
        else:
            if self.event_toggle:
                if negative_loss:
                    intensity_mask = torch.nn.functional.one_hot(events_next.long(), num_classes = self.num_events)
                                                                                       # [batch_size, seq_len, num_event]
                    # elude the nan loss caused by 0 intensity.
                    intensity += 1e-9                                                  # [batch_size, seq_len, num_event]
                    sum_of_intensity_and_neg = intensity.sum(dim = -1, keepdim = True) # [batch_size, seq_len, 1]
                    log_posterior = - torch.log(intensity / sum_of_intensity_and_neg)
                                                                                       # [batch_size, seq_len, num_event]
                    log_posterior *= intensity_mask                                    # [batch_size, seq_len, num_event]
                    neg_loss = (log_posterior.sum(dim = -1).clamp(max = 15)) * mask    # [batch_size, seq_len]
                    neg_loss = torch.sum(neg_loss)
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
        return [test_report[3],]
    
    metric_number = 1 # metric number is the length of the output of choose_metric