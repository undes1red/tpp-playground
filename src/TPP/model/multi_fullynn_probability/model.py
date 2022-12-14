from .submodel import FullyNN
from ..utils import BasicModule

from scipy.stats import spearmanr
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, top_k_accuracy_score, accuracy_score
from einops import rearrange, repeat, reduce

def check_tensor(x):
    assert (x < 0).any() == False

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

    def forward(self, input_time, input_events, mask, mean, var, evaluate = False):
        time_history, time_next = self.divide_history_and_next(input_time)     # 2 * [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # 2 * [batch_size, seq_len]
        _, mask_next = self.divide_history_and_next(mask)                      # [batch_size, seq_len]

        mae_mean_of_all_event = 0
        if evaluate:
            mae_mean_of_all_event = \
                        self.mean_absolute_error(events_history = events_history,
                                                 time_history = time_history, time_next = time_next, 
                                                 mask = mask_next, mean = mean, var = var)

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
                create_graph = True
            )[0]
            check_tensor(sub_intensity)                                        # [batch_size, seq_len]
            assert sub_intensity.shape == sub_integral.shape

            integral.append(sub_integral)
            intensity.append(sub_intensity)
        
        time_next.requires_grad = False

        integral = rearrange(integral, 'ne b s -> b s ne')                     # [batch_size, seq_len, num_events]
        intensity = rearrange(intensity, 'ne b s -> b s ne')                   # [batch_size, seq_len, num_events]

        '''
        This part is only available when evnet_toggle = True
        '''
        if self.event_toggle:
            log_intensity = torch.log(intensity + 1e-9)                        # [batch_size, seq_len, num_events]
            events_probability = torch.nn.functional.softmax(log_intensity, dim = -1)
                                                                               # [batch_size, seq_len, num_events]
            events_loss = torch.nn.functional.cross_entropy(rearrange(events_probability, 'b s ne -> b ne s'), \
                                                                      events_next.long(), reduction = 'none')
                                                                               # [batch_size, seq_len]
            events_loss = events_loss * mask_next                              # [batch_size, seq_len]
            events_loss = events_loss.sum()

            events_pred_index = torch.argmax(events_probability, dim = -1)[mask_next == 1].detach().cpu().numpy()
            events_true = events_next[mask_next == 1].detach().cpu().numpy()
            f1 = f1_score(y_true = events_true, y_pred = events_pred_index, average = 'macro')
        else:
            events_loss = torch.tensor(0., dtype = torch.float32)
            f1 = 0

        time_loss = self.time_loss_f(intensity = intensity, events_next = events_next, \
                                     intensity_integral = integral, mask = mask_next, negative_loss = self.negative_loss)
        the_number_of_events = mask_next.sum()

        return time_loss, events_loss, mae_mean_of_all_event, f1, the_number_of_events

    def evaluate(self, events_history, time_history, taus, mean, var, mask):
        integral = []
        # Train k FullyNN models for k different event types.
        for item in self.model:
            sub_integral = item(events_history, time_history, taus, mean = mean, var = var, mask = mask)
                                                                               # [batch_size, seq_len] * num_events
            integral.append(sub_integral)
        
        integral = rearrange(integral, 'ne b s -> b s ne')                     # [batch_size, seq_len, num_events]
        integral = reduce(integral, 'b s ne -> b s', 'sum')                    # [batch_size, seq_len, num_events]
        return integral

    def divide_history_and_next(self, input):
        input_history, input_next = input[:, :-1].clone(), input[:, 1:].clone()
        return input_history, input_next

    def mean_absolute_error(self, events_history, time_history, time_next, mask, mean, var):
        '''
        The input should be the original minibatch
        MAE evaluation part, dwg and fullynn exclusive

        Update: 2022-09-23
        Add event-wise MAE support.
        '''
        def bisect_target(events_history, time_history, taus, mean, var):
            return self.evaluate(events_history, time_history, taus, mean, var, mask) - \
                   torch.log(torch.tensor(self.mae_threshold, device = self.device))
            
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
        gap = (tau_pred - time_next) * mask
        mae_mean_of_all_events = torch.sum(torch.abs(gap)) / mask.sum()

        return mae_mean_of_all_events.item()
    
    def mean_absolute_error_per_event(self, input_time, input_events, mask, mean, var, fast = False):
        '''
        Well...We will do something totally different by performing event-wise MAE.
        First, predict the event types by \int_{t_i}^{+\infty}{\lambda^*_i(t)\exp(-\int_{t_0}^{\tau}{\lambda^*_i(t)dt})d\tau}
        Next, given time predictions. (Expectation? or probability bigger than 0.5?)
        
        Monte-Carlo estimation are required.
        '''

        # might be a good idea to utilise function_prober.
        # Now we need to build the input_data by ourselves.
        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # [batch_size, seq_len]
        _, mask_next = self.divide_history_and_next(mask)                      # [batch_size, seq_len]

        memory_ceiling = 5e7

        _, seq_len = events_next.shape

        # set a relatively large number as the infinity
        if mean == 0 and var == 1:
            max_ = input_time.mean() + 10 * input_time.var()
            # max_ can not be too huge for precesion concerns.
            max_ = min(max_, 1e5)

            time_next_inf = torch.ones_like(time_history, device = self.device) * max_
                                                                               # [batch_size, seq_len]
        else:
            max_ = mean + 10 * var
            # max_ can not be too huge for precesion concerns.
            max_ = min(max_, 1e5)

            time_next_inf = torch.ones_like(time_history, device = self.device) * max_
                                                                               # [batch_size, seq_len]

        resolution = min(max(int(torch.max(time_next_inf).item() // 0.005), 100), 5000)
        if seq_len * resolution * self.num_events > memory_ceiling:
            resolution = int(memory_ceiling // (seq_len * self.num_events))
                
        expand_integral_to_inf = []
        expand_intensity_to_inf = []
        for item in self.model:
            expand_integral_item_to_inf, expand_intensity_item_to_inf, timestamp \
                = item.integral_intensity(events_history, time_history, time_next_inf, resolution, mean, var, mask_next)
            expand_integral_to_inf.append(expand_integral_item_to_inf)
            expand_intensity_to_inf.append(expand_intensity_item_to_inf)
        
        expand_integral_to_inf = rearrange(expand_integral_to_inf, 'ne b (s r) -> b s r ne', r = resolution)
                                                                               # [batch_size, seq_len, resolution, num_events]
        expand_intensity_to_inf = rearrange(expand_intensity_to_inf, 'ne b (s r) -> b s r ne', r = resolution)
                                                                               # [batch_size, seq_len, resolution, num_events]
        timestamp = rearrange(timestamp, 'b (s r) -> b s r 1', r = resolution) # [batch_size, seq_len, resolution, 1]

        # step 1: find the events
        expand_integral_to_inf_sum = reduce(expand_integral_to_inf, 'b s r ne -> b s r ()', 'sum')
                                                                               # [batch_size, seq_len, resolution, 1]
        expand_probability_per_event = expand_intensity_to_inf * torch.exp(-expand_integral_to_inf_sum)
                                                                               # [batch_size, seq_len, resolution, num_events]
        probability_integral = expand_probability_per_event[:, :, :-1, :] * timestamp[:, :, 1:, :]
                                                                               # [batch_size, seq_len, resolution, num_events]
        probability_integral = reduce(probability_integral, 'b s r ne -> b s ne', 'sum')
                                                                               # [batch_size, seq_len, num_events]
        probability_integral_sum = reduce(probability_integral, 'b s ne -> b s', 'sum')
                                                                               # [batch_size, seq_len]
        predict_index = torch.argmax(probability_integral, dim = -1)           # [batch_size, seq_len]

        f1 = []
        top_k_acc = []
        for (ground_truth_per_seq, probability_integral_per_seq) in zip(events_next, probability_integral):
            f1.append(f1_score(y_true = ground_truth_per_seq.detach().cpu(),
                               y_pred = torch.argmax(probability_integral_per_seq, dim = -1).detach().cpu(), average = 'macro'))
        
            top_k_acc_single_event_seq = []
            if not fast:
                if self.num_events > 2:
                    for k in range(1, self.num_events + 1):
                        top_k_acc_single_event_seq.append(
                            top_k_accuracy_score(y_true = ground_truth_per_seq.detach().cpu(),
                                                 y_score = probability_integral_per_seq.detach().cpu(),
                                                 k = k,
                                                 labels = np.arange(self.num_events))
                        )
                else:
                    top_k_acc_single_event_seq.append(
                        accuracy_score(
                            y_true = ground_truth_per_seq.detach().cpu(),
                            y_pred = probability_integral_per_seq.detach().cpu()
                        )
                    )
                    top_k_acc_single_event_seq.append(1.0)
            top_k_acc.append(top_k_acc_single_event_seq)
        
        # F1:        [batch_size]
        # top_k_acc: [batch_size, num_events]
        
        predict_index_one_hot = torch.nn.functional.one_hot(predict_index.long(), num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]
        events_next_one_hot = torch.nn.functional.one_hot(events_next.long(), num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]
        p_x_predicted = reduce(probability_integral * predict_index_one_hot, '... ne -> ...', 'sum')
                                                                               # [batch_size, seq_len]
        p_x_real = reduce(probability_integral * events_next_one_hot, '... ne -> ...', 'sum')
                                                                               # [batch_size, seq_len]
        del expand_probability_per_event, timestamp, expand_intensity_to_inf, expand_integral_to_inf, probability_integral

        # step 2: get the time prediction for that kind of events
        if mean == 0:
            resolution = max(min(int(input_time.mean().item() // 0.005), 1000), 1)
        else:
            resolution = max(min(int(mean // 0.005), 1000), 1)
        
        mae_per_event_pure_predict = self.mean_absolute_error_per_event_worker(events_history, predict_index, time_history, time_next,
                                                                               p_x_predicted, resolution, mask_next, mean, var, max_)
        mae_per_event = self.mean_absolute_error_per_event_worker(events_history, events_next, time_history, time_next, 
                                                                  p_x_real, resolution, mask_next, mean, var, max_)
        
        mae_per_event_pure_predict_avg = torch.sum(mae_per_event_pure_predict, dim = -1) / mask_next.sum(dim = -1)
        mae_per_event_avg = torch.sum(mae_per_event, dim = -1) / mask_next.sum(dim = -1)

        return f1, top_k_acc, probability_integral_sum, mask_next, (mae_per_event_pure_predict_avg, mae_per_event_avg), \
               (mae_per_event_pure_predict, mae_per_event)

    def evaluate_per_event(self, events_history, events_next, time_history, taus, resolution, mean, var, mask):
        integral_all_events = []
        intensity_all_events = []

        # Train k FullyNN models for k different event types.
        for item in self.model:
            sub_integral, sub_intensity, timestamp = item.integral_intensity(events_history, time_history, taus, resolution, mean, var, mask)
                                                                               # [batch_size, seq_len * resolution] * n
            integral_all_events.append(sub_integral)
            intensity_all_events.append(sub_intensity)

        integral_all_events = rearrange(integral_all_events, 'ne b (s r) -> b s r ne', r = resolution)
                                                                               # [batch_size, seq_len, resolution, num_events]
        intensity_all_events = rearrange(intensity_all_events, 'ne b (s r) -> b s r ne', r = resolution)
                                                                               # [batch_size, seq_len, resolution, num_events]
        timestamp = rearrange(timestamp, 'b (s r) -> b s r', r = resolution)

        events_next_mask = torch.nn.functional.one_hot(events_next.long(), num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]
        events_next_mask = rearrange(events_next_mask, '... ne -> ... 1 ne')   # [batch_size, seq_len, 1, num_events]
        intensity_i = reduce(intensity_all_events * events_next_mask, 'b s r ne -> b s r', 'sum')
                                                                               # [batch_size, seq_len, resolution]
        integral_all_events_sum = reduce(integral_all_events, 'b s r ne -> b s r', 'sum')
                                                                               # [batch_size, seq_len, resolution]
        p_dist = intensity_i * torch.exp(-integral_all_events_sum)             # [batch_size, seq_len, resolution]
        probability = reduce(p_dist[:, :, :-1] * timestamp[:, :, 1:], '... r -> ...', 'sum')
                                                                               # [batch_size, seq_len]
        return probability

    def mean_absolute_error_per_event_worker(self, events_history, events_next, 
        time_history, time_next, p_x, resolution, mask, mean, var, max_val):
        '''
        The input should be the original minibatch
        MAE evaluation part, dwg and fullynn exclusive

        '''
        def bisect_target(events_history, time_history, taus, mean, var):
            p_xt = self.evaluate_per_event(events_history, events_next, time_history, taus,
                                           resolution, mean, var, mask)        # [batch_size, seq_len]
            p_t_x = p_xt / p_x                                                 # [batch_size, seq_len]
            p_gap = p_t_x - (1 / self.mae_threshold)                           # [batch_size, seq_len]

            return p_gap

        def median_prediction(events_history, time_history, l, r, mean, var):
            for _ in range(50):
                c = (l + r)/2
                v = bisect_target(events_history, time_history, c, mean, var)
                l = torch.where(v < 0, c, l)
                r = torch.where(v >= 0, c, r)

            return (l + r)/2
        
        l = 0.0001*torch.ones_like(time_history, dtype = torch.float32)        # [batch_size, seq_len]
        r = max_val*torch.ones_like(time_history, dtype = torch.float32)       # [batch_size, seq_len]
        tau_pred = median_prediction(events_history, time_history, l, r, mean, var)
                                                                               # [batch_size, seq_len]
        gap = (tau_pred - time_next) * mask
        gap = torch.abs(gap)

        return gap


    # All methods not required by BasicModule are intensity plotter exclusive.
    def function_prober(self, input_data, resolution, sum = True):
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
        
        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len, 1]
        events_history, _ = self.divide_history_and_next(input_events)         # [batch_size, seq_len]
        _, mask_next = self.divide_history_and_next(mask)                      # [batch_size, seq_len]

        expand_integral = []
        expand_intensity = []
        for item in self.model:
            expand_integral_item, expand_intensity_item, timestamp = item.integral_intensity(events_history, time_history, \
                                                                time_next, resolution, mean, var, mask_next)
            expand_integral.append(expand_integral_item)
            expand_intensity.append(expand_intensity_item)
        
        expand_integral = rearrange(expand_integral, 'ne b sr -> b sr ne')     # [batch_size, seq_len * resolution, num_events]
        expand_intensity = rearrange(expand_intensity, 'ne b sr -> b sr ne')   # [batch_size, seq_len * resolution, num_events]
        if sum:
            expand_integral = reduce(expand_integral, 'b sr ne -> b sr', 'sum')
                                                                               # [batch_size, seq_len * resolution]
            expand_intensity = reduce(expand_intensity, 'b sr ne -> b sr', 'sum')
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

        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history, _ = self.divide_history_and_next(input_events)         # [batch_size, seq_len]
        _, mask_next = self.divide_history_and_next(mask)                      # [batch_size, seq_len]

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

        expand_integral = rearrange(expand_integral, 'ne b sr -> b sr ne').cpu()
                                                                               # [batch_size, seq_len * resolution, num_events]
        expand_sum_integral = reduce(expand_integral, 'b sr ne -> b sr', 'sum')
                                                                               # [batch_size, seq_len * resolution]

        expand_intensity = rearrange(expand_intensity, 'ne b sr -> b sr ne').cpu()
                                                                               # [batch_size, seq_len * resolution, num_events]
        expand_sum_intensity = reduce(expand_intensity, 'b sr ne -> b sr', 'sum')
                                                                               # [batch_size, seq_len * resolution]
        probed_results['integral'] = expand_sum_integral
        probed_results['intensity'] = expand_sum_intensity

        f1, top_k, probability_sum, mask, maes_avg, maes = self.mean_absolute_error_per_event(input_time, input_events, mask, mean, var)
        mae_per_event_pure_predict_avg, mae_per_event_avg = maes_avg
        mae_per_event_pure_predict, mae_per_event = maes

        probability_sum = probability_sum.detach().cpu().numpy()               # [batch_size, seq_len]
        mae_per_event_pure_predict_avg = mae_per_event_pure_predict_avg.detach().cpu().numpy()
                                                                               # [batch_size]
        mae_per_event_avg = mae_per_event_avg.detach().cpu().numpy()           # [batch_size]
        mae_per_event_pure_predict = mae_per_event_pure_predict.detach().cpu().numpy()
                                                                               # [batch_size, seq_len]
        mae_per_event = mae_per_event.detach().cpu().numpy()                   # [batch_size, seq_len]

        packed_values = zip(f1, top_k, probability_sum, mae_per_event_pure_predict, mae_per_event_pure_predict_avg, \
                            mae_per_event, mae_per_event_avg, expand_intensity, time_next, mask_next)

        additional_plot = []

        for idx, (f1_per_seq, top_k_per_seq, probability_sum_per_seq, 
                  mae_per_event_pure_predict_per_seq, mae_per_event_pure_predict_avg_per_seq,
                  mae_per_event_per_seq, mae_per_event_avg_per_seq,
                  expand_intensity_per_seq, time_next_per_seq, mask_per_seq) \
            in enumerate(packed_values):
            '''
            the mean of pe-MAE of each event sequence against predicted events and real events
            '''
            data_mae_avg_per_seq = {
                'x': np.ones(2) * f1_per_seq,
                'y': [mae_per_event_pure_predict_avg_per_seq, mae_per_event_avg_per_seq],
                'marks': ['Predicted labels', 'True labels']
            }

            '''
            Top-K accuracy
            '''
            data_top_k_per_seq = {
                'x': np.arange(1, self.num_events + 1),
                'y': top_k_per_seq,
                'marks': 'Top-K accuracy'
            }

            '''
            Logarithm of pe-MAEs at each event
            '''
            seq_len = mask_per_seq.sum()
            data_maes_per_seq = {
                'x': list(range(seq_len)) * 2,
                'y': np.concatenate(
                    (np.log(1 + mae_per_event_pure_predict_per_seq[:seq_len]),
                    np.log(1 + mae_per_event_per_seq[:seq_len]))
                ),
                'marks': ['MAE_k against prediction'] * seq_len +  ['MAE_k against real events'] * seq_len
            }

            '''
            Check the sum of data probability over event types. The sum should be close to 1.
            '''
            data_probability_sum_per_seq = {
                'x': torch.arange(seq_len),
                'y': probability_sum_per_seq[:seq_len]
            }

            # additional plot, measure the spearman correlation across available events.
            additional_plot_per_seq = {
                'heatmap': [],
                'pointplot': [],
                'lineplot': []
            }

            # Point plot
            additional_plot_per_seq['pointplot'].append([
                'mae_per_event',
                {
                    'x': 'x',
                    'y': 'y',
                    'data': data_mae_avg_per_seq,
                    'hue': 'marks'
                },
                {
                    'horizontalalignment': 'center',
                    'color': 'black',
                    'weight': 'light'
                }
            ])

            # Line plot
            additional_plot_per_seq['lineplot'] = [[
                'top_k_accuracy',
                {
                    'x': 'x',
                    'y': 'y',
                    'hue': 'marks',
                    'data': data_top_k_per_seq,
                    'markers': True
                }
            ],
            [
                'probability_sum',
                {
                    'x': 'x',
                    'y': 'y',
                    'data': data_probability_sum_per_seq,
                    'markers': True
                }
            ],
            [
                'log_mae_k',
                {
                    'x': 'x',
                    'y': 'y',
                    'hue': 'marks',
                    'data': data_maes_per_seq,
                    'markers': True
                }
            ]]

            # Heatmap
            heatmap_data = {}
            # rho: spearman coefficient
            heatmap_data['spearman'] = spearmanr(expand_intensity_per_seq[:seq_len * resolution])[0]
            if self.num_events == 2:
                heatmap_data['spearman'] = np.array([[1, heatmap_data['spearman']], [heatmap_data['spearman'], 1]])

            # r: pearson coefficient
            heatmap_data['pearson'] = np.corrcoef(expand_intensity_per_seq[:seq_len * resolution], rowvar = False)
            # L^1 metric
            heatmap_data['L1'] = L1_distance(expand_intensity_per_seq[:seq_len * resolution],
                                             resolution = resolution, num_events = self.num_events,
                                             time_next = time_next_per_seq[:seq_len])

            # Transfer the result matrices into DataFrames.
            def matrix_to_pd(matrix, index_name, column_name, value_name):
                index, column = matrix.shape

                # The index and column list
                index_list = [ele for ele in range(index) for _ in range(column)]
                column_list = list(range(column)) * index

                df = pd.DataFrame.from_dict({
                    index_name: index_list,
                    column_name: column_list,
                    value_name: matrix.flatten()
                })

                df = df.pivot(index = index_name, columns = column_name, values = value_name)

                return df
            
            heatmap_data['pearson'] \
                = matrix_to_pd(heatmap_data['pearson'], index_name = 'Event type', column_name = 'Event type ', value_name = 'pearson')
            heatmap_data['spearman'] \
                = matrix_to_pd(heatmap_data['spearman'], index_name = 'Event type', column_name = 'Event type ', value_name = 'spearman')
            heatmap_data['L1'] \
                = matrix_to_pd(heatmap_data['L1'], index_name = 'Event type', column_name = 'Event type ', value_name = 'L1')

            # add plots
            for key, value in heatmap_data.items():
                additional_plot_per_seq['heatmap'].append(
                [
                    f'{key}',
                    {
                        'data': value,
                        'cmap': "YlGnBu",
                        'vmin': 0,
                        'vmax': max(1, np.max(value.values)),
                        'annot': True
                    }
                ])

            additional_plot.append(additional_plot_per_seq)

        return (probed_results, additional_plot), timestamp
    
    def time_loss_f(self, intensity, intensity_integral, mask, events_next, negative_loss):
        '''
        The definition of loss.
    
        Args:
            intensity:          [batch_size, seq_len, num_events]
            intensity_integral: [batch_size, seq_len, num_events]
            events_next:        [batch_size, seq_len]
            mask:               [batch_size, seq_len]
        '''
        neg_loss = 0
        if negative_loss:
            intensity_mask = nn.functional.one_hot(events_next.long(), num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]
            # elude the nan loss caused by 0 intensity.
            intensity += 1e-9                                                  # [batch_size, seq_len, num_events]
            sum_of_intensity_and_neg = reduce(intensity, '... ne -> ... ()', 'sum')
                                                                               # [batch_size, seq_len, 1]
            log_posterior = - torch.log(intensity / sum_of_intensity_and_neg)
                                                                               # [batch_size, seq_len, num_events]
            log_posterior = log_posterior * intensity_mask                     # [batch_size, seq_len, num_events]
            neg_loss = reduce(log_posterior, '... ne -> ...', 'sum')           # [batch_size, seq_len]
            neg_loss = (neg_loss * mask).clamp(max = 15)                       # [batch_size, seq_len]
            neg_loss = torch.sum(neg_loss)

        intensity_mask = nn.functional.one_hot(events_next.long(), num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]
        log_intensity = torch.log(intensity + 1e-9)
        log_intensity = log_intensity * intensity_mask                         # [batch_size, seq_len]
        log_intensity = reduce(log_intensity, '... ne -> ...', 'sum')          # [batch_size, seq_len]
        time_loss = -log_intensity + reduce(intensity_integral, '... ne -> ...', 'sum')
                                                                               # [batch_size, seq_len]
        time_loss = (time_loss * mask).clamp(max = 15)                         # [batch_size, seq_len]
        time_loss = torch.sum(time_loss)

        loss = time_loss + neg_loss
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
        return [test_report[3], ]
    
    metric_number = 1 # metric number is the length of the output of choose_metric


def L1_distance(input, resolution, num_events, time_next):
    '''
    This function calculates the L^1 distance between two functions in scattered form.
    Input:
    1. input:      function values
                   [seq_len * resolution, num_events]
    2. resolution: int
                   the number of points from [t_{i - 1}, t_i]
    3. num_event:  int
                   the number of event types
    4. time_next:  [seq_len]
                   the length of all intervals with interpolations.
    '''

    input = rearrange(input, '(s r) ne -> ne s r', r = resolution)             # [num_events, seq_len, resolution]
    intensity_1 = repeat(input, 'ne s r -> ne new_d s r', new_d = num_events)  # [num_events, num_events, seq_len, resolution]
    intensity_2 = repeat(input, 'ne s r -> new_d ne s r', new_d = num_events)  # [num_events, num_events, seq_len, resolution]
    delta_intensity = np.abs(intensity_1 - intensity_2)                        # [num_events, num_events, seq_len, resolution]

    gap = time_next.detach().cpu().numpy() / (resolution - 1)                  # [seq_len]
    gap = rearrange(gap, 's -> 1 1 s 1')                                       # [num_events, num_events, seq_len, 1]

    L1 = reduce((delta_intensity * gap)[:, :, :, :-1], 'ne1 ne2 s r -> ne1 ne2', 'sum')
                                                                               # [num_events, num_events]
    # round off the value smaller than 1e-6
    L1[L1 < 1e-6] = 0

    return L1