from .submodel import FullyNN, InvertedBottleneck
from ..utils import BasicModule
import torch
from sklearn.metrics import f1_score, top_k_accuracy_score, accuracy_score
from einops import rearrange, repeat, reduce
import numpy as np


def check_tensor(x):
    assert (x < 0).any() == False

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
        time_history, time_next = self.divide_history_and_next(input_time)     # 2 * [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # 2 * [batch_size, seq_len]
        _, mask_next = self.divide_history_and_next(mask)                      # [batch_size, seq_len]

        mae = 0
        if evaluate:
            mae = self.mean_absolute_error(events_history = events_history, time_history = time_history,\
                                           time_next = time_next, mask = mask_next, mean = mean, var = var)

        # preparing for multi-event training when needed
        if self.event_toggle:
            time_next = repeat(time_next, 'b s -> b s ne', ne = self.num_events)
                                                                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
        time_next.requires_grad = True
        integral = self.model(events_history, time_history, time_next, mean = mean, var = var, mask = mask_next)
                                                                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
        
        # Intensity values and their sum.
        intensity_for_each_event = torch.autograd.grad(
            outputs = integral,
            inputs = time_next,
            grad_outputs = torch.ones_like(integral),
            create_graph = True,
        )[0]
        check_tensor(intensity_for_each_event)                                 # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
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
                probability_for_each_event = intensity_for_each_event          # [batch_size, seq_len, num_events]
            else:
                '''
                Or, output the original intensity value directly.
                '''
                probability_for_each_event = torch.log(intensity_for_each_event + 1e-6)
                                                                               # [batch_size, seq_len, num_events]
            events_probability = torch.nn.functional.softmax(probability_for_each_event, dim = -1)
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
    
        time_loss = self.time_loss_f(intensity = intensity_for_each_event, events_next = events_next, \
                                     intensity_integral = integral, mask = mask_next, negative_loss = self.negative_loss)
        the_number_of_events = mask_next.sum()

        return time_loss, events_loss, mae, f1, the_number_of_events


    def evaluate(self, events_history, time_history, taus, mean, var, mask):
        if self.event_toggle:
            taus = repeat(taus, 'b s -> b s ne', ne = self.num_events)         # [batch_size, seq_len, num_events]
        integral = self.model(events_history, time_history, taus, mean, var, mask)
                                                                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
        if self.event_toggle:
            integral = integral.sum(dim = -1)                                  # [batch_size, seq_len]
        
        return integral

    def divide_history_and_next(self, input):
        input_history, input_next = input[:, :-1].clone(), input[:, 1:].clone()
        return input_history, input_next

    def mean_absolute_error_static(self, events_history, time_history, time_next, mask_history, mask_next, mean, var):
        return self.mean_absolute_error(events_history, time_history, time_next, mask_next, mean, var)

    def mean_absolute_error(self, events_history, time_history, time_next, mask, mean, var, sum = True):
        '''
        The input should be the original minibatch
        MAE evaluation part, dwg and fullynn exclusive
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
        gap = (tau_pred - time_next) * mask                                    # [batch_size, seq_len]

        if sum:
            gap_mean = torch.sum(torch.abs(gap)) / mask.sum()
            return gap_mean.item()
        else:
            return torch.abs(gap)


    def mean_absolute_error_per_event(self, input_time, input_events, mask, mean, var, fast = False):
        '''
        Well...We will do something totally different by performing event-wise MAE.
        First, predict the event types by \int_{t_i}^{+\infty}{\lambda^*_i(t)\exp(-\int_{t_0}^{\tau}{\lambda^*_i(t)dt})d\tau}
        Next, given time predictions. (Expectation? or probability bigger than 0.5?)
        
        Monte-Carlo estiamtion are required.
        '''

        # might be a good idea to utilise function_prober.
        # Now we need to build the input_data by ourselves.
        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # [batch_size, seq_len]
        _, mask_next = self.divide_history_and_next(mask)                      # [batch_size, seq_len]

        memory_ceiling = 7e6
        _, seq_len = events_next.shape

        # set a relatively large number as the infinity
        if mean == 0 and var == 1:
            max_ = input_time.mean() + 10 * input_time.var()
            time_next_inf = torch.ones_like(time_history, device = self.device) * max_
                                                                               # [batch_size, seq_len]
        else:
            max_ = mean + 10 * var
            time_next_inf = torch.ones_like(time_history, device = self.device) * max_
                                                                               # [batch_size, seq_len]
        resolution = min(max(int(torch.max(time_next_inf).item() // 0.005), 100), 5000)

        if seq_len * resolution * self.num_events > memory_ceiling:
            resolution = int(memory_ceiling // (seq_len * self.num_events))

        expand_integral_to_inf, expand_intensity_to_inf, timestamp \
                = self.model.integral_intensity(events_history, time_history, time_next_inf, resolution, mean, var, mask_next,
                                                sum = False)
        
        expand_integral_to_inf = rearrange(expand_integral_to_inf, 'b (s r) n -> b s r n', r = resolution)
                                                                               # [batch_size, seq_len, resolution, num_events]
        expand_intensity_to_inf = rearrange(expand_intensity_to_inf, 'b (s r) n -> b s r n', r = resolution)
                                                                               # [batch_size, seq_len, resolution, num_events]
        timestamp = rearrange(timestamp, 'b (s r) -> b s r', r = resolution)   # [batch_size, seq_len, resolution]

        # step 1: find the events
        expand_probability_per_event = expand_intensity_to_inf * torch.exp(-expand_integral_to_inf.sum(dim = -1, keepdim = True))
                                                                               # [batch_size, seq_len, resolution, num_events]
        timestamp = rearrange(timestamp[:, :, 1:], '... -> ... 1')
        probability_integral = expand_probability_per_event[:, :, :-1, :] * timestamp
                                                                               # [batch_size, seq_len, resolution, num_events]
        probability_integral = reduce(probability_integral, 'b s r ne -> b s ne', 'sum')
                                                                               # [batch_size, seq_len, num_events]
        probability_integral_sum = reduce(probability_integral, 'b s ne -> b s', 'sum')
                                                                               # [batch_size, seq_len]
        predict_index = torch.argmax(probability_integral, dim = -1)           # [batch_size, seq_len]

        
        f1 = []
        top_k_acc = []
        for (events_next_per_seq, probability_integral_per_seq) in zip(events_next, probability_integral):
            f1.append(f1_score(y_true = events_next_per_seq.detach().cpu(),
                          y_pred = torch.argmax(probability_integral_per_seq, dim = -1).detach().cpu(), average = 'macro'))
            top_k_acc_single_event_seq = []
            if not fast:
                if self.num_events > 2:
                    for k in range(1, self.num_events + 1):
                        top_k_acc_single_event_seq.append(
                            top_k_accuracy_score(y_true = events_next_per_seq.detach().cpu(),
                                                 y_score = probability_integral_per_seq.detach().cpu(),
                                                 k = k,
                                                 labels = np.arange(self.num_events))
                        )
                else:
                    top_k_acc_single_event_seq.append(
                        accuracy_score(
                            y_true = events_next_per_seq.detach().cpu(),
                            y_pred = probability_integral_per_seq.detach().cpu()
                        )
                    )
                    top_k_acc_single_event_seq.append(1.0)
                top_k_acc.append(top_k_acc_single_event_seq)

        predict_index_one_hot = torch.nn.functional.one_hot(predict_index.long(), num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]
        events_next_one_hot = torch.nn.functional.one_hot(events_next.long(), num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]
        p_x_predicted = reduce(probability_integral * predict_index_one_hot, '... ne -> ...', 'sum')
                                                                               # [batch_size, seq_len]
        p_x_real = reduce(probability_integral * events_next_one_hot, '... ne -> ...', 'sum')
                                                                               # [batch_size, seq_len]
        del expand_probability_per_event, timestamp, expand_intensity_to_inf, expand_integral_to_inf

        # step 2: get the time prediction for that kind of event
        if mean == 0:
            resolution = max(min(int(input_time.mean().item() // 0.005), 500), 1)
        else:
            resolution = max(min(int(mean // 0.005), 500), 1)

        tau_pred_all_event = self.prediction_with_all_event_types(events_history, predict_index, time_history, time_next,
                                                                  probability_integral, resolution, mask_next, mean, var, max_)
                                                                               # [batch_size, seq_len, num_events]

        mae_per_event_pure_predict = self.mean_absolute_error_per_event_worker(events_history, predict_index, time_history, time_next,
                                                                               p_x_predicted, resolution, mask_next, mean, var, max_)
        mae_per_event = self.mean_absolute_error_per_event_worker(events_history, events_next, time_history, time_next, 
                                                                  p_x_real, resolution, mask_next, mean, var, max_)
        
        mae_per_event_pure_predict_avg = torch.sum(mae_per_event_pure_predict, dim = -1) / mask_next.sum(dim = -1)
        mae_per_event_avg = torch.sum(mae_per_event, dim = -1) / mask_next.sum(dim = -1)

        return f1, top_k_acc, probability_integral_sum, tau_pred_all_event, (mae_per_event_pure_predict_avg, mae_per_event_avg), \
               (mae_per_event_pure_predict, mae_per_event)

    def evaluate_all_event(self, events_history, events_next, time_history, taus, resolution, mean, var, mask):
        # Train k FullyNN models for k different event types.
        integral_all_events, intensity_all_events, timestamp = self.model.integral_intensity(events_history, time_history, \
                                                 taus, resolution, mean, var, mask, sum = False, event_time_probe = True)
                                                                               # [batch_size, seq_len * resolution] * n

        intensity_all_events = rearrange(intensity_all_events, 'b (s r) n -> b s r n', r = resolution)
                                                                               # [batch_size, seq_len, resolution, num_events]
        integral_all_events = rearrange(integral_all_events, 'b (s r) n -> b s r n', r = resolution)
                                                                               # [batch_size, seq_len, resolution, num_events]

        integral_sum = reduce(integral_all_events, 'b s r ne -> b s r ()', 'sum')
                                                                               # [batch_size, seq_len, resolution, 1]
        p_dist = intensity_all_events * torch.exp(-integral_sum)               # [batch_size, seq_len, resolution, num_events]
        probability = reduce(p_dist[:, :, :-1, :] * timestamp[:, :, 1:, :], 'b s r ne -> b s ne', 'sum')
                                                                               # [batch_size, seq_len, num_events]
        return probability

    def prediction_with_all_event_types(self, events_history, events_next,
        time_history, time_next, p_x, resolution, mask, mean, var, max_val):
        '''
        The input should be the original minibatch
        MAE evaluation part, dwg and fullynn exclusive

        '''
        def bisect_target(events_history, time_history, taus, mean, var):
            p_xt = self.evaluate_all_event(events_history, events_next, time_history, taus,
                                           resolution, mean, var, mask)        # [batch_size, seq_len, num_events]
            p_t_x = p_xt / p_x                                                 # [batch_size, seq_len, num_events]
            p_gap = p_t_x - 1 / self.mae_threshold                             # [batch_size, seq_len, num_events]

            return p_gap
            
        def median_prediction(events_history, time_history, l, r, mean, var):
            for _ in range(50):
                c = (l + r)/2
                v = bisect_target(events_history, time_history, c, mean, var)
                l = torch.where(v < 0, c, l)
                r = torch.where(v >= 0, c, r)

            return (l + r)/2
        
        l = 0.0001*torch.ones((*time_history.shape, self.num_events), dtype = torch.float32, device = self.device)
                                                                               # [batch_size, seq_len, num_events]
        r = 1e6*torch.ones((*time_history.shape, self.num_events), dtype = torch.float32, device = self.device)
                                                                               # [batch_size, seq_len, num_events]
        tau_pred = median_prediction(events_history, time_history, l, r, mean, var)
                                                                               # [batch_size, seq_len, num_events]

        return tau_pred

    def evaluate_per_event(self, events_history, events_next, time_history, taus, resolution, mean, var, mask):
        # Train k FullyNN models for k different event types.
        integral_all_events, intensity_all_events, timestamp = self.model.integral_intensity(events_history, time_history, \
                                                 taus, resolution, mean, var, mask, sum = False)
                                                                               # [batch_size, seq_len * resolution] * n

        intensity_all_events = rearrange(intensity_all_events, 'b (s r) n -> b s r n', r = resolution)
                                                                               # [batch_size, seq_len, resolution, num_events]
        integral_all_events = rearrange(integral_all_events, 'b (s r) n -> b s r n', r = resolution)
                                                                               # [batch_size, seq_len, resolution, num_events]
        timestamp = rearrange(timestamp, 'b (s r) -> b s r', r = resolution)   # [batch_size, seq_len, resolution]

        events_next_index = torch.nn.functional.one_hot(events_next.long(), num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]
        events_next_index = rearrange(events_next_index, 'b s ne -> b s 1 ne') # [batch_size, seq_len, 1, num_events]
        intensity_i = reduce(intensity_all_events * events_next_index, 'b s r ne -> b s r', 'sum')
                                                                               # [batch_size, seq_len, resolution]
        integral_sum = reduce(integral_all_events, 'b s r ne -> b s r', 'sum') # [batch_size, seq_len, resolution]
        p_dist = intensity_i * torch.exp(-integral_sum)                        # [batch_size, seq_len, resolution]
        probability = reduce(p_dist[:, :, :-1] * timestamp[:, :, 1:], 'b s r -> b s', 'sum')
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
            p_gap = p_t_x - 1 / self.mae_threshold                             # [batch_size, seq_len]

            return p_gap
            
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
        
        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history, _ = self.divide_history_and_next(input_events)         # [batch_size, seq_len]
        _, mask_next = self.divide_history_and_next(mask)                      # [batch_size, seq_len]


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

        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history, _ = self.divide_history_and_next(input_events)         # [batch_size, seq_len]
        _, mask_next = self.divide_history_and_next(mask)                      # [batch_size, seq_len]


        mae = self.mean_absolute_error(events_history = events_history, time_history = time_history,\
                                           time_next = time_next, mask = mask_next, mean = mean, var = var, sum = False)
                                                                               # [batch_size, seq_len]
        probed_results, additional_plot, timestamp = self.model.model_probe_function(events_history, time_history, \
                                                                    time_next, resolution, mean, var, mask_next)
                                                                               # [batch_size, seq_len * resolution] * n
        
        accumulated_intensity = probed_results['accumulated_gradient']         # [batch_size, seq_len * resolution]
        accumulated_integral = probed_results['final_output']                  # [batch_size, seq_len * resolution]

        f1, top_k, probability_sum, tau_pred_all_event, maes_avg, maes = self.mean_absolute_error_per_event(input_time, input_events, mask, mean, var)
        mae = mae.detach().cpu().numpy()
        mae_per_event_pure_predict_avg, mae_per_event_avg = maes_avg
        mae_per_event_pure_predict, mae_per_event = maes

        accumulated_intensity = accumulated_intensity.detach().cpu().numpy()   # [batch_size, seq_len * resolution]
        accumulated_integral = accumulated_integral.detach().cpu().numpy()     # [batch_size, seq_len * resolution]

        probability_sum = probability_sum.detach().cpu().numpy()               # [batch_size, seq_len]
        mae_per_event_pure_predict_avg = mae_per_event_pure_predict_avg.detach().cpu().numpy()
                                                                               # [batch_size]
        mae_per_event_avg = mae_per_event_avg.detach().cpu().numpy()           # [batch_size]
        mae_per_event_pure_predict = mae_per_event_pure_predict.detach().cpu().numpy()
                                                                               # [batch_size, seq_len]
        mae_per_event = mae_per_event.detach().cpu().numpy()                   # [batch_size, seq_len]
        tau_pred_all_event = tau_pred_all_event.detach().cpu().numpy()         # [batch_size, seq_len, num_events]
        
        packed_values = zip(f1, top_k, probability_sum, tau_pred_all_event, mae, mae_per_event_pure_predict, mae_per_event_pure_predict_avg, \
                            mae_per_event, mae_per_event_avg, time_next, mask_next, accumulated_intensity, accumulated_integral)

        for idx, (f1_per_seq, top_k_per_seq, probability_sum_per_seq, tau_pred_all_event_per_seq, 
                  mae_per_seq, mae_per_event_pure_predict_per_seq, mae_per_event_pure_predict_avg_per_seq,
                  mae_per_event_per_seq, mae_per_event_avg_per_seq,
                  time_next_per_seq, mask_per_seq, accumulated_intensity_per_seq, accumulated_integral_per_seq) \
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
            '''
            The prediction against all events
            '''
            data_tau_pred_all_event_per_seq = {
                'x': list(range(seq_len)) * self.num_events,
                'y': np.log(1 + tau_pred_all_event_per_seq[:seq_len, :]).flatten(),
                'marks': [f'Event {i}' for i in range(self.num_events)] * seq_len
            }

            '''
            Logarithm of pe-MAEs at each event
            '''
            data_maes_per_seq = {
                'x': list(range(seq_len)) * 3,
                'y': np.concatenate(
                    (np.log(1 + mae_per_event_pure_predict_per_seq[:seq_len]),
                     np.log(1 + mae_per_event_per_seq[:seq_len]),
                     np.log(1 + mae_per_seq[:seq_len]))
                ),
                'marks': ['MAE_k against prediction'] * seq_len +  ['MAE_k against real events'] * seq_len + ['MAE'] * seq_len
            }

            data_probability_sum_per_seq = {
                'x': torch.arange(seq_len),
                'y': probability_sum_per_seq[:seq_len]
            }

            # Point plot
            additional_plot[idx]['pointplot'] = [[
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
                ],
            ]

            # Line plot
            additional_plot[idx]['lineplot'] = [[
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
                'log_mae_k',
                {
                    'x': 'x',
                    'y': 'y',
                    'hue': 'marks',
                    'data': data_maes_per_seq,
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
                't_pred_all_event',
                {
                    'x': 'x',
                    'y': 'y',
                    'hue': 'marks',
                    'data': data_tau_pred_all_event_per_seq,
                    'markers': True
                }
            ]]
        
        accumulated_probability_per_seq = \
            accumulated_intensity_per_seq * np.exp(-accumulated_integral_per_seq)
                                                                               # [seq_len * resolution]
        accumulated_probability_reshaped_per_seq = \
            rearrange(accumulated_probability_per_seq, '(s r) -> s r', r = resolution)
                                                                               # [seq_len, resolution]
        accumulated_probability_reshaped_per_seq_at_event = accumulated_probability_reshaped_per_seq[:, 0]
        accumulated_probability_reshaped_per_seq_no_event = accumulated_probability_reshaped_per_seq[:, 1:].flatten()

        df_probability = {
            'distribution_values': accumulated_probability_per_seq
        }
        df_probability_event = {
            'distribution_values': accumulated_probability_reshaped_per_seq_at_event
        }
        df_probability_no_event = {
            'distribution_values': accumulated_probability_reshaped_per_seq_no_event
        }

        # distplot, confirming the spiking issue.
        additional_plot[idx]['displot'] = [[
            'distribution_of_probability_values_at_events',
            {
                'data': df_probability_event,
                "kind": "kde",
                'height': 4,
                'aspect': 0.7
            }
        ],
        [
            'distribution_of_probability_values_no_events',
            {
                'data': df_probability_no_event,
                "kind": "kde",
                'height': 4,
                'aspect': 0.7
            }
        ],
        [
            'distribution_of_probability_values',
            {
                'data': df_probability,
                "kind": "kde",
                'height': 4,
                'aspect': 0.7
            }
        ],
        ]

        return (probed_results, additional_plot), timestamp
    
    def time_loss_f(self, intensity, intensity_integral, mask, events_next, negative_loss):
        '''
        The definition of loss.
    
        Args:
            intensity:          [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
            intensity_integral: [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
            mask:               [batch_size, seq_len]
            events_next:        [batch_size, seq_len]
        '''
        neg_loss = 0

        if self.reverse_bottleneck:
            if self.event_toggle:
                if negative_loss:
                    intensity_mask = torch.nn.functional.one_hot(events_next.long(), num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]
                    # elude the nan loss caused by 0 intensity.
                    intensity += 1e-9                                          # [batch_size, seq_len, num_events]
                    sum_of_intensity_and_neg = reduce(intensity, '... ne -> ... ()', 'sum')
                                                                               # [batch_size, seq_len, 1]
                    log_posterior = - torch.log(intensity / sum_of_intensity_and_neg)
                                                                               # [batch_size, seq_len, num_events]
                    log_posterior = log_posterior * intensity_mask             # [batch_size, seq_len, num_events]
                    neg_loss = reduce(log_posterior, '... ne -> ...', 'sum')   # [batch_size, seq_len]
                    neg_loss = (neg_loss * mask).clamp(max = 15)               # [batch_size, seq_len]
                    neg_loss = torch.sum(neg_loss)

                intensity = reduce(intensity, '... ne -> ...', 'sum')          # [batch_size, seq_len]
                intensity_integral = reduce(intensity_integral, '... ne -> ...', 'sum')
                                                                               # [batch_size, seq_len]
                log_intensity = torch.log(intensity + 1e-9)
                log_p = -log_intensity + intensity_integral
            else:
                log_intensity = torch.log(intensity + 1e-9)                    # [batch_size, seq_len]
                log_p = -log_intensity + intensity_integral
        else:
            if self.event_toggle:
                if negative_loss:
                    intensity_mask = torch.nn.functional.one_hot(events_next.long(), num_classes = self.num_events)
                                                                       # [batch_size, seq_len, num_events]
                    # elude the nan loss caused by 0 intensity.
                    intensity += 1e-9                                          # [batch_size, seq_len, num_events]
                    sum_of_intensity_and_neg = reduce(intensity, '... ne -> ... ()', 'sum')
                                                                               # [batch_size, seq_len, 1]
                    log_posterior = - torch.log(intensity / sum_of_intensity_and_neg)
                                                                               # [batch_size, seq_len, num_events]
                    log_posterior = log_posterior * intensity_mask             # [batch_size, seq_len, num_events]
                    neg_loss = reduce(log_posterior, '... ne -> ...', 'sum')   # [batch_size, seq_len]
                    neg_loss = (neg_loss * mask).clamp(max = 15)               # [batch_size, seq_len]
                    neg_loss = torch.sum(neg_loss)

                intensity_mask = torch.nn.functional.one_hot(events_next.long(), num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]
                log_intensity = torch.log(intensity + 1e-9) * intensity_mask
                log_intensity = reduce(log_intensity, '... ne -> ...', 'sum')  # [batch_size, seq_len]
                intensity_integral = reduce(intensity_integral, '... ne -> ...', 'sum')
                                                                               # [batch_size, seq_len]
                log_p = -log_intensity + intensity_integral                    # [batch_size, seq_len]
            else:
                log_intensity = torch.log(intensity + 1e-9)
                log_p = -log_intensity + intensity_integral
    
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

        if model.module.event_toggle and not model.module.reverse_bottleneck:
            loss = time_loss
        else:
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