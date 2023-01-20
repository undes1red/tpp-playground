from .submodel import FullyNN
from ..utils import BasicModule
import torch
from sklearn.metrics import f1_score, top_k_accuracy_score, accuracy_score
from einops import rearrange, repeat, reduce
import numpy as np

def check_tensor(x):
    assert (x < 0).any() == False, 'Negative numbers detected!'
    assert torch.isfinite(x).all() == True, 'inf detected in input!'
    assert torch.isnan(x).any() == False, 'Nan detected in input!'

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
                 split_comp_graph = True, negative_loss = False, additional_event_loss = False,
                 denominator_shift = 0.0, pretrain = False, alpha = 0.5, beta = 0.1):
        super(FullyNNModel, self).__init__()
        self.device = device
        self.mae_threshold = mae_threshold
        self.num_events = num_events
        self.event_toggle = event_toggle
        self.reverse_bottleneck = reverse_bottleneck if split_comp_graph else False
        self.split_comp_graph = split_comp_graph
        self.negative_loss = negative_loss
        self.additional_event_loss = additional_event_loss

        self.model = FullyNN(d_history = d_history, d_intensity = d_intensity, num_events = num_events,
                             dropout = dropout, history_module = history_module, history_module_layers = history_module_layers,
                             mlp_layers = mlp_layers, nonlinear = nonlinear, event_toggle = event_toggle, n_head = n_head,
                             wq_nonneg = wq_nonneg, wk_nonneg = wk_nonneg, wv_nonneg = wv_nonneg, split_comp_graph = split_comp_graph, 
                             denominator_shift = denominator_shift, pretrain = pretrain, alpha = alpha, beta = beta, device = device)

    def forward(self, input_time, input_events, mask, mean, var, evaluate = False):
        time_history, time_next = self.divide_history_and_next(input_time)     # 2 * [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # 2 * [batch_size, seq_len]
        _, mask_next = self.divide_history_and_next(mask)                      # [batch_size, seq_len]

        mae, f1, events_loss = 0, 0, 0
        if evaluate:
            mae, pred_time = self.mean_absolute_error(events_history = events_history, time_history = time_history,\
                                           time_next = time_next, mask = mask_next, mean = mean, var = var, output_pred = True)
            if self.event_toggle:
                time_next_pred = repeat(pred_time, 'b s -> b s ne', ne = self.num_events)
                                                                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
            time_next_pred.requires_grad = True                                # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]

            probability_integral_from_pred_to_infinite = self.model(events_history, time_history, time_next_pred, \
                                                          mean = mean, var = var, mask = mask_next)
                                                                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
            probability_for_each_event = - torch.autograd.grad(
                outputs = probability_integral_from_pred_to_infinite,
                inputs = time_next_pred,
                grad_outputs = torch.ones_like(probability_integral_from_pred_to_infinite)
            )[0]                                                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]                

            if self.event_toggle:
                events_pred_index = torch.argmax(probability_for_each_event, dim = -1)[mask_next == 1].detach().cpu().numpy()
                events_true = events_next[mask_next == 1].detach().cpu().numpy()
                f1 = f1_score(y_true = events_true, y_pred = events_pred_index, average = 'macro')
                # f1 = f1_score(y_true = events_true, y_pred = events_pred_index, average = 'micro')
                # f1 = accuracy_score(y_true = events_true, y_pred = events_pred_index)

        # preparing for multi-event training when needed
        if self.event_toggle:
            time_next = repeat(time_next, 'b s -> b s ne', ne = self.num_events)
                                                                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
        time_next.requires_grad = True
        # \int_{t}^{+\inf}{p(m, \tau|\mathcal{H})d\tau}
        probability_integral_from_t_to_infinite = self.model(events_history, time_history, time_next, mean = mean, var = var, mask = mask_next)
                                                                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
        check_tensor(probability_integral_from_t_to_infinite)                  # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]

        # probability distribution
        # p(m, t|\mathcal{H})
        probability_for_each_event = - torch.autograd.grad(
            outputs = probability_integral_from_t_to_infinite,
            inputs = time_next,
            grad_outputs = torch.ones_like(probability_integral_from_t_to_infinite),
            create_graph = True
        )[0]
        check_tensor(probability_for_each_event)                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
        assert probability_for_each_event.shape == probability_integral_from_t_to_infinite.shape
        time_next.requires_grad = False

        '''
        This part is only available when evnet_toggle = True
        TODO: fix the loss calculation error when self.reverse_bottleneck = False and evnet_toggle = True
        '''
        probability_integral_from_zero_to_infinite = None
        if self.additional_event_loss:
            time_next_zero = torch.zeros_like(time_next, requires_grad = True) # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
            probability_integral_from_zero_to_infinite = self.model(events_history, time_history, time_next_zero, mean = mean, var = var, mask = mask_next)
                                                                               # [batch_size, seq_len, num_events]
    
        time_loss, events_loss = self.time_loss_f(probability = probability_for_each_event, 
                                     probability_0_inf = probability_integral_from_zero_to_infinite, \
                                     mask = mask_next, events_next = events_next)
        
        the_number_of_events = mask_next.sum()

        return time_loss, events_loss, mae, f1, the_number_of_events

    def evaluate(self, integral_from_zero_to_inf, events_history, time_history, taus, mean, var, mask):
        if self.event_toggle:
            taus = repeat(taus, 'b s -> b s ne', ne = self.num_events)         # [batch_size, seq_len, num_events]
        probability_integral_from_t_to_inf = self.model(events_history, time_history, taus, mean, var, mask)
                                                                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
        # P_m(t) = \int_{0}^{t}{p(t|m, \mathcal{H})}
        probability_integral = integral_from_zero_to_inf - probability_integral_from_t_to_inf
                                                                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
        
        if self.event_toggle:
            probability_integral = reduce(probability_integral, '... ne -> ...', 'sum')
                                                                               # [batch_size, seq_len]

        # if self.event_toggle:
        #     mingled_probability_integral = reduce(1 - probability_integral, '... ne -> ...', 'prod')
        #                                                                        # [batch_size, seq_len]
        # else:
        #     mingled_probability_integral = 1 - probability_integral            # [batch_size, seq_len]

        return probability_integral

    def divide_history_and_next(self, input):
        input_history, input_next = input[:, :-1].clone(), input[:, 1:].clone()
        return input_history, input_next
    
    def mean_absolute_error_and_f1(self, events_history, time_history, events_next, time_next, mask_history, mask_next, mean, var):
        mae, pred_time = self.mean_absolute_error(events_history, time_history, time_next, mask_next, mean, var, \
                                                  sum = False, output_pred = True)
        if self.event_toggle:
            time_next_pred = repeat(pred_time, 'b s -> b s ne', ne = self.num_events)
                                                                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
        time_next_pred.requires_grad = True                                    # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]

        probability_integral_from_pred_to_infinite = self.model(events_history, time_history, time_next_pred, \
                                                      mean = mean, var = var, mask = mask_next)
                                                                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
        probability_for_each_event = - torch.autograd.grad(
            outputs = probability_integral_from_pred_to_infinite,
            inputs = time_next_pred,
            grad_outputs = torch.ones_like(probability_integral_from_pred_to_infinite)
        )[0]                                                                   # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]                

        if self.event_toggle:
            events_pred_index = torch.argmax(probability_for_each_event, dim = -1)[mask_next == 1].detach().cpu().numpy()
            events_true = events_next[mask_next == 1].detach().cpu().numpy()
            f1 = f1_score(y_true = events_true, y_pred = events_pred_index, average = 'macro')
        
        return mae, f1

    def mean_absolute_error(self, events_history, time_history, time_next, mask, mean, var, sum = True, output_pred = False):
        '''
        The input should be the original minibatch
        MAE evaluation part, dwg and fullynn exclusive
        '''
        def bisect_target(integral_from_zero_to_inf, events_history, time_history, taus, mean, var):
            return self.evaluate(integral_from_zero_to_inf, events_history, time_history, taus, mean, var, mask) - 1 / self.mae_threshold
            
        def median_prediction(integral_from_zero_to_inf, events_history, time_history, l, r, mean, var):
            for _ in range(50):
                c = (l + r)/2
                v = bisect_target(integral_from_zero_to_inf, events_history, time_history, c, mean, var)
                l = torch.where(v < 0, c, l)
                r = torch.where(v >= 0, c, r)

            return (l + r)/2
        
        l = 0.0001*torch.ones_like(time_history, dtype = torch.float32)        # [batch_size, seq_len]
        r = 1e6*torch.ones_like(time_history, dtype = torch.float32)           # [batch_size, seq_len]

        time_next_zero = torch.zeros_like(time_next)                           # [batch_size, seq_len]
        if self.event_toggle:
            time_next_zero = repeat(time_next_zero, 'b s -> b s ne', ne = self.num_events)
                                                                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
        integral_from_zero_to_inf = self.model(events_history, time_history, time_next_zero, mean = mean, var = var, mask = mask)
                                                                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
        tau_pred = median_prediction(integral_from_zero_to_inf, events_history, time_history, l, r, mean, var)
                                                                               # [batch_size, seq_len]
        gap = (tau_pred - time_next) * mask                                    # [batch_size, seq_len]
        gap = torch.abs(gap)                                                   # [batch_size, seq_len]

        if sum:
            gap_mean = torch.sum(gap) / mask.sum()
            if output_pred:
                return gap_mean.item(), tau_pred
            else:
                gap_mean = torch.sum(gap) / mask.sum()
                return gap_mean.item()
        else:
            if output_pred:
                return gap, tau_pred
            else:
                return gap

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

        time_zero = torch.zeros_like(time_next)                                # [batch_size, seq_len]
        # preparing for multi-event training when needed
        if self.event_toggle:
            time_zero = repeat(time_zero, 'b s -> b s ne', ne = self.num_events)
                                                                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]

        probability_integral_from_zero_to_infinite = \
            self.model(events_history, time_history, time_zero, mean = mean, var = var, mask = mask_next)
                                                                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]

        probability_integral_sum = reduce(probability_integral_from_zero_to_infinite, 'b s ne -> b s', 'sum')
                                                                               # [batch_size, seq_len]
        predict_index = torch.argmax(probability_integral_from_zero_to_infinite, dim = -1)
                                                                               # [batch_size, seq_len]

        f1 = []
        top_k_acc = []
        for (events_next_per_seq, probability_integral_per_seq) in zip(events_next, probability_integral_from_zero_to_infinite):
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
                            y_pred = torch.argmax(probability_integral_per_seq, dim = -1).detach().cpu()
                        )
                    )
                    top_k_acc_single_event_seq.append(1.0)
                top_k_acc.append(top_k_acc_single_event_seq)

        predict_index_one_hot = torch.nn.functional.one_hot(predict_index.long(), num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]
        events_next_one_hot = torch.nn.functional.one_hot(events_next.long(), num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]
        p_x_predicted = reduce(probability_integral_from_zero_to_infinite * predict_index_one_hot, '... ne -> ...', 'sum')
                                                                               # [batch_size, seq_len]
        p_x_real = reduce(probability_integral_from_zero_to_infinite * events_next_one_hot, '... ne -> ...', 'sum')
                                                                               # [batch_size, seq_len]

        # step 2: get the time prediction for that kind of event
        if mean == 0:
            resolution = max(min(int(input_time.mean().item() // 0.005), 500), 1)
        else:
            resolution = max(min(int(mean // 0.005), 500), 1)
        
        tau_pred_all_event = self.prediction_with_all_event_types(events_history, events_next,
                                             time_history, time_next, probability_integral_from_zero_to_infinite, 
                                             probability_integral_from_zero_to_infinite, resolution, mask, mean, var)
                                                                               # [batch_size, seq_len, num_events]

        mae_per_event_pure_predict = self.mean_absolute_error_per_event_worker(events_history, predict_index, time_history, time_next,
                                                                               p_x_predicted, probability_integral_from_zero_to_infinite,
                                                                               resolution, mask_next, mean, var)
        mae_per_event = self.mean_absolute_error_per_event_worker(events_history, events_next, time_history, time_next, 
                                                                  p_x_real, probability_integral_from_zero_to_infinite,
                                                                  resolution, mask_next, mean, var)
        
        mae_per_event_pure_predict_avg = torch.sum(mae_per_event_pure_predict, dim = -1) / mask_next.sum(dim = -1)
        mae_per_event_avg = torch.sum(mae_per_event, dim = -1) / mask_next.sum(dim = -1)

        return f1, top_k_acc, probability_integral_sum, tau_pred_all_event, (mae_per_event_pure_predict_avg, mae_per_event_avg), \
               (mae_per_event_pure_predict, mae_per_event)

    def evaluate_per_event(self, events_history, events_next, time_history, taus, probability_integral_from_zero_to_infinite, 
                           resolution, mean, var, mask):
        # Train k FullyNN models for k different event types.
        if self.event_toggle:
            taus = repeat(taus, 'b s -> b s ne', ne = self.num_events)
                                                                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
        taus.requires_grad = True
        # \int_{t}^{+\inf}{p(m, \tau|\mathcal{H})d\tau}
        probability_integral_from_t_to_infinite = self.model(events_history, time_history, taus, 
                                                             mean = mean, var = var, mask = mask)
                                                                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
        # \int_{0}^{t}{p(m, \tau|\mathcal{H})d\tau}
        probability_integral_from_zero_to_t = probability_integral_from_zero_to_infinite - probability_integral_from_t_to_infinite
                                                                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]

        if self.event_toggle:
            events_next_index = torch.nn.functional.one_hot(events_next.long(), num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]
            probability_from_zero_to_t = probability_integral_from_zero_to_t * events_next_index
                                                                               # [batch_size, seq_len, num_events]
            probability_from_zero_to_t = reduce(probability_from_zero_to_t, 'b s ne -> b s', 'sum')
                                                                               # [batch_size, seq_len]
        else:
            probability_from_zero_to_t = probability_integral_from_zero_to_t   # [batch_size, seq_len]

        return probability_from_zero_to_t

    def evaluate_all_event(self, events_history, events_next, time_history, taus, probability_integral_from_zero_to_infinite, 
                           resolution, mean, var, mask):
        # \int_{t}^{+\inf}{p(m, \tau|\mathcal{H})d\tau}
        probability_integral_from_t_to_infinite = self.model(events_history, time_history, taus, 
                                                             mean = mean, var = var, mask = mask)
                                                                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
        # \int_{0}^{t}{p(m, \tau|\mathcal{H})d\tau}
        probability_from_zero_to_t = probability_integral_from_zero_to_infinite - probability_integral_from_t_to_infinite
                                                                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]

        return probability_from_zero_to_t

    def prediction_with_all_event_types(self, events_history, events_next,
        time_history, time_next, p_x, probability_integral_from_zero_to_infinite, resolution, mask, mean, var):
        '''
        The input should be the original minibatch
        MAE evaluation part, dwg and fullynn exclusive

        '''
        def bisect_target(events_history, time_history, taus, mean, var):
            p_xt = self.evaluate_all_event(events_history, events_next, time_history, taus, probability_integral_from_zero_to_infinite, 
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

        max_ = min(1e6, mean + 10 * var)
        
        l = 0.0001*torch.ones((*time_history.shape, self.num_events), dtype = torch.float32, device = self.device)
                                                                               # [batch_size, seq_len, num_events]
        r = max_*torch.ones((*time_history.shape, self.num_events), dtype = torch.float32, device = self.device)
                                                                               # [batch_size, seq_len, num_events]
        tau_pred = median_prediction(events_history, time_history, l, r, mean, var)
                                                                               # [batch_size, seq_len, num_events]

        return tau_pred

    def mean_absolute_error_per_event_worker(self, events_history, events_next,
        time_history, time_next, p_x, probability_integral_from_zero_to_infinite, resolution, mask, mean, var):
        '''
        The input should be the original minibatch
        MAE evaluation part, dwg and fullynn exclusive

        '''
        def bisect_target(events_history, time_history, taus, mean, var):
            p_xt = self.evaluate_per_event(events_history, events_next, time_history, taus, probability_integral_from_zero_to_infinite, 
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

        max_ = min(1e6, mean + 10 * var)
        
        l = 0.0001*torch.ones_like(time_history, dtype = torch.float32)        # [batch_size, seq_len]
        r = max_*torch.ones_like(time_history, dtype = torch.float32)          # [batch_size, seq_len]
        tau_pred = median_prediction(events_history, time_history, l, r, mean, var)
        gap = (tau_pred - time_next) * mask
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


        expand_probability, timestamp = \
                        self.model.probability(events_history, time_history, \
                                                      time_next, resolution, mean, var, mask_next)
                                                                               # [batch_size, seq_len * resolution]

        check_tensor(expand_probability)
        return expand_probability, timestamp

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


        probed_results, additional_plot, timestamp = self.model.model_probe_function(events_history, time_history, \
                                                                    time_next, resolution, mean, var, mask_next)
                                                                               # [batch_size, seq_len * resolution] * n

        mae = self.mean_absolute_error(events_history = events_history, time_history = time_history,\
                                           time_next = time_next, mask = mask_next, mean = mean, var = var, sum = False)
                                                                               # [batch_size, seq_len]
        f1, top_k, probability_sum, tau_pred_all_event, maes_avg, maes = self.mean_absolute_error_per_event(input_time, input_events, mask, mean, var)
        mae = mae.detach().cpu().numpy()
        mae_per_event_pure_predict_avg, mae_per_event_avg = maes_avg
        mae_per_event_pure_predict, mae_per_event = maes

        accumulated_probability_distribution = probed_results['accumulated_gradient'].detach().cpu().numpy()
                                                                               # [batch_size, seq_len * resolution]

        probability_sum = probability_sum.detach().cpu().numpy()               # [batch_size, seq_len]
        mae_per_event_pure_predict_avg = mae_per_event_pure_predict_avg.detach().cpu().numpy()
                                                                               # [batch_size]
        mae_per_event_avg = mae_per_event_avg.detach().cpu().numpy()           # [batch_size]
        mae_per_event_pure_predict = mae_per_event_pure_predict.detach().cpu().numpy()
                                                                               # [batch_size, seq_len]
        mae_per_event = mae_per_event.detach().cpu().numpy()                   # [batch_size, seq_len]
        tau_pred_all_event = tau_pred_all_event.detach().cpu().numpy()         # [batch_size, seq_len, num_events]
        
        packed_values = zip(f1, top_k, probability_sum, tau_pred_all_event, mae, mae_per_event_pure_predict, mae_per_event_pure_predict_avg, \
                            mae_per_event, mae_per_event_avg, time_next, mask_next, accumulated_probability_distribution)

        for idx, (f1_per_seq, top_k_per_seq, probability_sum_per_seq, tau_pred_all_event_per_seq, 
                  mae_per_seq, mae_per_event_pure_predict_per_seq, mae_per_event_pure_predict_avg_per_seq,
                  mae_per_event_per_seq, mae_per_event_avg_per_seq,
                  time_next_per_seq, mask_per_seq, accumulated_probability_distribution_per_seq) \
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
                'marks': ['MAE_k against prediction'] * seq_len + ['MAE_k against real events'] * seq_len + ['MAE'] * seq_len
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
            accumulated_probability_distribution_reshaped_per_seq = \
                rearrange(accumulated_probability_distribution_per_seq, '(s r) -> s r', r = resolution)
            accumulated_probability_distribution_per_seq_at_event = accumulated_probability_distribution_reshaped_per_seq[:, 0]
            accumulated_probability_distribution_per_seq_no_event = accumulated_probability_distribution_reshaped_per_seq[:, 1:].flatten()

            df_probability = {
                'distribution_values': accumulated_probability_distribution_per_seq
            }
            df_probability_event = {
                'distribution_values': accumulated_probability_distribution_per_seq_at_event
            }
            df_probability_no_event = {
                'distribution_values': accumulated_probability_distribution_per_seq_no_event
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
    
    def time_loss_f(self, probability, probability_0_inf, events_next, mask):
        '''
        The definition of loss.
    
        Args:
            probability:        [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
            probability_0_inf:  [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
            mask:               [batch_size, seq_len]
            events_next:        [batch_size, seq_len]
        '''
        if self.event_toggle:
            probability_mask = torch.nn.functional.one_hot(events_next.long(), num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]
            log_probability = - torch.log(probability + 1e-12) * probability_mask
            log_probability = reduce(log_probability, '... ne -> ...', 'sum')
                                                                               # [batch_size, seq_len]
        else:
            log_probability = - torch.log(probability + 1e-12)                 # [batch_size, seq_len]

        loss_event_prediction_without_time = 0
        if self.additional_event_loss:
            assert probability_0_inf is not None
            if self.event_toggle:
                probability_mask = torch.nn.functional.one_hot(events_next.long(), num_classes = self.num_events)
                event_prediction_without_time = - torch.log(probability_0_inf + 1e-12) * probability_mask
                                                                               # [batch_size, seq_len, num_events]
                loss_event_prediction_without_time = reduce(event_prediction_without_time, '... ne -> ...', 'sum')
                                                                               # [batch_size, seq_len]
            else:
                loss_event_prediction_without_time = 0

        original_loss = log_probability + loss_event_prediction_without_time   # [batch_size, seq_len]
        original_loss = torch.clamp(original_loss, max = 15) * mask            # [batch_size, seq_len]
        original_pred_wo_t = loss_event_prediction_without_time * mask         # [batch_size, seq_len]
        
        loss = torch.sum(original_loss)
        event_loss = torch.sum(original_pred_wo_t)

        return loss, event_loss

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

        if model.module.event_toggle and model.module.additional_event_loss:
            loss = time_loss + events_loss
        else:
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
        return [test_report[0],]
    
    metric_number = 1 # metric number is the length of the output of choose_metric