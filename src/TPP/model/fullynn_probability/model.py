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
                 split_comp_graph = True, negative_loss = False, additional_event_loss = False):
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
                             device = device)
        if reverse_bottleneck and event_toggle:
            pass

    def forward(self, input_time, input_events, mask, mean, var, evaluate = False):
        time_history, time_next = self.divide_history_and_next(input_time)     # 2 * [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # 2 * [batch_size, seq_len]
        _, mask_next = self.divide_history_and_next(mask)                      # [batch_size, seq_len]

        mae, f1, events_loss = 0, 0, 0
        if evaluate:
            mae = self.mean_absolute_error(events_history = events_history, time_history = time_history,\
                                           time_next = time_next, mask = mask_next, mean = mean, var = var)
            time_next_zero = torch.zeros_like(time_next)
            if self.event_toggle:
                time_next_zero = repeat(time_next_zero, 'b s -> b s ne', ne = self.num_events)
                                                                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
            probability_integral_to_infinite = self.model(events_history, time_history, time_next_zero, \
                                                          mean = mean, var = var, mask = mask_next)
                                                                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
            if self.event_toggle:
                events_pred_index = torch.argmax(probability_integral_to_infinite, dim = -1)[mask_next == 1].detach().cpu().numpy()
                events_true = events_next[mask_next == 1].detach().cpu().numpy()
                f1 = f1_score(y_true = events_true, y_pred = events_pred_index, average = 'macro')

        # preparing for multi-event training when needed
        if self.event_toggle:
            time_next = repeat(time_next, 'b s -> b s ne', ne = self.num_events)
                                                                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
        time_next.requires_grad = True
        # \int_{t}^{+\inf}{p(m, \tau|\mathcal{H})d\tau}
        probability_integral_from_t_to_infinite = self.model(events_history, time_history, time_next, mean = mean, var = var, mask = mask_next)
                                                                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]

        # probability distribution
        # p(m, t|\mathcal{H})
        probability_for_each_event = - torch.autograd.grad(
            outputs = probability_integral_from_t_to_infinite,
            inputs = time_next,
            grad_outputs = torch.ones_like(probability_integral_from_t_to_infinite),
            create_graph = True,
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

    def mean_absolute_error(self, events_history, time_history, time_next, mask, mean, var):
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
                            y_pred = probability_integral_per_seq.detach().cpu()
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

        mae_per_event_pure_predict = self.mean_absolute_error_per_event_worker(events_history, predict_index, time_history, time_next,
                                                                               p_x_predicted, resolution, mask_next, mean, var)
        mae_per_event = self.mean_absolute_error_per_event_worker(events_history, events_next, time_history, time_next, 
                                                                  p_x_real, resolution, mask_next, mean, var)
        
        mae_per_event_pure_predict_avg = torch.sum(mae_per_event_pure_predict, dim = -1) / mask_next.sum(dim = -1)
        mae_per_event_avg = torch.sum(mae_per_event, dim = -1) / mask_next.sum(dim = -1)

        return f1, top_k_acc, probability_integral_sum, (mae_per_event_pure_predict_avg, mae_per_event_avg), \
               (mae_per_event_pure_predict, mae_per_event)

    def evaluate_per_event(self, events_history, events_next, time_history, taus, resolution, mean, var, mask):
        # Train k FullyNN models for k different event types.
        if self.event_toggle:
            taus = repeat(taus, 'b s -> b s ne', ne = self.num_events)
                                                                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
        taus.requires_grad = True
        # \int_{t}^{+\inf}{p(m, \tau|\mathcal{H})d\tau}
        probability_integral_from_t_to_infinite = self.model(events_history, time_history, taus, 
                                                             mean = mean, var = var, mask = mask)
                                                                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
        # probability distribution
        # p(m, t|\mathcal{H})
        probability_for_each_event = - torch.autograd.grad(
            outputs = probability_integral_from_t_to_infinite,
            inputs = taus,
            grad_outputs = torch.ones_like(probability_integral_from_t_to_infinite)
        )[0]                                                                   # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
        check_tensor(probability_for_each_event)
        
        if self.event_toggle:
            events_next_index = torch.nn.functional.one_hot(events_next.long(), num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]
            probability_at_t = probability_for_each_event * events_next_index  # [batch_size, seq_len, num_events]
            probability_at_t = reduce(probability_at_t, 'b s ne -> b s', 'sum')# [batch_size, seq_len]
        else:
            probability_at_t = probability_for_each_event                      # [batch_size, seq_len]

        return probability_at_t

    def mean_absolute_error_per_event_worker(self, events_history, events_next,
        time_history, time_next, p_x, resolution, mask, mean, var):
        '''
        The input should be the original minibatch
        MAE evaluation part, dwg and fullynn exclusive

        '''
        def bisect_target(events_history, time_history, taus, mean, var):
            p_xt = self.evaluate_per_event(events_history, events_next, time_history, taus,
                                           resolution, mean, var, mask)        # [batch_size, seq_len]
            p_t_x = p_xt / p_x                                                 # [batch_size, seq_len]
            p_gap = (1 / self.mae_threshold) - p_t_x                           # [batch_size, seq_len]

            return p_gap
            
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

        f1, top_k, probability_sum, maes_avg, maes = self.mean_absolute_error_per_event(input_time, input_events, mask, mean, var)
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
                            mae_per_event, mae_per_event_avg, time_next, mask_next)

        for idx, (f1_per_seq, top_k_per_seq, probability_sum_per_seq, 
                  mae_per_event_pure_predict_per_seq, mae_per_event_pure_predict_avg_per_seq,
                  mae_per_event_per_seq, mae_per_event_avg_per_seq,
                  time_next_per_seq, mask_per_seq) \
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
            ]]

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
                event_prediction_without_time = - torch.log(probability_0_inf + 1e-12) * probability_mask
                                                                               # [batch_size, seq_len, num_events]
                loss_event_prediction_without_time = reduce(event_prediction_without_time, '... ne -> ...', 'sum')
                                                                               # [batch_size, seq_len]

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
        return [test_report[1],]
    
    metric_number = 1 # metric number is the length of the output of choose_metric