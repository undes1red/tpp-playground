import torch
import numpy as np
from sklearn.metrics import f1_score, top_k_accuracy_score, accuracy_score
from einops import rearrange, repeat, reduce

from src.TPP.model.tifib.submodel import TIFIB
from src.TPP.model.utils import BasicModule
from src.TPP.model.tifib.utils import *
from src.TPP.model.tifib.plot import *


class TIFIBModel(BasicModule):
    def __init__(self, d_history,
                 d_intensity,
                 dropout,
                 d_hidden,
                 n_layers,
                 n_head,
                 d_qk,
                 d_v,
                 mlp_layers,
                 nonlinear,
                 probability_threshold,
                 num_events,
                 device,
                 event_toggle = False, additional_event_loss = False,
                 denominator_shift = 0.0, pretrain = False, alpha = 0.5, beta = 0.25):
        super(TIFIBModel, self).__init__()
        self.device = device
        self.probability_threshold = probability_threshold
        self.num_events = num_events
        self.event_toggle = event_toggle
        self.additional_event_loss = additional_event_loss
        self.zero_shift_factor = 1e-12

        self.model = TIFIB(d_history = d_history, d_intensity = d_intensity, num_events = num_events, \
                           dropout = dropout, d_hidden = d_hidden, n_layers = n_layers, n_head = n_head, \
                           d_qk = d_qk, d_v = d_v, mlp_layers = mlp_layers, nonlinear = nonlinear, \
                           event_toggle = event_toggle, denominator_shift = denominator_shift, pretrain = pretrain, \
                           alpha = alpha, beta = beta, device = device)


    def divide_history_and_next(self, input):
        input_history, input_next = input[:, :-1].clone(), input[:, 1:].clone()
        return input_history, input_next


    def forward(self, input_time, input_events, mask, mean, var, evaluate):
        '''
        The entrance of the FullyNN wrapper.
        
        Args:
        * input_time    type: torch.tensor shape: [batch_size, seq_len + 1]
                        The original time sequence. We should extract the history and target sequence from it
                        by divide_history_and_next().
        * input_events  type: torch.tensor shape: [batch_size, seq_len + 1]
                        The original event sequence. We should extract the history and target sequence from it
                        by divide_history_and_next().
        * mask          type: torch.tensor shape: [batch_size, seq_len + 1]
                        We use mask to mask out unneeded outputs.
        * mean          type: float shape: N/A
                        The mean of all $ t_i - t_{i - 1} $ in the entire dataset. Dataloader is responsible to provide
                        this value if needed.
        * var           type: float shape: N/A
                        The mean of all $ t_i - t_{i - 1} $ in the entire dataset. Dataloader is responsible to provide
                        this value if needed.
        * evaluate      type: bool shape: N/A
                        perform a model training step when evaluate == False
                        perform a model evaluate step when evaluate == True
        
        Outputs:
        Refers to train() and evaluate()'s documentation for detailed information.

        '''
        return self.evaluate_procedure(input_time, input_events, mask, mean, var) if evaluate \
            else self.train_procedure(input_time, input_events, mask, mean, var)


    def train_procedure(self, input_time, input_events, mask, mean, var):
        time_history, time_next = self.divide_history_and_next(input_time)     # 2 * [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # 2 * [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]

        if self.event_toggle:
            time_next = repeat(time_next, 'b s -> b s ne', ne = self.num_events)
                                                                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
        time_next.requires_grad = True

        '''
        \int_{t}^{+\inf}{p(m, \tau|\mathcal{H})d\tau}
        '''
        probability_integral_from_t_to_infinite = self.model(events_history, time_history, time_next, mask_history, mean = mean, var = var)
                                                                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]

        '''
        the value of probability distribution at t, or p(m, t|\mathcal{H})
        '''
        probability_for_each_event = - torch.autograd.grad(
            outputs = probability_integral_from_t_to_infinite,
            inputs = time_next,
            grad_outputs = torch.ones_like(probability_integral_from_t_to_infinite),
            create_graph = True
        )[0]
        time_next.requires_grad = False
        check_tensor(probability_for_each_event)                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
        check_tensor(probability_integral_from_t_to_infinite)                  # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
        assert probability_for_each_event.shape == probability_integral_from_t_to_infinite.shape

        '''
        This part is only available when evnet_toggle = True
        '''
        log_probability_for_each_event = torch.log(probability_for_each_event + self.zero_shift_factor)
                                                                           # [batch_size, seq_len, num_events]
        events_probability = torch.nn.functional.softmax(log_probability_for_each_event, dim = -1)
                                                                           # [batch_size, seq_len, num_events]
        events_loss = torch.nn.functional.cross_entropy(rearrange(events_probability, 'b s ne -> b ne s'), \
                                                                  events_next.long(), reduction = 'none')
                                                                           # [batch_size, seq_len]
        events_loss = events_loss * mask_next                              # [batch_size, seq_len]
        events_loss = events_loss.sum()

        time_loss = self.nll_loss(probability = probability_for_each_event, mask_next = mask_next, events_next = events_next)
        the_number_of_events = mask_next.sum().item()

        if self.event_toggle and self.additional_event_loss:
            loss = time_loss + events_loss
        else:
            loss = time_loss

        return loss, time_loss, events_loss, the_number_of_events


    def evaluate_procedure(self, input_time, input_events, mask, mean, var):
        time_history, time_next = self.divide_history_and_next(input_time)     # 2 * [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # 2 * [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]
        the_number_of_events = mask_next.sum().item()
        
        mae, pred_time = self.mean_absolute_error(events_history = events_history, time_history = time_history,\
                                                  time_next = time_next, mask_history = mask_history, \
                                                  mask_next = mask_next, mean = mean, var = var)
                                                                               # 2 * [batch_size, seq_len]
        mae = mae.sum().item() / the_number_of_events

        if self.event_toggle:
            time_next = repeat(time_next, 'b s -> b s ne', ne = self.num_events)
                                                                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
        time_zero = torch.zeros_like(time_next, device = self.device)          # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]


        time_next.requires_grad = True                                         # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
        
        probability_integral_from_zero_to_infinite = self.model(events_history, time_history, time_zero, mask_history, mean = mean, var = var)
                                                                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
        probability_integral_from_time_next_to_infinite = self.model(events_history, time_history, time_next, mask_history, mean = mean, var = var)
                                                                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]

        probability_for_each_event_at_time_next = - torch.autograd.grad(
            outputs = probability_integral_from_time_next_to_infinite,
            inputs = time_next,
            grad_outputs = torch.ones_like(probability_integral_from_time_next_to_infinite)
        )[0]                                                                   # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]                

        time_next.requires_grad = False

        f1_pred, f1_pred_at_time_next = 0, 0
        if self.event_toggle:
            '''
            macro-F1 value, event predictions are made without time predictions.
            '''
            events_pred_index = torch.argmax(probability_integral_from_zero_to_infinite, dim = -1)[mask_next == 1]
            events_true = events_next[mask_next == 1]
            events_pred_index, events_true = move_from_tensor_to_ndarray(events_pred_index, events_true)
            f1_pred = f1_score(y_true = events_true, y_pred = events_pred_index, average = 'macro')

            '''
            macro-F1 value, event predictions are made with time predictions.
            '''
            events_pred_index_at_time_next = torch.argmax(probability_for_each_event_at_time_next, dim = -1)[mask_next == 1]
            events_true = events_next[mask_next == 1]
            events_pred_index_at_time_next, events_true = move_from_tensor_to_ndarray(events_pred_index_at_time_next, events_true)
            f1_pred_at_time_next = f1_score(y_true = events_true, y_pred = events_pred_index_at_time_next, average = 'macro')

            '''
            Event loss, event predictions are made with time predictions.
            '''
            log_probability_for_each_event_at_time_next = torch.log(probability_for_each_event_at_time_next + self.zero_shift_factor)
                                                                               # [batch_size, seq_len, num_events]
            events_probability = torch.nn.functional.softmax(log_probability_for_each_event_at_time_next, dim = -1)
                                                                               # [batch_size, seq_len, num_events]
            events_loss = torch.nn.functional.cross_entropy(rearrange(events_probability, 'b s ne -> b ne s'), \
                                                                      events_next.long(), reduction = 'none')
                                                                               # [batch_size, seq_len]
            events_loss = events_loss * mask_next                              # [batch_size, seq_len]
            events_loss = events_loss.sum()
    
        time_loss = self.nll_loss(probability = probability_for_each_event_at_time_next, mask_next = mask_next, events_next = events_next)

        return time_loss, events_loss, mae, f1_pred, f1_pred_at_time_next, the_number_of_events


    def nll_loss(self, probability, events_next, mask_next):
        '''
        The definition of loss.
    
        Args:
            probability:        [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
            events_next:        [batch_size, seq_len]
            mask_next:          [batch_size, seq_len]
        '''
        if self.event_toggle:
            probability_mask = torch.nn.functional.one_hot(events_next.long(), num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]
            log_probability = - torch.log(probability + self.zero_shift_factor) * probability_mask
            log_probability = reduce(log_probability, '... ne -> ...', 'sum')  # [batch_size, seq_len]
        else:
            log_probability = - torch.log(probability + self.zero_shift_factor)# [batch_size, seq_len]

        loss = log_probability * mask_next                                     # [batch_size, seq_len]
        loss = torch.sum(loss)

        return loss


    def mean_absolute_error_and_f1(self, events_history, time_history, events_next, time_next, mask_history, mask_next, mean, var):
        mae, pred_time = self.mean_absolute_error(events_history, time_history, time_next, mask_history, mask_next, mean, var)
        if self.event_toggle:
            time_next_pred = repeat(pred_time, 'b s -> b s ne', ne = self.num_events)
                                                                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
        time_next_pred.requires_grad = True                                    # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]

        probability_integral_from_pred_to_infinite = self.model(events_history, time_history, time_next_pred, mean = mean, var = var)
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


    def mean_absolute_error(self, events_history, time_history, time_next, mask_history, mask_next, mean, var):
        '''
        The input should be the original minibatch
        MAE evaluation part, dwg and fullynn exclusive
        '''
        def evaluate(integral_from_zero_to_inf, taus):
            if self.event_toggle:
                taus = repeat(taus, 'b s -> b s ne', ne = self.num_events)     # [batch_size, seq_len, num_events]
            probability_integral_from_t_to_inf = self.model(events_history, time_history, taus, mask_history, mean, var)
                                                                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
            # P_m(t) = \int_{0}^{t}{p(t|m, \mathcal{H})}
            probability_integral = integral_from_zero_to_inf - probability_integral_from_t_to_inf
                                                                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
            if self.event_toggle:
                probability_integral = reduce(probability_integral, '... ne -> ...', 'sum')
                                                                               # [batch_size, seq_len]
            return probability_integral

        def bisect_target(integral_from_zero_to_inf, taus):
            return evaluate(integral_from_zero_to_inf, taus) - self.probability_threshold
            
        def median_prediction(integral_from_zero_to_inf, l, r):
            for _ in range(50):
                c = (l + r)/2
                v = bisect_target(integral_from_zero_to_inf, c)
                l = torch.where(v < 0, c, l)
                r = torch.where(v >= 0, c, r)

            return (l + r)/2
        
        l = 0.0001*torch.ones_like(time_history, dtype = torch.float32)        # [batch_size, seq_len]
        r = 1e6*torch.ones_like(time_history, dtype = torch.float32)           # [batch_size, seq_len]
        time_next_zero = torch.zeros_like(time_next)                           # [batch_size, seq_len]
        if self.event_toggle:
            time_next_zero = repeat(time_next_zero, 'b s -> b s ne', ne = self.num_events)
                                                                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
        integral_from_zero_to_inf = self.model(events_history, time_history, time_next_zero, mask_history, mean = mean, var = var)
                                                                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
        tau_pred = median_prediction(integral_from_zero_to_inf, l, r)          # [batch_size, seq_len]
        gap = (tau_pred - time_next) * mask_next                               # [batch_size, seq_len]
        gap = torch.abs(gap)                                                   # [batch_size, seq_len]

        return gap, tau_pred


    def mean_absolute_error_e(self, events_history, events_next, time_history, time_next, mask_history, mask_next, mean, var):
        '''
        Well...We will do something totally different by performing event-wise MAE.
        First, predict the event types by \int_{t_i}^{+\infty}{\lambda^*_i(t)\exp(-\int_{t_0}^{\tau}{\lambda^*_i(t)dt})d\tau}
        Next, given time predictions. (Expectation? or probability bigger than 0.5?)
        '''
        time_zero = torch.zeros_like(time_next)                                # [batch_size, seq_len]
        # preparing for multi-event training when needed
        if self.event_toggle:
            time_zero = repeat(time_zero, 'b s -> b s ne', ne = self.num_events)
                                                                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]

        probability_integral_from_zero_to_infinite = \
            self.model(events_history, time_history, time_zero, mask_history, mean = mean, var = var)
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
            if self.num_events > 2:
                for k in range(1, self.num_events):
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

        # step 2: get the time prediction for that kind of event
        tau_pred_all_event = self.prediction_with_all_event_types(events_history, time_history, probability_integral_from_zero_to_infinite, \
                                                                  mask_history, mean, var)
                                                                               # [batch_size, seq_len, num_events]
        mae_per_event_pure_predict = torch.abs((tau_pred_all_event * predict_index_one_hot).sum(dim = -1) - time_next) * mask_next
                                                                               # [batch_size, seq_len, num_events]
        mae_per_event = torch.abs((tau_pred_all_event * events_next_one_hot).sum(dim = -1) - time_next) * mask_next
                                                                               # [batch_size, seq_len, num_events]

        mae_per_event_pure_predict_avg = torch.sum(mae_per_event_pure_predict, dim = -1) / mask_next.sum(dim = -1)
        mae_per_event_avg = torch.sum(mae_per_event, dim = -1) / mask_next.sum(dim = -1)

        return f1, top_k_acc, probability_integral_sum, tau_pred_all_event, (mae_per_event_pure_predict_avg, mae_per_event_avg), \
               (mae_per_event_pure_predict, mae_per_event)


    def prediction_with_all_event_types(self, events_history, time_history, p_m, mask_history, mean, var):
        '''
        The input should be the original minibatch
        MAE evaluation part, dwg and fullynn exclusive

        '''
        def evaluate_all_event(taus):
            # \int_{tau}^{+\inf}{p(m, \tau|\mathcal{H})d\tau}
            probability_integral_from_t_to_infinite = self.model(events_history, time_history, taus, mask_history, mean = mean, var = var)
                                                                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
            # \int_{0}^{tau}{p(m, \tau|\mathcal{H})d\tau}
            probability_from_zero_to_t = p_m - probability_integral_from_t_to_infinite
                                                                               # [batch_size, seq_len, num_events] if we need events else [batch_size, seq_len]
            return probability_from_zero_to_t

        def bisect_target(taus):
            p_mt = evaluate_all_event(taus)                                    # [batch_size, seq_len, num_events]
            p_t_m = p_mt / p_m                                                 # [batch_size, seq_len, num_events]
            p_gap = p_t_m - self.probability_threshold                         # [batch_size, seq_len, num_events]

            return p_gap
            
        def median_prediction(l, r):
            for _ in range(50):
                c = (l + r)/2
                v = bisect_target(c)
                l = torch.where(v < 0, c, l)
                r = torch.where(v >= 0, c, r)

            return (l + r)/2

        max_ = min(1e6, mean + 10 * var)
        
        l = 0.0001*torch.ones((*time_history.shape, self.num_events), dtype = torch.float32, device = self.device)
                                                                               # [batch_size, seq_len, num_events]
        r = max_*torch.ones((*time_history.shape, self.num_events), dtype = torch.float32, device = self.device)
                                                                               # [batch_size, seq_len, num_events]
        tau_pred = median_prediction(l, r)                                     # [batch_size, seq_len, num_events]

        return tau_pred


    def plot(self, minibatch, opt):
        plot_type_to_functions = {
            'intensity': self.intensity,
            'integral': self.integral,
            'probability': self.probability,
            'debug': self.debug
        }
    
        return plot_type_to_functions[opt.plot_type](minibatch, opt)


    def extract_plot_data(self, minibatch):
        '''
        This function extracts input_time, input_events, input_intensity, mask, mean, and var from the minibatch.

        Args:
        * minibatch  type: list shape: [[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]
                     data structure: [[input_time, input_events, score, mask], (mean, var)]
        
        Outputs:
        * input_time    type: torch.tensor shape: [batch_size, seq_len + 1]
                        Raw event timestamp sequence.
        * input_events  type: torch.tensor shape: [batch_size, seq_len + 1]
                        Raw event marks sequence.
        * mask          type: torch.tensor shape: [batch_size, seq_len + 1]
                        Raw mask sequence.
        * mean          type: int shape: N/A
                        The mean of all $ t_i - t_{i - 1} $ in the entire dataset. Dataloader is responsible to provide
                        this value if needed.
        * var           type: int shape: N/A
                        The mean of all $ t_i - t_{i - 1} $ in the entire dataset. Dataloader is responsible to provide
                        this value if needed.
        '''
        input_time, input_events, _, mask, input_intensity = minibatch[0]
        mean, var = minibatch[1]

        return input_time, input_events, input_intensity, mask, mean, var


    def intensity(self, input_data, opt):
        '''
        Function prober, used by tpp_ploter to draw plots.

        Args:
        * input_data  type: list shape: [[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]
                      The original minibatch. Detailed information is available in extract_plot_data()
        * resolution  type: int shape: N/A
                      How many interpretive numbers we have between an event interval?
        '''

        return NotImplementedError('IFIB is intensity-free. Therefore, it can not provide the plot for the intensity function.')


    def integral(self, input_data, opt):
        '''
        Function prober, used by tpp_ploter to draw plots.

        Args:
        * input_data  type: list shape: [[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]
                      The original minibatch. Detailed information is available in extract_plot_data()
        * resolution  type: int shape: N/A
                      How many interpretive numbers we have between an event interval?
        '''
        return NotImplementedError('IFIB is intensity-free. Therefore, it can not provide the plot for the intensity integral.')


    def probability(self, input_data, opt):
        '''
        Function prober, used by tpp_ploter to draw plots.

        Args:
        * input_data  type: list shape: [[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]
                      The original minibatch. Detailed information is available in extract_plot_data()
        * resolution  type: int shape: N/A
                      How many interpretive numbers we have between an event interval?
        '''
        self.model.eval()

        input_time, input_events, input_intensity, mask, mean, var = self.extract_plot_data(input_data)
        
        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]

        expand_probability, timestamp = \
            self.model.probability(events_history, time_history, time_next, mask_history, opt.resolution, mean, var)
                                                                               # [batch_size, seq_len, resolution, num_events] if we need events else [batch_size, seq_len, resolution] + [batch_size, seq_len, resolution]

        data = {
            'time_next': time_next,
            'events_next': events_next,
            'mask_next': mask_next,
            'expand_probability': expand_probability,
            'input_intensity': input_intensity
            }
        plots = plot_probability(data, timestamp, opt)
        return plots


    def debug(self, input_data, opt):
        '''
        Args:
        time: [batch_size(always 1), seq_len + 1]
              The original dataset records. 
        resolution: int
              How many interpretive numbers we have between an event interval?
        '''
        self.model.eval()

        input_time, input_events, input_intensity, mask, mean, var = self.extract_plot_data(input_data)

        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]

        mae, f1_1 = self.mean_absolute_error_and_f1(events_history, time_history, events_next, \
                                                    time_next, mask_history, mask_next, mean, var)
                                                                               # [batch_size, seq_len]
        
        f1_2, top_k, probability_sum, tau_pred_all_event, maes_avg, maes \
            = self.mean_absolute_error_e(events_history, events_next, time_history, time_next, mask_history, mask_next, mean, var)

        data, timestamp = self.model.model_probe_function(events_history, time_history, time_next, mask_history, \
                                                          mask_next, opt.resolution, mean, var)

        '''
        Append additional info into the data dict.
        '''
        data['events_next'] = events_next
        data['time_next'] = time_next
        data['mask_next'] = mask_next
        data['f1_after_time_pred'] = f1_1
        data['f1_before_time_pred'] = f1_2
        data['top_k'] = top_k
        data['probability_sum'] = probability_sum
        data['tau_pred_all_event'] = tau_pred_all_event
        data['mae_before_event'] = mae
        data['maes_after_event_avg'] = maes_avg
        data['maes_after_event'] = maes

        plots = plot_debug(data, timestamp, opt)

        return plots


    '''
    Evaluation over the entire dataset.
    '''
    def get_spearman_and_l1(self, input_data, opt):
        input_time, input_events, input_intensity, mask, mean, var = self.extract_plot_data(input_data)
        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # [batch_size, seq_len]
        _, mask_next = self.divide_history_and_next(mask)                      # [batch_size, seq_len]

        expand_probability, timestamp = \
            self.model.probability(events_history, time_history, time_next, opt.resolution, mean, var)
                                                                               # [batch_size, seq_len, resolution, num_events] if we need events else [batch_size, seq_len, resolution] + [batch_size, seq_len, resolution]

        if opt.event_toggle:
            expand_probability = expand_probability.sum(dim = -1)              # [batch_size, seq_len, resolution]
        true_probability = expand_true_probability(time_next, input_intensity, opt)
                                                                               # [batch_size, seq_len, resolution] or batch_size * None
        
        expand_probability, true_probability, timestamp = move_from_tensor_to_ndarray(expand_probability, true_probability, timestamp)
        zipped_data = zip(expand_probability, true_probability, timestamp, mask_next)

        spearman = 0
        l1 = 0
        for expand_probability_per_seq, true_probability_per_seq, timestamp_per_seq, mask_next_per_seq in zipped_data:
            seq_len = mask_next_per_seq.sum()

            spearman_per_seq = \
                spearmanr(expand_probability_per_seq[:seq_len, :].flatten(), true_probability_per_seq[:seq_len, :].flatten())[0]

            l1_per_seq = L1_distance_between_two_funcs(
                                        x = true_probability_per_seq[:seq_len, :], y = expand_probability_per_seq[:seq_len, :], \
                                        timestamp = timestamp_per_seq, resolution = opt.resolution)
            spearman += spearman_per_seq
            l1 += l1_per_seq

        batch_size = mask_next.shape[0]
        spearman /= batch_size
        l1 /= batch_size

        return spearman, l1
    

    def get_mae_and_f1(self, input_data, opt):
        input_time, input_events, input_intensity, mask, mean, var = self.extract_plot_data(input_data)
        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]

        mae, f1_1 = self.mean_absolute_error_and_f1(events_history, time_history, events_next, \
                                                    time_next, mask_history, mask_next, mean, var)
                                                                               # [batch_size, seq_len]
        
        return mae, f1_1

    
    def get_mae_e_and_f1(self, input_data, opt):
        input_time, input_events, input_intensity, mask, mean, var = self.extract_plot_data(input_data)
        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]

        f1_2, top_k, probability_sum, tau_pred_all_event, maes_avg, maes \
            = self.mean_absolute_error_e(events_history, events_next, time_history, time_next, mask_history, mask_next, mean, var)
        
        _, maes = move_from_tensor_to_ndarray(*maes)

        return maes, f1_2


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
        loss, time_loss, events_loss, the_number_of_events = model(         
                input_time = time_seq, input_events = event_seq, mask = mask, \
                    mean = mean, var = var, evaluate = False
        )
        
        loss.backward()
    
        time_loss = time_loss.item() / the_number_of_events
        events_loss = events_loss.item() / the_number_of_events
        fact = score.sum().item() / the_number_of_events
        
        return time_loss, fact, events_loss
    
    def evaluation_step(model, minibatch, device):
        ''' Epoch operation in evaluation phase '''
    
        model.eval()
        [time_seq, event_seq, score, mask], (mean, var) = minibatch
        time_loss, events_loss, mae, f1_pred, f1_pred_at_time_next, the_number_of_events = model(
                input_time = time_seq, input_events = event_seq, mask = mask, evaluate = True,\
                mean = mean, var = var
        )
    
        time_loss = time_loss.item() / the_number_of_events
        events_loss = events_loss.item() / the_number_of_events
        fact = score.sum().item() / the_number_of_events
        
        return time_loss, fact, events_loss, mae, f1_pred, f1_pred_at_time_next

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
            return [input[0], input[0] - input[1], input[2], input[3], input[4], input[5]]
        
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
            format_dict['f1_without_time'] = input[4]
            format_dict['f1_with_time'] = input[5]
            format_dict['num_format'] = {'absolute_loss': ':6.5f', 'relative_loss': ':6.5f',
                                         'events_loss': ':6.5f', 'mae': ':2.8f', 
                                         'f1_without_time': ':2.8f', 'f1_with_time': ':2.8f'}
            return format_dict
        
        return (train_log_print_format(input) if procedure == 'Training' else test_log_print_format(input))

    format_dict_length = 6
    
    
    def choose_metric(evaluation_report_format_dict, test_report_format_dict):
        '''
        [relative loss on evaluation dataset, relative loss on test dataset, event loss on test dataset]
        '''
        return [evaluation_report_format_dict['absolute_loss'], test_report_format_dict['absolute_loss']], \
               ['evaluation_absolute_loss', 'test_absolute_loss']
    
    metric_number = 2 # metric number is the length of the output of choose_metric