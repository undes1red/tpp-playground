from einops import rearrange, reduce, repeat
from sklearn.metrics import f1_score, top_k_accuracy_score, accuracy_score
import torch
import numpy as np
from scipy.stats import spearmanr

from src.TPP.model.utils import BasicModule
from src.TPP.model.rmtpp.rmtpp import RMTPPModule
from src.TPP.model.rmtpp.utils import *
from src.TPP.model.rmtpp.plot import *


class RMTPP(BasicModule):
    def __init__(self, device, input_size, hidden_size, history_encoder_layers, dropout, num_events, event_toggle, 
                 output_size, limited_history_norm, original_mark_generation, time_scalar_min = 1e-4, 
                 probability_threshold = 0.5):
        super(RMTPP, self).__init__()
        self.device = device
        self.num_events = num_events
        self.event_toggle = event_toggle
        self.limited_history_norm = limited_history_norm
        self.original_mark_generation = original_mark_generation
        self.probability_threshold = probability_threshold
        self.zero_shift = 1e-12

        self.model = RMTPPModule(input_size = input_size, hidden_size = hidden_size, history_encoder_layers = history_encoder_layers, 
                                 dropout = dropout, num_events = num_events, output_size = output_size, event_toggle = self.event_toggle, 
                                 limited_history_norm = limited_history_norm, original_mark_generation = original_mark_generation, 
                                 time_scalar_min = time_scalar_min, device = device)


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


    def divide_history_and_next(self, input):
        history, next = input[:, :-1].clone(), input[:, 1:].clone()
        return history, next                                                   # [batch_size, seq_len, 1] or [batch_size, seq_len]


    def train_procedure(self, events, time, mask, mean, var):
        events_history, events_next = self.divide_history_and_next(events)
                                                                               # [batch_size, seq_len]
        time_history, time_next = self.divide_history_and_next(time)           # [batch_size, seq_len]
        _, mask_next = self.divide_history_and_next(mask)                      # [batch_size, seq_len]

        integral, intensity, mark, constant = self.model(events_history, time_history, time_next, mean, var)
                                                                               # [batch_size, seq_len, num_events] if self.event_toggle else [batch_size, seq_len, 1] * 2, [batch_size, seq_length, num_events], and [batch_size, seq_len, num_events] if self.event_toggle else [batch_size, seq_len, 1]

        check_tensor(intensity)
        check_tensor(integral)

        loss, time_loss, events_loss, the_number_of_events = \
                   self.loss_function(intensity, integral, mark, events_next, mask_next)

        return loss, time_loss, events_loss, the_number_of_events, constant


    def evaluate_procedure(self, events, time, mask, mean, var):
        events_history, events_next = self.divide_history_and_next(events)     # [batch_size, seq_len]
        time_history, time_next = self.divide_history_and_next(time)           # [batch_size, seq_len]
        _, mask_next = self.divide_history_and_next(mask)                      # [batch_size, seq_len]

        '''
        Calculating MAE here.
        '''
        mae, pred_time = self.mean_absolute_error(events_history, time_history, time_next, mask_next, mean, var)
                                                                               # [batch_size, seq_len] * 2

        integral_time_next, intensity_time_next, mark_time_next, constant_time_next \
                   = self.model(events_history, time_history, time_next, mean, var)
                                                                               # [batch_size, seq_len, num_events] if self.event_toggle else [batch_size, seq_len, 1] * 2, [batch_size, seq_length, num_events], and [batch_size, seq_len, num_events] if self.event_toggle else [batch_size, seq_len, 1]
        integral_pred_time, intensity_pred_time, mark_pred_time, constant_pred_time \
                   = self.model(events_history, time_history, pred_time, mean, var)
                                                                               # [batch_size, seq_len, num_events] if self.event_toggle else [batch_size, seq_len, 1] * 2, [batch_size, seq_length, num_events], and [batch_size, seq_len, num_events] if self.event_toggle else [batch_size, seq_len, 1]
        check_tensor(intensity_time_next)
        check_tensor(integral_time_next)
        check_tensor(intensity_pred_time)
        check_tensor(integral_pred_time)
        
        loss_time_next, time_loss_time_next, events_loss_time_next, the_number_of_events = \
                   self.loss_function(intensity_time_next, integral_time_next, mark_time_next, events_next, mask_next)
        loss_pred_time, time_loss_pred_time, events_loss_pred_time, the_number_of_events = \
                   self.loss_function(intensity_pred_time, integral_pred_time, mark_pred_time, events_next, mask_next)

        if self.original_mark_generation:
            predicted_events = torch.argmax(mark_pred_time, dim = -1)[mask_next == 1]
            events_true = events_next[mask_next == 1]
            predicted_events, events_true = move_from_tensor_to_ndarray(predicted_events, events_true)
                                                                           # [batch_size, seq_len] * 2
            f1 = f1_score(y_pred = predicted_events, y_true = events_true, average = 'macro')
        else:
            predicted_events = torch.argmax(intensity_pred_time, dim = -1)[mask_next == 1]
            events_true = events_next[mask_next == 1]
            predicted_events, events_true = move_from_tensor_to_ndarray(predicted_events, events_true)
                                                                           # [batch_size, seq_len] * 2
            f1 = f1_score(y_pred = predicted_events, y_true = events_true, average = 'macro')


        return loss_time_next, time_loss_time_next, events_loss_time_next, loss_pred_time, time_loss_pred_time, events_loss_pred_time, \
               mae, f1, the_number_of_events, constant_time_next, constant_pred_time


    def loss_function(self, intensity, integral, mark, events_next, mask_next):
        # temporal point process loss
        # intensity shape: [batch, seq_length]
        # so does tensor mask.

        loss = 0
        time_loss, events_loss = 0, 0
        if self.event_toggle:
            events_loss = \
                torch.nn.functional.cross_entropy(input = mark.transpose(1, 2), \
                                                  target = events_next.long(), \
                                                  reduction = 'none')          # [batch_size, seq_len]
            events_loss = events_loss * mask_next
            events_loss = events_loss.sum()
        else:
            events_loss = torch.tensor(0., device = self.device)

        if self.original_mark_generation:
            time_loss = -torch.log(intensity + self.zero_shift) + integral     # [batch_size, seq_len]
            time_loss = time_loss * mask_next
            time_loss = time_loss.sum()

            loss = time_loss + events_loss
        else:
            events_mask = torch.nn.functional.one_hot(events_next.long(), num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]
            intensity = (intensity * events_mask).sum(dim = -1)                # [batch_size, seq_len]
            integral = integral.sum(dim = -1)                                  # [batch_size, seq_len]
            time_loss = - torch.log(intensity + self.zero_shift) + integral    # [batch_size, seq_len]
            time_loss = time_loss * mask_next                                  # [batch_size, seq_len]
            time_loss = time_loss.sum()
            
            loss = time_loss

        return loss, time_loss, events_loss, mask_next.sum()


    def mean_absolute_error_and_f1(self, events_history, time_history, events_next, time_next, mask_history, mask_next, mean, var):
        mae, pred_time = self.mean_absolute_error(events_history, time_history, time_next, mask_next, mean, var)
        integral, intensity, mark, constant = self.model(events_history, time_history, pred_time, mean, var)
        if self.original_mark_generation:
            predicted_events = torch.argmax(mark, dim = -1)[mask_next == 1].detach().cpu().numpy()
                                                                           # [batch_size, seq_len]
            events_true = events_next[mask_next == 1].detach().cpu().numpy()
            f1 = f1_score(y_pred = predicted_events, y_true = events_true, average = 'macro')
        else:
            predicted_events = torch.argmax(intensity, dim = -1)[mask_next == 1].detach().cpu().numpy()
                                                                           # [batch_size, seq_len]
            events_true = events_next[mask_next == 1].detach().cpu().numpy()
            f1 = f1_score(y_pred = predicted_events, y_true = events_true, average = 'macro')

        return mae, f1


    def mean_absolute_error(self, events_history, time_history, time_next, mask_next, mean, var):
        '''
        The input should be the original minibatch
        MAE evaluation part, dwg and fullynn exclusive
        '''
        def evaluate(taus):
            integral, _, _, _ = self.model(events_history, time_history, taus, mean, var)
                                                                               # [batch_size, seq_len, num_events] if self.event_toggle else [batch_size, seq_len]
            if self.event_toggle and not self.original_mark_generation:
                integral = integral.sum(dim = -1)

            return integral

        def bisect_target(taus):
            return evaluate(taus) + torch.log(1 - torch.tensor(self.probability_threshold, device = self.device))
            
        def median_prediction(l, r):
            for _ in range(50):
                c = (l + r)/2
                v = bisect_target(c)
                l = torch.where(v < 0, c, l)
                r = torch.where(v >= 0, c, r)

            return (l + r)/2
        
        l = 0.0001*torch.ones_like(time_history, dtype = torch.float32)        # [batch_size, seq_len]
        r = 1e6*torch.ones_like(time_history, dtype = torch.float32)           # [batch_size, seq_len]
        tau_pred = median_prediction(l, r)                                     # [batch_size, seq_len]
        gap = (tau_pred - time_next) * mask_next                               # [batch_size, seq_len]
        gap = torch.abs(gap)                                                   # [batch_size, seq_len]

        return gap, tau_pred


    def mean_absolute_error_e(self, events_history, events_next, time_history, time_next, mask_next, mean, var):
        '''
        MAE-E evaluation module.

        Args:
        * events_history  type: torch.tensor shape: [batch_size, seq_len]
                          Historical event sequences. Commonly, this sequence is a slice of 
                          the original event sequence from 0 to seq_len - 1(included).
        * events_next     type: torch.tensor shape: [batch_size, seq_len]
                          The mark of the events that we need to predict.
        * time_history    type: torch.tensor shape: [batch_size, seq_len]
                          Historical time sequences. Similar to events_history, we always generate
                          this sequence as a slice of the original time sequence from 0 to seq_len - 1(included).
        * time_next       type: torch.tensor shape: [batch_size, seq_len, num_events] if self.event_toggle else [batch_size, seq_len]
                          When the next event actually happens. 
        * mask_next       type: torch.tensor shape: [batch_size, seq_len]
                          Needed mask to mask out unneeded loss values.
        * mean            type: float shape: N/A
                          The mean of all $ t_i - t_{i - 1} $ in the entire dataset. Dataloader is responsible to provide
                          this value if needed.
        * var             type: float shape: N/A
                          The mean of all $ t_i - t_{i - 1} $ in the entire dataset. Dataloader is responsible to provide
                          this value if needed.
        Outputs:
        * mae             type: torch.tensor shape: [batch_size, seq_len]
                          MAE(Mean Absolute Error) between predicted time and ground truth.
        * tau_pred        type: torch.tensor shape: [batch_size, seq_len]
                          Time predicted by the sum of all intensity functions $ \lambda^*(m, t) $ over $ m $.
        '''
        if self.original_mark_generation:
            raise Exception('The original RMTPP model is in fact a TPP model, not a native MTPP model. So MAE-E calculation is unavailable.')

        '''
        Set the memory limit
        '''
        memory_ceiling = 2e7

        '''
        set a relatively large number as the infinity and decide resolution based on this large value and
        the memory_ceiling.
        '''
        if mean == 0 and var == 1:
            max_ = time_next.mean() + 10 * time_next.var()
        else:
            max_ = mean + 10 * var
        
        max_ = min(1e6, max_)
        time_next_inf = torch.ones_like(time_history, device = self.device) * max_
                                                                               # [batch_size, seq_len]
        resolution = max(int(max_ // 0.005), 100)

        _, seq_len = events_next.shape
        if seq_len * resolution * self.num_events > memory_ceiling:
            resolution = int(memory_ceiling // (seq_len * self.num_events))

        '''
        Step 1: obtain p^*(m) = \int_{t_l}^{+infty}{p(m, t)\dt}
        '''
        expand_integral_to_inf, expand_intensity_to_inf, time_interval \
                = self.model.integral_intensity_time_next_2d(events_history, time_history, time_next_inf, resolution, mean, var)
                                                                               # [batch_size, seq_len, resolution, num_events]

        '''
        Step 2: provide event predictions
        '''        
        expand_probability_per_event = expand_intensity_to_inf * torch.exp(-expand_integral_to_inf.sum(dim = -1, keepdim = True))
                                                                               # [batch_size, seq_len, resolution, num_events]
        expand_probability_per_event_for_monte_carlo = expand_probability_per_event[:, :, :-1, :]
                                                                               # [batch_size, seq_len, resolution - 1, num_events]
        time_interval_used_for_monte_carlo = time_interval[:, :, 1:].unsqueeze(dim = -1)
                                                                               # [batch_size, seq_len, resolution - 1, 1]
        probability_integral = expand_probability_per_event_for_monte_carlo * time_interval_used_for_monte_carlo
                                                                               # [batch_size, seq_len, resolution - 1, num_events]
        p_m = reduce(probability_integral, 'b s r ne -> b s ne', 'sum')        # [batch_size, seq_len, num_events]
        probability_integral_sum = reduce(p_m, 'b s ne -> b s', 'sum')         # [batch_size, seq_len]
        predict_index = torch.argmax(p_m, dim = -1)                            # [batch_size, seq_len]

        '''
        Step 3: calculate macro-F1 and top-K accuracy
        '''
        f1 = []
        top_k_acc = []
        for (events_next_per_seq, p_m_per_seq) in zip(events_next, p_m):
            f1.append(f1_score(y_true = events_next_per_seq.detach().cpu(),
                               y_pred = torch.argmax(p_m_per_seq, dim = -1).detach().cpu(), average = 'macro'))
            
            top_k_acc_single_event_seq = []
            if self.num_events > 2:
                for k in range(1, self.num_events):
                    top_k_acc_single_event_seq.append(
                        top_k_accuracy_score(y_true = events_next_per_seq.detach().cpu(),
                                             y_score = p_m_per_seq.detach().cpu(),
                                             k = k,
                                             labels = np.arange(self.num_events))
                    )
            else:
                top_k_acc_single_event_seq.append(
                    accuracy_score(
                        y_true = events_next_per_seq.detach().cpu(),
                        y_pred = p_m_per_seq.detach().cpu()
                    )
                )
            top_k_acc.append(top_k_acc_single_event_seq)

        predict_index_one_hot_mask = torch.nn.functional.one_hot(predict_index.long(), num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]
        events_next_one_hot_mask = torch.nn.functional.one_hot(events_next.long(), num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]
        '''
        Step 4: get the time prediction for all, predicted, and real events.
        '''
        if mean == 0:
            resolution = max(min(int(time_next.mean().item() // 0.005), 500), 10)
        else:
            resolution = max(min(int(mean // 0.005), 500), 10)

        tau_pred_all_event = self.prediction_with_all_event_types(events_history, time_history, p_m, resolution, mean, var, max_)
                                                                               # [batch_size, seq_len, num_events]
        mae_per_event_with_predict_index = torch.abs(((tau_pred_all_event * predict_index_one_hot_mask).sum(dim = -1)) - time_next) * mask_next
                                                                               # [batch_size, seq_len]
        mae_per_event_with_event_next = torch.abs(((tau_pred_all_event * events_next_one_hot_mask).sum(dim = -1)) - time_next) * mask_next
                                                                               # [batch_size, seq_len]

        mae_per_event_with_predict_index_avg = torch.sum(mae_per_event_with_predict_index, dim = -1) / mask_next.sum(dim = -1)
        mae_per_event_with_event_next_avg = torch.sum(mae_per_event_with_event_next, dim = -1) / mask_next.sum(dim = -1)

        return f1, top_k_acc, probability_integral_sum, tau_pred_all_event, \
               (mae_per_event_with_predict_index_avg, mae_per_event_with_event_next_avg), \
               (mae_per_event_with_predict_index, mae_per_event_with_event_next)


    def prediction_with_all_event_types(self, events_history, time_history, p_m, resolution, mean, var, max_val):
        '''
        The time prediction of every marker whose probability is not 0.

        Still, this function is currently buggy.

        Args:
        * events_history  type: torch.tensor shape: [batch_size, seq_len]
                          Historical event sequences. Commonly, this sequence is a slice of 
                          the original event sequence from 0 to seq_len - 1(included). 
        * time_history    type: torch.tensor shape: [batch_size, seq_len]
                          Historical time sequences. Similar to events_history, we always generate
                          this sequence as a slice of the original time sequence from 0 to seq_len - 1(included).
        * p_m             type: torch.tensor shape: [batch_size, seq_len]
                          the value of p(m) with given markers.
        * resolution      type: int shape: N/A
                          How many values do we need in each time interval [t_{i}, t_{i + 1}].
        * mask_next       type: torch.tensor shape: [batch_size, seq_len]
                          Needed mask to mask out unneeded loss values.
        * mean            type: float shape: N/A
                          The mean of all $ t_i - t_{i - 1} $ in the entire dataset. Dataloader is responsible to provide
                          this value if needed.
        * var             type: float shape: N/A
                          The mean of all $ t_i - t_{i - 1} $ in the entire dataset. Dataloader is responsible to provide
                          this value if needed.
        * max_val         type: float shape: N/A
                          The upper bound used in the bisect method.
        Outputs:
        * tau_pred        type: torch.tensor shape: [batch_size, seq_len]
                          Time predicted by the sum of all intensity functions $ \lambda^*(m, t) $ over $ m $.
        '''
        def evaluate_all_event(taus):
            '''
            placeholder
            '''
            # Train k FullyNN models for k different event types.
            integral_all_events, intensity_all_events, time_interval \
                    = self.model.integral_intensity_time_next_3d(events_history, time_history, taus, resolution, mean, var)
                                                                               # 2 * [batch_size, seq_len, resolution, num_events, num_events] + [batch_size, seq_len, resolution, num_events]
            event_mask = torch.diag(torch.ones(self.num_events, device = self.device))
                                                                               # [num_events, num_events]
            event_mask = repeat(event_mask, 'ne ne1 -> 1 1 1 ne ne1')          # [batch_size, seq_len, resolution, num_events, num_events]
            intensity_all_events = reduce(intensity_all_events * event_mask, '... ne -> ...', 'sum')
                                                                               # [batch_size, seq_len, resolution, num_events]
            integral_all_events = reduce(integral_all_events, 'b s r ne ne1 -> b s r ne', 'sum')
                                                                               # [batch_size, seq_len, resolution, num_events]
            
            p_dist = intensity_all_events * torch.exp(-integral_all_events)    # [batch_size, seq_len, resolution, num_events]
            
            p_dist_for_monte_carlo = p_dist[:, :, :-1, :]                      # [batch_size, seq_len, resolution - 1, num_events]
            time_interval_for_monte_carlo = time_interval[:, :, 1:, :]         # [batch_size, seq_len, resolution - 1, num_events]
            probability = reduce(p_dist_for_monte_carlo * time_interval_for_monte_carlo, 'b s r ne -> b s ne', 'sum')
                                                                               # [batch_size, seq_len, num_events]
            return probability

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
        
        l = 0.0001*torch.ones((*time_history.shape, self.num_events), dtype = torch.float32, device = self.device)
                                                                               # [batch_size, seq_len, num_events]
        r = max_val*torch.ones((*time_history.shape, self.num_events), dtype = torch.float32, device = self.device)
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
        self.model.eval()

        input_time, input_events, input_intensity, mask, mean, var = self.extract_plot_data(input_data)
        
        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]

        expand_integral, expand_intensity, timestamp = \
            self.model.integral_intensity_time_next_2d(events_history, time_history, time_next, opt.resolution, mean, var)
                                                                               # 3 * [batch_size, seq_len, resolution, num_events]
        
        check_tensor(expand_integral)
        check_tensor(expand_intensity)
        assert expand_intensity.shape == expand_integral.shape

        data = {
            'time_next': time_next,
            'events_next': events_next,
            'mask_next': mask_next,
            'expand_intensity': expand_intensity,
            'input_intensity': input_intensity
            }
        plots = plot_intensity(data, timestamp, opt)
        
        return plots


    def integral(self, input_data, opt):
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

        expand_integral, expand_intensity, timestamp = \
            self.model.integral_intensity_time_next_2d(events_history, time_history, time_next, opt.resolution, mean, var)
                                                                               # 3 * [batch_size, seq_len, resolution, num_events]
        
        check_tensor(expand_integral)
        check_tensor(expand_intensity)
        assert expand_intensity.shape == expand_integral.shape

        data = {
            'time_next': time_next,
            'events_next': events_next,
            'mask_next': mask_next,
            'expand_integral': expand_integral,
            'input_intensity': input_intensity
            }
        plots = plot_integral(data, timestamp, opt)
        return plots


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

        expand_integral, expand_intensity, timestamp = \
            self.model.integral_intensity_time_next_2d(events_history, time_history, time_next, opt.resolution, mean, var)
                                                                               # 3 * [batch_size, seq_len, resolution, num_events]

        check_tensor(expand_integral)
        check_tensor(expand_intensity)
        assert expand_intensity.shape == expand_integral.shape
        expand_probability = expand_intensity * torch.exp(-expand_integral.sum(dim = -1, keepdim = True))
                                                                               # [batch_size, seq_len, resolution, num_events]

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

        data, timestamp = self.model.model_probe_function(events_history, time_history, \
                                                          time_next, opt.resolution, mean, var, mask_next)

        '''
        Append additional info into the data dict.
        '''
        data['events_next'] = events_next
        data['time_next'] = time_next
        data['mask_next'] = mask_next
        data['f1_after_time_pred'] = f1_1
        data['mae_before_event'] = mae

        # Only allowed when self.original_mark_generation is False
        if self.event_toggle and not self.original_mark_generation:
            f1_2, top_k, probability_sum, tau_pred_all_event, maes_avg, maes \
                = self.mean_absolute_error_e(events_history, events_next, time_history, time_next, mask_next, mean, var)

            data['f1_before_time_pred'] = f1_2
            data['top_k'] = top_k
            data['probability_sum'] = probability_sum
            data['tau_pred_all_event'] = tau_pred_all_event
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

        expand_integral, expand_intensity, timestamp = \
            self.model.integral_intensity_time_next_2d(events_history, time_history, time_next, opt.resolution, mean, var)
                                                                               # 3 * [batch_size, seq_len, resolution, num_events]

        check_tensor(expand_integral)
        check_tensor(expand_intensity)
        assert expand_intensity.shape == expand_integral.shape
        expand_probability = expand_intensity * torch.exp(-expand_integral.sum(dim = -1, keepdim = True))
                                                                               # [batch_size, seq_len, resolution, num_events]
        expand_probability = expand_probability.sum(dim = -1)                  # [batch_size, seq_len, resolution]
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
                                        timestamp = timestamp_per_seq, resolution = opt.resolution
                                        )
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
        raise NotImplemented("RMTPP is a TPP model, so MAE-E is not supported.")


    def train_step(model, minibatch, device):
        model.train()
        
        [time, events, score, mask], (mean, var) = minibatch                   # 4 * [batch_size, seq_len + 1]
        loss, time_loss, events_loss, the_number_of_events, constant \
                   = model(events, time, mask, mean, var, evaluate = False)

        loss.backward()

        loss = loss.item() / the_number_of_events
        time_loss = time_loss.item() / the_number_of_events
        events_loss = events_loss.item() / the_number_of_events
        fact = score.sum().item() / the_number_of_events
        constant_norm = torch.linalg.norm(constant).detach().item() / the_number_of_events

        return loss, time_loss, fact, events_loss, constant_norm


    def evaluation_step(model, minibatch, device):
        model.eval()

        [time, events, score, mask], (mean, var) = minibatch                   # 4 * [batch_size, seq_len + 1]
        loss_time_next, time_loss_time_next, events_loss_time_next, \
        loss_pred_time, time_loss_pred_time, events_loss_pred_time, \
        mae, f1, the_number_of_events, constant_time_next, \
        constant_pred_time = model(events, time, mask, mean, var, evaluate = True)
        
        # Loss values and other metrics at time_next
        loss_time_next = loss_time_next.item() / the_number_of_events
        time_loss_time_next = time_loss_time_next.item() / the_number_of_events
        fact = score.sum().item() / the_number_of_events
        events_loss_time_next = events_loss_time_next.item() / the_number_of_events
        constant_time_next = torch.linalg.norm(constant_time_next).item() / the_number_of_events

        # Loss values and other metrics at pred_time
        loss_pred_time = loss_pred_time.item() / the_number_of_events
        time_loss_pred_time = time_loss_pred_time.item() / the_number_of_events
        events_loss_pred_time = events_loss_pred_time.item() / the_number_of_events
        constant_time_next = torch.linalg.norm(constant_time_next).item() / the_number_of_events
        mae = mae.sum().item() / the_number_of_events
        constant_pred_time = torch.linalg.norm(constant_pred_time).item() / the_number_of_events


        return loss_time_next, time_loss_time_next, fact, events_loss_time_next, \
               constant_time_next, loss_pred_time, time_loss_pred_time, \
               events_loss_pred_time, constant_pred_time, mae, f1


    def postprocess(input, procedure):
        if procedure == 'Training':
            return [input[0], input[1], input[1] - input[2], input[3], input[4]]
        else:
            return [input[0], input[1], input[1] - input[2], input[3], \
                    input[4], input[5], input[6], input[7], input[8], \
                    input[9], input[10]]


    format_dict_length = 11


    def log_print_format(input, procedure):
        def format_training(input):
            format_dict = {}
            format_dict['loss'] = input[0]
            format_dict['absolute_time_loss'] = input[1]
            format_dict['relative_time_loss'] = input[2]
            format_dict['events_loss'] = input[3]
            format_dict['constant_norm'] = input[4]
            format_dict['num_format'] = {'loss': ':8.5f', 'absolute_time_loss': ':8.5f', 'relative_time_loss': ':8.5f', \
                                         'events_loss': ':8.5f', 'constant_norm': ':8.5f'}
            return format_dict

        def format_eva_and_test(input):
            format_dict = {}
            '''
            loss_time_next, time_loss_time_next, fact, events_loss_time_next,
            constant_time_next, loss_pred_time, time_loss_pred_time,
            events_loss_pred_time, constant_pred_time, mae, f1
            '''
            format_dict['loss_time_next'] = input[0]
            format_dict['absolute_time_loss_time_next'] = input[1]
            format_dict['relative_time_loss_time_next'] = input[2]
            format_dict['events_loss_time_next'] = input[3]
            format_dict['constant_norm_time_next'] = input[4]
            format_dict['loss_pred_time'] = input[5]
            format_dict['absolute_time_loss_pred_time'] = input[6]
            format_dict['events_loss_pred_time'] = input[7]
            format_dict['constant_norm_pred_time'] = input[8]
            format_dict['mae'] = input[9]
            format_dict['f1'] = input[10]

            format_dict['num_format'] = {
                'loss_time_next': ':8.5f', 'absolute_time_loss_time_next': ':8.5f', 'relative_time_loss_time_next': ':8.5f', 
                'events_loss_time_next': ':8.5f', 'constant_norm_time_next': ':8.5f', 'loss_pred_time': ':8.5f',
                'absolute_time_loss_pred_time': ':8.5f', 'events_loss_pred_time': ':8.5f', 'constant_norm_pred_time': ':8.5f', 
                'mae': ':2.8f', 'f1': ':2.8f'
            }
            return format_dict

        return format_training(input) if procedure == 'Training' else format_eva_and_test(input)
    
    
    logfile_format = {'step': '', 'absolute loss': ':8.5f', 'relative loss': ':8.5f', 'events_loss': ':8.5f', 'constant_norm': ':8.5f'}


    def logfile_print_format(input):
        format_dict = {}
        format_dict['absolute loss'] = input[0]
        format_dict['events_loss'] = input[1]
        format_dict['relative loss'] = input[2]
        format_dict['constant_norm'] = input[3]
        return format_dict
    

    def choose_metric(evaluation_report, test_report):
        '''
        [relative loss on evaluation dataset, relative loss on test dataset]
        '''
        return [evaluation_report[0], test_report[0]]
    
    metric_number = 2 # metric number is the length of the output of choose_metric