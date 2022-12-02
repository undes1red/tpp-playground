import torch
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
from sklearn.metrics import f1_score, top_k_accuracy_score, accuracy_score
import numpy as np

from .transformers import TransformerEncoder
from ..utils import BasicModule
from .utils import *

class SAHP(BasicModule):
    def __init__(self, num_events, device, d_input = 64, d_rnn = 64, d_hidden = 256, n_layers = 3,
                 n_head = 3, d_qk = 64, d_v = 64, dropout = 0.1, mae_threshold = 2):
        super(SAHP, self).__init__()
        self.device = device
        self.num_events = num_events if num_events > 0 else 1
        self.mae_threshold = mae_threshold
        self.gelu = nn.GELU()

        # The original paper makes people believe SAHP is a RMTPP-like model.
        # However, this model in fact decays the hidden embedding so it is akin to CTLSTM.
        # The following three layers find the \eta_{u, i+1}, \mu_{u, i+1}, and \gamma_{u i+1}
        self.start_layer = nn.Sequential(
            nn.Linear(d_input, d_input, bias = True, device = self.device),
            self.gelu
        )

        self.converge_layer = nn.Sequential(
            nn.Linear(d_input, d_input, bias = True, device = self.device),
            self.gelu
        )

        self.decay_layer = nn.Sequential(
            nn.Linear(d_input, d_input, bias = True, device = self.device)
            ,nn.Softplus(beta = 10.0)
        )

        # This layer translates decayed hidden states into intensity function values.
        self.intensity_layer = nn.Sequential(
            nn.Linear(d_input, self.num_events, bias = True, device = self.device)
            ,nn.Softplus(beta = 1.)
        )

        # History encoder. SAHP employs a plain transformer to encode marked temporal history
        self.history_encoder = TransformerEncoder(num_events, device = self.device, \
                                                  d_input = d_input, d_rnn = d_rnn, \
                                                  d_hidden = d_hidden, n_layers = n_layers, \
                                                  n_head = n_head, d_qk = d_qk, d_v = d_v, \
                                                  dropout = dropout)
    
    '''
    Functions for model propagation and evaluation
    '''
    def forward(self, time, events, mask, evaluate = False):
        '''
        Check if events data is present.
        Now, we assume that no event data is available.
        Args:
        1. time: the sequence containing events' timestamps. shape: [batch_size, seq_len + 1]
        2. events: the sequence containing information about events. shape: [batch_size, seq_len + 1]
        3. mask: filter out the padding events in the event batches. shape: [batch_size, seq_len + 1]
        '''

        time_history, time_next = self.divide_history_and_next(time)           # [batch_size, seq_len] * 2
        events_history, events_next = self.divide_history_and_next(events)     # [batch_size, seq_len] * 2
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]

        history = self.history_encoder(time_history, events_history, mask_history)
                                                                               # [batch_size, seq_len, d_input]
        eta = self.start_layer(history)                                        # [batch_size, seq_len, d_input]
        mu = self.converge_layer(history)                                      # [batch_size, seq_len, d_input]
        gamma = self.decay_layer(history)                                      # [batch_size, seq_len, d_input]

        # temporal point process loss
        log_likeli_loss, marker_loss, events_prediction_probability = self.log_likelihood(
             eta = eta, mu = mu, gamma = gamma, time = time_next, \
             events = events_next, mask = mask_next
        )

        mae_mean_of_all_event = 0
        f1 = 0
        if evaluate:
            mae_mean_of_all_event = \
                        self.mean_absolute_error(eta = eta, mu = mu, gamma = gamma,
                                                 time_history = time_history, time_next = time_next, 
                                                 mask = mask_next)
            predicted_events = torch.argmax(events_prediction_probability, dim = -1)[mask_next == 1].detach().cpu().numpy()
            events_true = events_next[mask_next == 1].detach().cpu().numpy()
            f1 = f1_score(y_pred = predicted_events, y_true = events_true, average = 'macro')
        '''
        Event loss. This loss should not be counted into the backward loss
        '''

        the_number_of_events = mask_next.sum()

        return log_likeli_loss, marker_loss, f1, mae_mean_of_all_event, the_number_of_events

    def state_decay(self, mu, eta, gamma, duration_t):
        '''
        mu, eta, gamma: shape: [batch_size, seq_len, d_hidden, (resolution)]
        dutation_t:     shape: [batch_size, seq_len, (resolution)]
        '''
        duration_t = rearrange(duration_t, '... -> ... 1')                     # [batch_size, seq_len, (resolution), 1]
        cell_t = torch.tanh(mu + (eta - mu) * torch.exp(-gamma * duration_t))  # [batch_size, seq_len, (resolution), d_input]
        return cell_t

    def mean_absolute_error(self, eta, mu, gamma, time_history, time_next, mask):
        '''
        The input should be the original minibatch
        MAE evaluation part, dwg and fullynn exclusive

        Update: 2022-09-23
        Add event-wise MAE support.
        '''
        def bisect_target(taus):
            return self.evaluate(eta, mu, gamma, taus, mask) - \
                   torch.log(torch.tensor(self.mae_threshold, device = self.device))
            
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
        gap = (tau_pred - time_next) * mask
        mae_mean_of_all_events = torch.sum(torch.abs(gap)) / mask.sum()

        return mae_mean_of_all_events.item()

    def evaluate(self, eta, mu, gamma, time, mask):
        '''
        Args:
        1. time: the sequence containing events' timestamps. shape: [batch_size, seq_len + 1]
        2. events: the sequence containing information about events. shape: [batch_size, seq_len + 1]
        3. mask: the padding mask introduced by the dataloader. shape: [batch_size, seq_len + 1]
        '''
        intensity_integral = self.compute_integral_unbiased(eta, mu, gamma, time, mask)
                                                                               # [batch_size, seq_len]
        return intensity_integral

    def divide_history_and_next(self, input):
        input_history, input_next = input[:, :-1].clone(), input[:, 1:].clone()
        return input_history, input_next

    '''
    Loss functions
    '''
    def log_likelihood(self, eta, mu, gamma, time, events, mask):
        """ Log-likelihood of sequence. """
            
        if events is not None:
            type_mask = F.one_hot(events.long(), num_classes = self.num_events)# [batch_size, seq_len, num_events]
        else:
            type_mask_shape = (*eta.shape[:2], self.num_events)
            type_mask = torch.ones(type_mask_shape, device = self.device)      # [batch_size, seq_len, num_events]

        '''
        MTPP loss function
        '''
        hidden_state_at_t = self.state_decay(mu = mu, eta = eta, gamma = gamma, duration_t = time)
                                                                               # [batch_size, seq_len, d_input]
        intensity_all_events = self.intensity_layer(hidden_state_at_t)         # [batch_size, seq_len, num_events]
        intensity = reduce(intensity_all_events * type_mask, '... ne -> ...', 'sum')
                                                                               # [batch_size, seq_len]

        # event log-likelihood
        log_intensity = compute_event(intensity, mask)                         # [batch_size, seq_len]
    
        # non-event log-likelihood, either numerical integration or MC integration
        intensity_integral = self.compute_integral_unbiased(eta, mu, gamma, time, mask)
                                                                               # [batch_size, seq_len]
        ll = (-log_intensity + intensity_integral).clamp(max = 15)             # [batch_size, seq_len]
    
        mtpp_loss = torch.sum(ll)

        '''
        Event loss function. Only for evaluation, do not use this loss as a part of the training loss.
        '''
        events_prediction_probability = intensity_all_events / intensity_all_events.sum(dim = -1, keepdim = True)
                                                                               # [batch_size, seq_len, num_events]
        events_prediction_probability = rearrange(events_prediction_probability, 'b s ne -> b ne s')
                                                                               # [batch_size, num_events, seq_len]
        events_loss = F.cross_entropy(input = events_prediction_probability, target = events.long(), reduction = 'none')
                                                                               # [batch_size, seq_len]
        events_loss = (events_loss * mask).sum()

        return mtpp_loss, events_loss, events_prediction_probability
    
    def compute_integral_unbiased(self, eta, mu, gamma, time, non_pad_mask, resolution = 100):
        """ Log-likelihood of non-events, using Monte Carlo integration. """
    
        diff_time = time * non_pad_mask
        temp_time = rearrange(diff_time, '... -> ... 1') * \
                    torch.rand([*diff_time.size(), resolution], device = self.device)
                                                                               # [batch_size, seq_len, resolution]
        
        mu = rearrange(mu, 'b s d_i -> b s 1 d_i')                             # [batch_size, seq_len, 1, d_input]
        eta = rearrange(eta, 'b s d_i -> b s 1 d_i')                           # [batch_size, seq_len, 1, d_input]
        gamma = rearrange(gamma, 'b s d_i -> b s 1 d_i')                       # [batch_size, seq_len, 1, d_input]
        hidden_state_at_t = self.state_decay(mu = mu, eta = eta, gamma = gamma, duration_t = temp_time)
                                                                               # [batch_size, seq_len, resolution, d_input]
        intensity_all_events = self.intensity_layer(hidden_state_at_t)         # [batch_size, seq_len, resolution, num_events]
        intensity_all_events_mean = reduce(intensity_all_events, 'b s r ne -> b s ne', 'mean')
                                                                               # [batch_size, seq_len, num_events]
        integral_all_events = intensity_all_events_mean * rearrange(time, '... -> ... 1')
                                                                               # [batch_size, seq_len, num_events]
        integral = reduce(integral_all_events, 'b s ne -> b s', 'sum')         # [batch_size, seq_len]

        return integral

    def mean_absolute_error_per_event(self, input_time, input_events, mask, mean, var, fast):
        '''
        The precedure resembles the compute_integral_unbiased() but the output of small step MC takes would
        be recorded as part of the output.
        '''
        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        _, events_next = self.divide_history_and_next(input_events)            # [batch_size, seq_len]
        _, mask_next = self.divide_history_and_next(mask)                      # [batch_size, seq_len]
        
        if mean == 0 and var == 1:
            '''
            This dataset does not apply normalisation, so we need to calculate the mean and variance here.
            '''
            mean = input_time.mean()
            var = input_time.var()
        
        # Use a relatively large number as the positive infinity.
        max_ = min(1e5, mean + 10 * var)

        resolution = min(int(max_ * 100), 5000)

        # history information
        history = self.evaluate(input_time, input_events, mask)                # [batch_size, seq_len, d_input]
        
        # Intensity and integral estimation
        time_multiplier = torch.linspace(0, 1, resolution, device = self.device)
        expanded_time = torch.ones_like(time_next, device = self.device) * max_
                                                                               # [batch_size, seq_len]
        expanded_time = expanded_time.unsqueeze(dim = -1) * time_multiplier    # [batch_size, seq_len, resolution]
        expanded_time_gap = torch.diff(expanded_time, dim = -1).mean(dim = -1, keepdim = True)
                                                                               # [batch_size, seq_len, 1]
        aggregated_time = torch.cumsum(time_next, dim = -1).unsqueeze(dim = -1)# [batch_size, seq_len, 1]
        scaled_expanded_time = expanded_time / aggregated_time                 # [batch_size, seq_len, resolution]
        scaled_expanded_time = scaled_expanded_time.unsqueeze(dim = -1)        # [batch_size, seq_len, resolution, 1]

        intensity_for_each_event = self.model.linear(history).detach()         # [batch_size, seq_len, num_events]
        intensity_for_each_event = intensity_for_each_event.unsqueeze(dim = -2)# [batch_size, seq_len, 1, num_events]
        
        expanded_intensity = softplus_ext(self.alpha * scaled_expanded_time + intensity_for_each_event, F.softplus(self.beta))
                                                                               # [batch_size, seq_len, resolution, num_events]
        intensity_sum_across_events = torch.sum(expanded_intensity, dim = -1)  # [batch_size, seq_len, resolution]
        integral_sum_across_events = torch.cumsum(intensity_sum_across_events * expanded_time_gap, dim = -1)
                                                                               # [batch_size, seq_len, resolution]
        probabilty_expanded_events = integral_sum_across_events.unsqueeze(dim = -1) * expanded_intensity
                                                                               # [batch_size, seq_len, resolution, num_events]
        probability = expanded_time_gap.unsqueeze(dim = -1) * probabilty_expanded_events
                                                                               # [batch_size, seq_len, resolution, num_events]
        probability = probability.sum(dim = -2)                                # [batch_size, seq_len, num_events]
        probability_integral_sum = probability.sum(dim = -1)                   # [batch_size, seq_len]
        predicted_events = torch.argmax(probability, dim = -1)                 # [batch_size, seq_len]

        # F1 value and top_k_acc are only avaliable when batch_size = 1
        f1 = f1_score(y_true = events_next.squeeze().detach().cpu(),
                      y_pred = predicted_events.squeeze().detach().cpu(), average = 'macro')
        
        # Only available when batch_size = 1
        top_k_acc = []
        if not fast:
            if self.num_events > 2:
                for k in range(1, self.num_events + 1):
                    top_k_acc.append(
                        top_k_accuracy_score(y_true = events_next.squeeze().detach().cpu(),
                                             y_score = probability.reshape(-1, self.num_events).detach().cpu(),
                                             k = k,
                                             labels = np.arange(self.num_events))
                    )
            else:
                top_k_acc.append(
                    accuracy_score(
                        y_true = events_next.squeeze().detach().cpu(),
                        y_pred = predicted_events.squeeze().detach().cpu()
                    )
                )
                top_k_acc.append(1.0)

        if mean == 0:
            resolution = max(min(int(input_time.mean().item() * 200), 1000), 1)
        else:
            resolution = max(min(int(mean * 200), 1000), 1)
        
        mae_per_event_pure_predict = self.mean_absolute_error_per_event_worker(history, predicted_events, time_history, time_next,
                                                                               probability, resolution, mask_next, max_)
        mae_per_event = self.mean_absolute_error_per_event_worker(history, events_next, time_history, time_next, 
                                                                  probability, resolution, mask_next, max_)

        mae_per_event_pure_predict_avg = torch.sum(mae_per_event_pure_predict) / mask_next.sum()
        mae_per_event_avg = torch.sum(mae_per_event) / mask_next.sum()
        
        return f1, top_k_acc, probability_integral_sum, (mae_per_event_pure_predict_avg.item(), mae_per_event_avg.item()), \
               (mae_per_event_pure_predict, mae_per_event)

    def evaluate_per_event(self, history, events_mask, time_next, tau, resolution):
        # Intensity and integral estimation
        time_multiplier = torch.linspace(0, 1, resolution, device = self.device)
        expanded_time = tau * time_multiplier                                  # [batch_size, seq_len, resolution]
        expanded_time_gap = torch.diff(expanded_time, dim = -1).mean(dim = -1, keepdim = True)
                                                                               # [batch_size, seq_len, 1]
        aggregated_time = torch.cumsum(time_next, dim = -1).unsqueeze(dim = -1)# [batch_size, seq_len, 1]
        scaled_expanded_time = expanded_time / aggregated_time                 # [batch_size, seq_len, resolution]
        scaled_expanded_time = scaled_expanded_time.unsqueeze(dim = -1)        # [batch_size, seq_len, resolution, 1]

        intensity_for_each_event = self.model.linear(history).detach()         # [batch_size, seq_len, num_events]
        intensity_for_each_event = intensity_for_each_event.unsqueeze(dim = -2)# [batch_size, seq_len, 1, num_events]
        
        expanded_intensity = softplus_ext(self.alpha * scaled_expanded_time + intensity_for_each_event, F.softplus(self.beta))
                                                                               # [batch_size, seq_len, resolution, num_events]
        intensity_sum_across_events = torch.sum(expanded_intensity, dim = -1)  # [batch_size, seq_len, resolution]
        integral_sum_across_events = torch.cumsum(intensity_sum_across_events * expanded_time_gap, dim = -1)
                                                                               # [batch_size, seq_len, resolution]
        probabilty_expanded_events = integral_sum_across_events.unsqueeze(dim = -1) * expanded_intensity
                                                                               # [batch_size, seq_len, resolution, num_events]
        probability = expanded_time_gap.unsqueeze(dim = -1) * probabilty_expanded_events
                                                                               # [batch_size, seq_len, resolution, num_events]
        probability = probability.sum(dim = -2)                                # [batch_size, seq_len, num_events]
        probability = (probability * events_mask).sum(dim = -1)                # [batch_size, seq_len]

        return probability

    def mean_absolute_error_per_event_worker(self, history, events_next, 
        time_history, time_next, probability_integral, resolution, mask, max_val):
        '''
        The input should be the original minibatch
        MAE evaluation part, dwg and fullynn exclusive

        '''
        def bisect_target(events_history, taus):
            events_next_one_hot = torch.nn.functional.one_hot(events_next.long(), num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]
            p_xt = self.evaluate_per_event(events_history, events_next_one_hot , time_next, taus, resolution)
                                                                               # [batch_size, seq_len]
            p_x = torch.sum(probability_integral * events_next_one_hot, dim = -1)
                                                                               # [batch_size, seq_len]
            p_t_x = p_xt / p_x                                                 # [batch_size, seq_len]
            p_gap = p_t_x - (1 / self.mae_threshold)                           # [batch_size, seq_len]

            return p_gap.unsqueeze(dim = -1)
            
        def median_prediction(events_history, l, r):
            for _ in range(50):
                c = (l + r)/2
                v = bisect_target(events_history, c)
                l = torch.where(v < 0, c, l)
                r = torch.where(v >= 0, c, r)

            return (l + r)/2
        
        l = 0.0001*torch.ones_like(time_history, dtype = torch.float32).unsqueeze(dim = -1)
                                                                               # [batch_size, seq_len]
        r = max_val*torch.ones_like(time_history, dtype = torch.float32).unsqueeze(dim = -1)
                                                                               # [batch_size, seq_len]
        tau_pred = median_prediction(history, l, r)
        gap = (tau_pred - time_next).squeeze(-1) * mask
        gap = torch.abs(gap)

        return gap

    def function_prober(self, input_data, resolution):
        '''
        Probe the learned intensity function from the model.
        This task should be pretty easy for the explicit form of intensity functions.
        '''

        time, events, _, mask, _ = input_data[0]                               # 3 * [batch_size, seq_len + 1]
        time_history, time_next = self.divide_history_and_next(time)           # [batch_size, seq_len] * 2
        events_history, _ = self.divide_history_and_next(events)               # [batch_size, seq_len] * 2
        mask_history, _ = self.divide_history_and_next(mask)                   # [batch_size, seq_len]

        history = self.history_encoder(time_history, events_history, mask_history)
                                                                               # [batch_size, seq_len, d_input]
        eta = self.start_layer(history)                                        # [batch_size, seq_len, d_input]
        mu = self.converge_layer(history)                                      # [batch_size, seq_len, d_input]
        gamma = self.decay_layer(history)                                      # [batch_size, seq_len, d_input]

        eta = rearrange(eta, 'b s di -> b s 1 di')                             # [batch_size, seq_len, 1, d_input]
        mu = rearrange(mu, 'b s di -> b s 1 di')                               # [batch_size, seq_len, 1, d_input]
        gamma = rearrange(gamma, 'b s di -> b s 1 di')                         # [batch_size, seq_len, 1, d_input]

        time_multiplier = torch.linspace(0, 1, resolution, device = self.device)
        expanded_time = rearrange(time_next, '... -> ... 1') * time_multiplier # [batch_size, seq_len, resolution]
        expanded_time_gap = reduce(torch.diff(expanded_time, dim = -1), 'b s r -> b s ()', 'mean')
                                                                               # [batch_size, seq_len, 1]
        hidden_state = self.state_decay(mu, eta, gamma, expanded_time)         # [batch_size, seq_len, resolution, d_input]
        expanded_intensity_all_events = self.intensity_layer(hidden_state)     # [batch_size, seq_len, resolution, num_events]

        expanded_integral_all_events = expanded_intensity_all_events * rearrange(expanded_time_gap, '... -> ... 1')
                                                                               # [batch_size, seq_len, resolution, num_events]
        expanded_integral_all_events = expanded_integral_all_events.cumsum(dim = -2)
                                                                               # [batch_size, seq_len, resolution, num_events]
        
        expanded_intensity = reduce(expanded_intensity_all_events, 'b s r ne -> b (s r)', 'sum')
                                                                               # [batch_size, seq_len * resolution]
        expanded_integral = reduce(expanded_integral_all_events, 'b s r ne -> b (s r)', 'sum')
                                                                               # [batch_size, seq_len * resolution]
        # aggregated timestamp
        batch_size, seq_len = time_history.shape
        timestamp = torch.cat(
            (torch.zeros((batch_size, seq_len, 1), device = self.device), expanded_time.diff(dim = -1)),
            dim = -1)                                                          # [batch_size, seq_len, resolution]
        timestamp = timestamp.reshape(batch_size, seq_len * resolution)        # [batch_size, seq_len * resolution]
        
        return expanded_integral, expanded_intensity, timestamp
    
    def model_prober(self, input_data, resolution):
        '''
        Probe the learned intensity function from the model.
        This task should be pretty easy for the explicit form of intensity functions.
        '''

        time, events, _, mask, _ = input_data[0]                               # 3 * [batch_size, seq_len + 1]
        time_history, time_next = self.divide_history_and_next(time)           # [batch_size, seq_len] * 2
        events_history, _ = self.divide_history_and_next(events)               # [batch_size, seq_len] * 2
        mask_history, _ = self.divide_history_and_next(mask)                   # [batch_size, seq_len]

        history = self.history_encoder(time_history, events_history, mask_history)
                                                                               # [batch_size, seq_len, d_input]
        eta = self.start_layer(history)                                        # [batch_size, seq_len, d_input]
        mu = self.converge_layer(history)                                      # [batch_size, seq_len, d_input]
        gamma = self.decay_layer(history)                                      # [batch_size, seq_len, d_input]

        eta = rearrange(eta, 'b s di -> b s 1 di')                             # [batch_size, seq_len, 1, d_input]
        mu = rearrange(mu, 'b s di -> b s 1 di')                               # [batch_size, seq_len, 1, d_input]
        gamma = rearrange(gamma, 'b s di -> b s 1 di')                         # [batch_size, seq_len, 1, d_input]

        time_multiplier = torch.linspace(0, 1, resolution, device = self.device)
        expanded_time = rearrange(time_next, '... -> ... 1') * time_multiplier # [batch_size, seq_len, resolution]
        expanded_time_gap = reduce(torch.diff(expanded_time, dim = -1), 'b s r -> b s ()', 'mean')
                                                                               # [batch_size, seq_len, 1]
        hidden_state = self.state_decay(mu, eta, gamma, expanded_time)         # [batch_size, seq_len, resolution, d_input]
        expanded_intensity_all_events = self.intensity_layer(hidden_state)     # [batch_size, seq_len, resolution, num_events]

        expanded_integral_all_events = expanded_intensity_all_events * rearrange(expanded_time_gap, '... -> ... 1')
                                                                               # [batch_size, seq_len, resolution, num_events]
        expanded_integral_all_events = expanded_integral_all_events.cumsum(dim = -2)
                                                                               # [batch_size, seq_len, resolution, num_events]
        
        intensity_and_integral_plot = {}
        expanded_intensity = rearrange(expanded_intensity_all_events, 'b s r ne -> b (s r) ne')
                                                                               # [batch_size, seq_len * resolution, num_events]
        expanded_integral = rearrange(expanded_integral_all_events, 'b s r ne -> b (s r) ne')
                                                                               # [batch_size, seq_len * resolution, num_events]
        expanded_intensity = torch.chunk(expanded_intensity, chunks = self.num_events, dim = -1)
                                                                               # [batch_size, seq_len * resolution] * num_events
        expanded_integral = torch.chunk(expanded_integral, chunks = self.num_events, dim = -1)
                                                                               # [batch_size, seq_len * resolution] * num_events

        for idx, (intensity, integral) in enumerate(zip(expanded_intensity, expanded_integral)):
            intensity_and_integral_plot[f'event_intensity_{idx}'] = rearrange(intensity, '... 1 -> ...')
            intensity_and_integral_plot[f'event_integral_{idx}'] = rearrange(integral, '... 1 -> ...')

        # aggregated timestamp
        batch_size, seq_len = time_history.shape
        timestamp = torch.cat(
            (torch.zeros((batch_size, seq_len, 1), device = self.device), expanded_time.diff(dim = -1)),
            dim = -1)                                                          # [batch_size, seq_len, resolution]
        timestamp = timestamp.reshape(batch_size, seq_len * resolution)        # [batch_size, seq_len * resolution]
        
        # Additional plots
        additional_plot = []
        return (intensity_and_integral_plot, additional_plot), timestamp

    '''
    Static methods
    '''
    def train_step(model, minibatch, device):
        ''' Epoch operation in training phase'''
        model.train()

        '''
        Maybe need another function to extract data from minibatches.
        Currently, we don't acquire any prediction loss to assist the model training.  
        '''
        time, events, fact, mask = minibatch[0]                                 # 3 * [batch_size, seq_len + 1, 1] & [batch_size, seq_len, 1]
        tpp_loss, mark_loss, f1, mae, the_number_of_events = model(time, events, mask)
        loss = tpp_loss
        loss.backward()

        tpp_loss, mark_loss = tpp_loss.item(), mark_loss.item()
        fact = fact.sum()
    
        return tpp_loss / the_number_of_events , mark_loss / the_number_of_events, fact / the_number_of_events
    
    def evaluation_step(model, minibatch, device):
        ''' Epoch operation in evaluation phase '''
    
        model.eval()

        time, events, fact, mask = minibatch[0]                                 # 3 * [batch_size, seq_len + 1, 1] & [batch_size, seq_len, 1]
        tpp_loss, mark_loss, f1, mae, the_number_of_events = model(time, events, mask, evaluate = True)

        tpp_loss, mark_loss = tpp_loss.item(), mark_loss.item()
        fact = fact.sum()

        return tpp_loss / the_number_of_events, mark_loss / the_number_of_events, fact / the_number_of_events, \
               f1, mae

    def postprocess(input, procedure):
        def train():
            return [input[0], input[0] - input[2], input[1]]
        def evaluate():
            return [input[0], input[0] - input[2], input[1], input[3], input[4]]
        return train() if procedure == 'Training' else evaluate()

    def log_print_format(input, procedure):
        def train():
            format_dict = {}
            format_dict['absolute_loss'] = input[0]
            format_dict['relative_loss'] = input[1]
            format_dict['events_loss'] = input[2]
            format_dict['num_format'] = {'absolute_loss': ':8.5f', 'relative_loss': ':8.5f', 'events_loss': ':8.5f'}
            return format_dict
        def evaluate():
            format_dict = {}
            format_dict['absolute_loss'] = input[0]
            format_dict['relative_loss'] = input[1]
            format_dict['events_loss'] = input[2]
            format_dict['f1'] = input[3]
            format_dict['MAE'] = input[4]
            format_dict['num_format'] = {'absolute_loss': ':8.5f', 'relative_loss': ':8.5f', 'events_loss': ':8.5f', 'f1': ':8.5f', 'MAE': ':2.8f'}
            return format_dict    
        return train() if procedure == 'Training' else evaluate()

    format_dict_length = 5
    
    logfile_format = {'step': '', 'absolute loss': ':8.5f', 'relative loss': ':8.5f', 'events loss': ':8.5f'}

    def logfile_print_format(input):
        format_dict = {}
        format_dict['absolute loss'] = input[0]
        format_dict['relative loss'] = input[1]
        format_dict['events loss'] = input[2]
        return format_dict
    
    def choose_metric(evaluation_report, test_report):
        '''
        [relative loss on evaluation dataset, relative loss on test dataset]
        '''
        return [evaluation_report[1].item() + evaluation_report[-1], test_report[1].item()+ test_report[-1]]
    
    metric_number = 2 # metric number is the length of the output of choose_metric