import torch
import torch.nn.functional as F
from einops import rearrange, repeat, reduce, pack
from sklearn.metrics import f1_score, top_k_accuracy_score, accuracy_score
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

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
        mu, eta, gamma: shape: [batch_size, seq_len, (resolution), d_hidden]
        dutation_t:     shape: [batch_size, seq_len, (resolution)]
        '''
        duration_t = rearrange(duration_t, '... -> ... 1')                     # [batch_size, seq_len, (resolution), 1]
        cell_t = torch.tanh(mu + (eta - mu) * torch.exp(-gamma * duration_t))  # [batch_size, seq_len, (resolution), d_input]
        return cell_t
    
    def mean_absolute_error_and_f1(self, events_history, time_history, events_next, time_next, mask_history, mask_next, mean, var):
        history = self.history_encoder(time_history, events_history, mask_history)
                                                                               # [batch_size, seq_len, d_input]
        eta = self.start_layer(history)                                        # [batch_size, seq_len, d_input]
        mu = self.converge_layer(history)                                      # [batch_size, seq_len, d_input]
        gamma = self.decay_layer(history)                                      # [batch_size, seq_len, d_input]

        mae_mean_of_all_event, pred_time = \
                    self.mean_absolute_error(eta = eta, mu = mu, gamma = gamma,
                                             time_history = time_history, time_next = time_next, 
                                             mask = mask_next, output_pred = True, sum = False)

        # temporal point process loss
        _, _, events_prediction_probability = self.log_likelihood(
             eta = eta, mu = mu, gamma = gamma, time = pred_time, \
             events = events_next, mask = mask_next
        )

        predicted_events = torch.argmax(events_prediction_probability, dim = -1)[mask_next == 1].detach().cpu().numpy()
        events_true = events_next[mask_next == 1].detach().cpu().numpy()
        f1 = f1_score(y_pred = predicted_events, y_true = events_true, average = 'macro')
        '''
        Event loss. This loss should not be counted into the backward loss
        '''

        return mae_mean_of_all_event, f1

    def mean_absolute_error(self, eta, mu, gamma, time_history, time_next, mask, sum = True, output_pred = False):
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
        reshaped_events_prediction_probability = rearrange(events_prediction_probability, 'b s ne -> b ne s')
                                                                               # [batch_size, num_events, seq_len]
        events_loss = F.cross_entropy(input = reshaped_events_prediction_probability, target = events.long(), reduction = 'none')
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
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]
        
        if mean == 0 and var == 1:
            '''
            This dataset does not apply normalisation, so we need to calculate the mean and variance here.
            '''
            mean = input_time.mean()
            var = input_time.var()
        
        # Use a relatively large number as the positive infinity.
        max_ = min(1e6, mean + 10 * var)
        time_inf = torch.ones_like(time_next) * max_                           # [batch_size, seq_len]

        resolution = min(int(max_ * 100), 50000)

        memory_ceiling = 1e9
        _, seq_len = events_next.shape
        if seq_len * resolution * self.num_events > memory_ceiling:
            resolution = int(memory_ceiling // (seq_len * self.num_events))

        history = self.history_encoder(time_history, events_history, mask_history)
                                                                               # [batch_size, seq_len, d_input]
        eta = self.start_layer(history)                                        # [batch_size, seq_len, d_input]
        mu = self.converge_layer(history)                                      # [batch_size, seq_len, d_input]
        gamma = self.decay_layer(history)                                      # [batch_size, seq_len, d_input]

        eta = rearrange(eta, 'b s di -> b s 1 di')                             # [batch_size, seq_len, 1, d_input]
        mu = rearrange(mu, 'b s di -> b s 1 di')                               # [batch_size, seq_len, 1, d_input]
        gamma = rearrange(gamma, 'b s di -> b s 1 di')                         # [batch_size, seq_len, 1, d_input]

        time_multiplier = torch.linspace(0, 1, resolution, device = self.device)
        expanded_time_inf = rearrange(time_inf, '... -> ... 1') * time_multiplier
                                                                               # [batch_size, seq_len, resolution]
        zero_gap = torch.zeros_like(expanded_time_inf[:, :, 0], device = self.device)
                                                                               # [batch_size, seq_len]
        expanded_time_gap_inf = torch.diff(expanded_time_inf, dim = -1)        # [batch_size, seq_len, resolution - 1]
        expanded_time_gap_inf, expanded_time_gap_inf_ps = pack(
            (zero_gap, expanded_time_gap_inf), 'b s *'
        )                                                                      # [batch_size, seq_len, resolution]

        hidden_state_inf = self.state_decay(mu, eta, gamma, expanded_time_inf) # [batch_size, seq_len, resolution, d_input]
        expanded_intensity_all_events_inf = self.intensity_layer(hidden_state_inf)
                                                                               # [batch_size, seq_len, resolution, num_events]
        expanded_integral_all_events_inf = expanded_intensity_all_events_inf * rearrange(expanded_time_gap_inf, '... -> ... 1')
                                                                               # [batch_size, seq_len, resolution, num_events]
        expanded_integral_all_events_inf = expanded_integral_all_events_inf.cumsum(dim = -2)
                                                                               # [batch_size, seq_len, resolution, num_events]

        expanded_integral_inf = reduce(expanded_integral_all_events_inf, 'b s r ne -> b s r ()', 'sum')
                                                                               # [batch_size, seq_len, resolution, 1]
        
        expanded_probability_inf = expanded_intensity_all_events_inf * torch.exp(-expanded_integral_inf)
                                                                               # [batch_size, seq_len, resolution, num_events]
        expanded_probability_inf = expanded_probability_inf[:, :, :-1, :] * expanded_time_gap_inf.unsqueeze(dim = -1)[:, :, 1:, :]
                                                                               # [batch_size, seq_len, resolution, num_events]
        
        probability = expanded_probability_inf[:, :, 1:, :].sum(dim = -2)      # [batch_size, seq_len, num_events]
        probability_integral_sum = probability.sum(dim = -1)                   # [batch_size, seq_len]
        predicted_events = torch.argmax(probability, dim = -1)                 # [batch_size, seq_len]

        # F1 value and top_k_acc are only avaliable when batch_size = 1
        
        f1 = []
        top_k_acc = []
        for (ground_truth_per_seq, probability_integral_per_seq) in zip(events_next, probability):
            f1.append(f1_score(y_true = ground_truth_per_seq.detach().cpu(),
                               y_pred = torch.argmax(probability_integral_per_seq, dim = -1).detach().cpu(), average = 'macro'))
            
            # Only available when batch_size = 1
            top_k_acc_per_seq = []
            if not fast:
                if self.num_events > 2:
                    for k in range(1, self.num_events + 1):
                        top_k_acc_per_seq.append(
                            top_k_accuracy_score(y_true = ground_truth_per_seq.detach().cpu(),
                                                 y_score = probability_integral_per_seq.detach().cpu(),
                                                 k = k,
                                                 labels = np.arange(self.num_events))
                        )
                else:
                    top_k_acc_per_seq.append(
                        accuracy_score(
                            y_true = ground_truth_per_seq.detach().cpu(),
                            y_pred = probability_integral_per_seq.detach().cpu()
                        )
                    )
                    top_k_acc.append(1.0)
            top_k_acc.append(top_k_acc_per_seq)

        # F1:        [batch_size]
        # top_k_acc: [batch_size, num_events]

        if mean == 0:
            resolution = max(min(int(input_time.mean().item() * 200), 1000), 1)
        else:
            resolution = max(min(int(mean * 200), 1000), 1)
        
        # tau_pred_all_event = self.prediction_with_all_event_types(history, events_next,
        #                                                           time_history, time_next,
        #                                                           probability, resolution, mask, mean, var, max_)
        #                                                                        # [batch_size, seq_len, num_events]
        tau_pred_all_event = torch.ones((1, seq_len, 97))                      # [batch_size, seq_len, num_events]

        mae_per_event_pure_predict = self.mean_absolute_error_per_event_worker(history, predicted_events, time_history, time_next,
                                                                               probability, resolution, mask_next, max_)
        mae_per_event = self.mean_absolute_error_per_event_worker(history, events_next, time_history, time_next, 
                                                                  probability, resolution, mask_next, max_)

        mae_per_event_pure_predict_avg = torch.sum(mae_per_event_pure_predict, dim = -1) / mask_next.sum(dim = -1)
        mae_per_event_avg = torch.sum(mae_per_event, dim = -1) / mask_next.sum(dim = -1)
        
        return f1, top_k_acc, probability_integral_sum, tau_pred_all_event, (mae_per_event_pure_predict_avg, mae_per_event_avg), \
               (mae_per_event_pure_predict, mae_per_event)

    def evaluate_all_event(self, history, tau, resolution):
        # Intensity and integral estimation
        eta = self.start_layer(history)                                        # [batch_size, seq_len, d_input]
        mu = self.converge_layer(history)                                      # [batch_size, seq_len, d_input]
        gamma = self.decay_layer(history)                                      # [batch_size, seq_len, d_input]

        eta = rearrange(eta, 'b s di -> b s 1 1 di')                           # [batch_size, seq_len, 1, 1, d_input]
        mu = rearrange(mu, 'b s di -> b s 1 1 di')                             # [batch_size, seq_len, 1, 1, d_input]
        gamma = rearrange(gamma, 'b s di -> b s 1 1 di')                       # [batch_size, seq_len, 1, 1, d_input]

        time_multiplier = torch.linspace(0, 1, resolution, device = self.device)
        expanded_time = rearrange(tau, '... ne -> ... 1 ne') * \
                        rearrange(time_multiplier, 'r -> 1 1 r 1')             # [batch_size, seq_len, resolution, num_events]

        expanded_time_gap = torch.diff(expanded_time, dim = -2)                # [batch_size, seq_len, resolution - 1, num_events]
        zero_gap = torch.zeros_like(expanded_time_gap[:, :, 0, :])             # [batch_size, seq_len, num_events]
        expanded_time_gap, expanded_time_gap_ps = pack(
            (zero_gap, expanded_time_gap), 'b s * ne'
        )                                                                      # [batch_size, seq_len, resolution, num_events]
    
        hidden_state = self.state_decay(mu, eta, gamma, expanded_time)         # [batch_size, seq_len, resolution, num_events, d_input]
        expanded_intensity_all_events = self.intensity_layer(hidden_state)     # [batch_size, seq_len, resolution, num_events, num_events]
        
        intensity_mask = F.one_hot(torch.arange(self.num_events, device = self.device), num_classes = self.num_events)
                                                                               # [num_events, num_events]
        intensity_mask = rearrange(intensity_mask, ' ne ne1 -> 1 1 1 ne ne1')  # [batch_size, seq_len, resolution, num_events, num_events]
        expanded_integral_all_events = expanded_intensity_all_events * rearrange(expanded_time_gap, 'b s r ne -> b s r ne 1')
                                                                               # [batch_size, seq_len, resolution, num_events, num_events]
        expanded_integral_all_events_sum = reduce(expanded_integral_all_events, '... ne -> ...', 'sum')
                                                                               # [batch_size, seq_len, resolution, num_events]
        expanded_intensity_all_events = reduce(expanded_intensity_all_events * intensity_mask, 'b s r ne ne1 -> b s r ne', 'sum')
                                                                               # [batch_size, seq_len, resolution, num_events]
        probabilty_expanded_events = torch.exp(-expanded_integral_all_events_sum) * expanded_intensity_all_events
                                                                               # [batch_size, seq_len, resolution, num_events]
        probability = expanded_time_gap[:, :, 1:, :] * probabilty_expanded_events[:, :, :-1, :]
                                                                               # [batch_size, seq_len, resolution - 1, num_events]
        probability = probability.sum(dim = -2)                                # [batch_size, seq_len, num_events]

        return probability

    def prediction_with_all_event_types(self, history, events_next, time_history, 
                                        time_next, p_x, resolution, mask, mean, var, max_val):
        '''
        The input should be the original minibatch
        MAE evaluation part, dwg and fullynn exclusive

        '''
        def bisect_target(history, taus):
            p_xt = self.evaluate_all_event(history, taus, resolution)
                                                                               # [batch_size, seq_len, num_events]
            p_t_x = p_xt / p_x                                                 # [batch_size, seq_len, num_events]
            p_gap = p_t_x - 1 / self.mae_threshold                             # [batch_size, seq_len, num_events]

            return p_gap
            
        def median_prediction(history, l, r):
            for _ in range(50):
                c = (l + r)/2
                v = bisect_target(history, c)
                l = torch.where(v < 0, c, l)
                r = torch.where(v >= 0, c, r)

            return (l + r)/2
        
        l = 0.0001*torch.ones((*time_history.shape, self.num_events), dtype = torch.float32, device = self.device)
                                                                               # [batch_size, seq_len, num_events]
        r = max_val*torch.ones((*time_history.shape, self.num_events), dtype = torch.float32, device = self.device)
                                                                               # [batch_size, seq_len, num_events]
        tau_pred = median_prediction(history, l, r)                            # [batch_size, seq_len, num_events]

        return tau_pred

    def evaluate_per_event(self, history, events_mask, time_next, tau, resolution):
        # Intensity and integral estimation

        eta = self.start_layer(history)                                        # [batch_size, seq_len, d_input]
        mu = self.converge_layer(history)                                      # [batch_size, seq_len, d_input]
        gamma = self.decay_layer(history)                                      # [batch_size, seq_len, d_input]

        eta = rearrange(eta, 'b s di -> b s 1 di')                             # [batch_size, seq_len, 1, d_input]
        mu = rearrange(mu, 'b s di -> b s 1 di')                               # [batch_size, seq_len, 1, d_input]
        gamma = rearrange(gamma, 'b s di -> b s 1 di')                         # [batch_size, seq_len, 1, d_input]

        time_multiplier = torch.linspace(0, 1, resolution, device = self.device)
        expanded_time = rearrange(tau, '... -> ... 1') * time_multiplier       # [batch_size, seq_len, resolution]

        expanded_time_gap = torch.diff(expanded_time, dim = -1)                # [batch_size, seq_len, resolution - 1]
        zero_gap = torch.zeros_like(expanded_time_gap[:, :, 0])                # [batch_size, seq_len]
        expanded_time_gap, expanded_time_gap_ps = pack(
            (zero_gap, expanded_time_gap), 'b s *'
        )                                                                      # [batch_size, seq_len, resolution]

        hidden_state = self.state_decay(mu, eta, gamma, expanded_time)         # [batch_size, seq_len, resolution, d_input]
        expanded_intensity_all_events = self.intensity_layer(hidden_state)     # [batch_size, seq_len, resolution, num_events]
        intensity_sum_across_events = torch.sum(expanded_intensity_all_events, dim = -1)
                                                                               # [batch_size, seq_len, resolution]
        integral_sum_across_events = torch.cumsum(intensity_sum_across_events * expanded_time_gap, dim = -1)
                                                                               # [batch_size, seq_len, resolution]
        probabilty_expanded_events = torch.exp(-integral_sum_across_events.unsqueeze(dim = -1)) * expanded_intensity_all_events
                                                                               # [batch_size, seq_len, resolution, num_events]
        probability = expanded_time_gap.unsqueeze(dim = -1)[:, :, 1:, :] * probabilty_expanded_events[:, :, :-1, :]
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
        def bisect_target(history, taus):
            events_next_one_hot = torch.nn.functional.one_hot(events_next.long(), num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]
            p_xt = self.evaluate_per_event(history, events_next_one_hot , time_next, taus, resolution)
                                                                               # [batch_size, seq_len]
            p_x = torch.sum(probability_integral * events_next_one_hot, dim = -1)
                                                                               # [batch_size, seq_len]
            p_t_x = p_xt / p_x                                                 # [batch_size, seq_len]
            p_gap = p_t_x - 1 / self.mae_threshold                             # [batch_size, seq_len]

            return p_gap
            
        def median_prediction(history, l, r):
            for _ in range(50):
                c = (l + r)/2
                v = bisect_target(history, c)
                l = torch.where(v < 0, c, l)
                r = torch.where(v >= 0, c, r)

            return (l + r)/2
        
        l = 0.0001*torch.ones_like(time_history, dtype = torch.float32)        # [batch_size, seq_len]
        r = max_val*torch.ones_like(time_history, dtype = torch.float32)       # [batch_size, seq_len]
        tau_pred = median_prediction(history, l, r)
        gap = (tau_pred - time_next) * mask
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

        zero_gap = torch.zeros_like(expanded_time[:, :, 0], device = self.device)
                                                                               # [batch_size, seq_len]
        expanded_time_gap = torch.diff(expanded_time, dim = -1)                # [batch_size, seq_len, resolution - 1]
        expanded_time_gap, expanded_time_gap_ps = pack(
            (zero_gap, expanded_time_gap), 'b s *'
        )                                                                      # [batch_size, seq_len, resolution]
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
        mean, var = input_data[1]

        time_history, time_next = self.divide_history_and_next(time)           # [batch_size, seq_len] * 2
        events_history, events_next = self.divide_history_and_next(events)     # [batch_size, seq_len] * 2
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]

        history = self.history_encoder(time_history, events_history, mask_history)
                                                                               # [batch_size, seq_len, d_input]
        eta = self.start_layer(history)                                        # [batch_size, seq_len, d_input]
        mu = self.converge_layer(history)                                      # [batch_size, seq_len, d_input]
        gamma = self.decay_layer(history)                                      # [batch_size, seq_len, d_input]

        eta_ = rearrange(eta, 'b s di -> b s 1 di')                            # [batch_size, seq_len, 1, d_input]
        mu_ = rearrange(mu, 'b s di -> b s 1 di')                              # [batch_size, seq_len, 1, d_input]
        gamma_ = rearrange(gamma, 'b s di -> b s 1 di')                        # [batch_size, seq_len, 1, d_input]

        time_multiplier = torch.linspace(0, 1, resolution, device = self.device)
        expanded_time = rearrange(time_next, '... -> ... 1') * time_multiplier # [batch_size, seq_len, resolution]
        zero_gap = torch.zeros_like(expanded_time[:, :, 0], device = self.device)
                                                                               # [batch_size, seq_len]
        expanded_time_gap = torch.diff(expanded_time, dim = -1)                # [batch_size, seq_len, resolution - 1]
        expanded_time_gap, expanded_time_gap_ps = pack(
            (zero_gap, expanded_time_gap), 'b s *'
        )                                                                      # [batch_size, seq_len, resolution]

        hidden_state = self.state_decay(mu_, eta_, gamma_, expanded_time)      # [batch_size, seq_len, resolution, d_input]
        expanded_intensity_all_events = self.intensity_layer(hidden_state)     # [batch_size, seq_len, resolution, num_events]

        expanded_integral_all_events = expanded_intensity_all_events * rearrange(expanded_time_gap, '... -> ... 1')
                                                                               # [batch_size, seq_len, resolution, num_events]
        expanded_integral_all_events = expanded_integral_all_events.cumsum(dim = -2)
                                                                               # [batch_size, seq_len, resolution, num_events]
        
        intensity_and_integral_plot = {}
        expand_intensity = rearrange(expanded_intensity_all_events, 'b s r ne -> b (s r) ne')
                                                                               # [batch_size, seq_len * resolution, num_events]
        expand_integral = rearrange(expanded_integral_all_events, 'b s r ne -> b (s r) ne')
                                                                               # [batch_size, seq_len * resolution, num_events]
        expanded_intensity = torch.chunk(expand_intensity, chunks = self.num_events, dim = -1)
                                                                               # [batch_size, seq_len * resolution] * num_events
        expanded_integral = torch.chunk(expand_integral, chunks = self.num_events, dim = -1)
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

        mae = self.mean_absolute_error(eta, mu, gamma, time_history, time_next, mask_next, sum = False)
                                                                               # [batch_size, seq_len]
        f1, top_k, probability_sum, tau_pred_all_event, maes_avg, maes \
                = self.mean_absolute_error_per_event(
                    input_time = time, input_events = events, mask = mask, mean = mean, var = var, fast = False
                )
        mae_per_event_pure_predict_avg, mae_per_event_avg = maes_avg
        mae_per_event_pure_predict, mae_per_event = maes

        mae = mae.detach().cpu().numpy()                                       # [batch_size, seq_len]
        tau_pred_all_event = tau_pred_all_event.detach().cpu().numpy()         # [batch_size, seq_len, num_events]
        probability_sum = probability_sum.detach().cpu().numpy()               # [batch_size, seq_len]
        mae_per_event_pure_predict_avg = mae_per_event_pure_predict_avg.detach().cpu().numpy()
                                                                               # [batch_size]
        mae_per_event_avg = mae_per_event_avg.detach().cpu().numpy()           # [batch_size]
        mae_per_event_pure_predict = mae_per_event_pure_predict.detach().cpu().numpy()
                                                                               # [batch_size, seq_len]
        mae_per_event = mae_per_event.detach().cpu().numpy()                   # [batch_size, seq_len]
        expand_integral = expand_integral.detach().cpu().numpy()               # [batch_size, seq_len * resolution, num_events]
        expand_intensity = expand_intensity.detach().cpu().numpy()             # [batch_size, seq_len * resolution, num_events]

        packed_values = zip(f1, top_k, probability_sum, tau_pred_all_event, mae, mae_per_event_pure_predict,\
                            mae_per_event_pure_predict_avg, mae_per_event, mae_per_event_avg, \
                            expand_intensity, expand_integral, time_next, mask_next)

        for idx, (f1_per_seq, top_k_per_seq, probability_sum_per_seq, tau_pred_all_event_per_seq, mae_per_seq,
                  mae_per_event_pure_predict_per_seq, mae_per_event_pure_predict_avg_per_seq,
                  mae_per_event_per_seq, mae_per_event_avg_per_seq,
                  expand_intensity_per_seq, expand_integral_per_seq, time_next_per_seq, mask_per_seq) \
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
                'marks': ['MAE_k against prediction'] * seq_len +  ['MAE_k against real events'] * seq_len + ['MAE'] * seq_len
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

            # Heatmap
            heatmap_data = {}
            expand_probability_per_seq = expand_intensity_per_seq * np.exp(-expand_integral_per_seq)
                                                                               # [(seq_len * resolution), num_events]
            # rho: spearman coefficient
            heatmap_data['spearman'] = spearmanr(expand_probability_per_seq[:seq_len * resolution])[0]
            if self.num_events == 2:
                heatmap_data['spearman'] = np.array([[1, heatmap_data['spearman']], [heatmap_data['spearman'], 1]])

            # r: pearson coefficient
            heatmap_data['pearson'] = np.corrcoef(expand_probability_per_seq[:seq_len * resolution], rowvar = False)
            # L^1 metric
            heatmap_data['L1'] = L1_distance(expand_probability_per_seq[:seq_len * resolution],
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

            accumulated_probability_distribution_per_seq = \
                expand_intensity_per_seq.sum(axis = -1) * np.exp(-expand_integral_per_seq.sum(axis = -1))
                                                                               # [seq_len * resolution]
            accumulated_probability_distribution_reshaped_per_seq = \
                rearrange(accumulated_probability_distribution_per_seq, '(s r) -> s r', r = resolution)
                                                                               # [seq_len * resolution]
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
                'distribution_of_probability_values',
                {
                    'data': df_probability,
                    "kind": "kde",
                    'height': 4,
                    'aspect': 0.7
                }
            ],
            [
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
            ]

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
        return [evaluation_report[0], test_report[0]]
    
    metric_number = 2 # metric number is the length of the output of choose_metric

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