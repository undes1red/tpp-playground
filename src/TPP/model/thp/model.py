import torch
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
from sklearn.metrics import f1_score, top_k_accuracy_score, accuracy_score
from scipy.stats import spearmanr
import numpy as np
import pandas as pd

from .transformers import TransformerTPP
from ..utils import BasicModule
from .utils import *

class THP(BasicModule):
    def __init__(self, num_events, device, d_input = 64, d_rnn = 64, d_hidden = 256, n_layers = 3,
                 n_head = 3, d_qk = 64, d_v = 64, dropout = 0.1, beta = 0, mae_threshold = 2):
        super(THP, self).__init__()
        self.device = device
        self.num_events = num_events if num_events > 0 else 1
        self.mae_threshold = mae_threshold

        # parameter for the weight of time difference
        self.alpha = nn.Parameter(torch.ones((self.num_events), dtype = torch.float32, \
                                  device = self.device, requires_grad = True))

        # parameter for the softplus function
        self.beta = nn.Parameter(torch.ones((self.num_events), dtype = torch.float32, \
                                  device = self.device, requires_grad = True) * beta)
        # self.beta =  beta

        self.model = TransformerTPP(num_events, device = self.device, d_input = d_input, d_rnn = d_rnn, d_hidden = d_hidden,\
                                    n_layers = n_layers, n_head = n_head, d_qk = d_qk, d_v = d_v, dropout = dropout)
    
    '''
    Functions for model propagation and evaluation
    '''
    def forward(self, time, events, mask):
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

        history = self.model(time_history, events_history, mask_history)       # [batch_size, seq_len, d_input]

        # temporal point process loss
        log_likeli_loss, marker_loss = self.log_likelihood(
             history = history, time = time_next, events = events_next, mask = mask_next
        )
        
        '''
        Event loss. This loss should not be counted into the backward loss
        '''

        the_number_of_events = mask_next.sum()

        return log_likeli_loss, marker_loss, the_number_of_events
    
    def evaluate(self, time, events, mask):
        '''
        Args:
        1. time: the sequence containing events' timestamps. shape: [batch_size, seq_len + 1]
        2. events: the sequence containing information about events. shape: [batch_size, seq_len + 1]
        3. mask: the padding mask introduced by the dataloader. shape: [batch_size, seq_len + 1]
        '''

        time_history, _ = self.divide_history_and_next(time)                   # [batch_size, seq_len]
        events_history, _ = self.divide_history_and_next(events)               # [batch_size, seq_len]
        mask_history, _ = self.divide_history_and_next(mask)                   # [batch_size, seq_len]

        history = self.model(time_history, events_history, mask_history)       # [batch_size, seq_len, num_events]
        return history.detach()

    def divide_history_and_next(self, input):
        input_history, input_next = input[:, :-1].clone(), input[:, 1:].clone()
        return input_history, input_next

    '''
    Loss functions
    '''
    def log_likelihood(self, history, time, events, mask):
        """ Log-likelihood of sequence. """
            
        if events is not None:
            type_mask = F.one_hot(events.long(), num_classes = self.num_events)# [batch_size, seq_len, num_events]
        else:
            type_mask_shape = (*history.shape, self.num_events)
            type_mask = torch.ones(type_mask_shape, device = self.device)      # [batch_size, seq_len, num_events]

        '''
        MTPP loss function
        '''
        aggregate_time = time.cumsum(dim = -1)                                 # [batch_size, seq_len]
        scaled_time = rearrange((time / aggregate_time), '... -> ... 1')       # [batch_size, seq_len, 1]
        intensity_all_events = softplus_ext(self.model.linear(history) + self.alpha * scaled_time, beta = F.softplus(self.beta))
                                                                               # [batch_size, seq_len, num_events]
        # intensity_all_events = F.softplus(self.model.linear(history) + self.alpha * scaled_time, beta = self.beta)
                                                                               # [batch_size, seq_len, num_events]

        intensity = torch.sum(intensity_all_events * type_mask, dim = -1)      # [batch_size, seq_len]

        # event log-likelihood
        log_intensity = compute_event(intensity, mask)                         # [batch_size, seq_len]
    
        # non-event log-likelihood, either numerical integration or MC integration
        intensity_integral = self.compute_integral_unbiased(history, time, mask)
                                                                               # [batch_size, seq_len]
        ll = (-log_intensity + intensity_integral).clamp(max = 15)             # [batch_size, seq_len]
    
        mtpp_loss = torch.sum(ll)

        '''
        Event loss function. Only for evaluation, do not use this loss as a part of the training loss.
        '''
        events_prediction_probability = intensity_all_events / intensity_all_events.sum(dim = -1, keepdim = True)
                                                                               # [batch_size, seq_len, num_events]
        events_prediction_probability = rearrange(events_prediction_probability, 'b s n -> (b s) n')
                                                                               # [batch_size * seq_len, num_events]
        events = rearrange(events, 'b s -> (b s)')                             # [batch_size * seq_len]
        mask = rearrange(mask, 'b s -> (b s)')                                 # [batch_size * seq_len]
        events_loss = F.cross_entropy(input = events_prediction_probability, target = events.long(), reduction = 'none')
                                                                               # [batch_size, seq_len]
        events_loss = (events_loss * mask).sum()

        return mtpp_loss, events_loss
    
    def compute_integral_unbiased(self, history, time, non_pad_mask, resolution = 100):
        """ Log-likelihood of non-events, using Monte Carlo integration. """
    
        diff_time = time * non_pad_mask
        aggregate_time = rearrange(diff_time.cumsum(dim = -1), '... -> ... 1 1')
                                                                               # [batch_size, seq_len, 1, 1]
        temp_time = rearrange(diff_time, '... -> ... 1') * \
                    torch.rand([*diff_time.size(), resolution], device = self.device)
                                                                               # [batch_size, seq_len, resolution]
        temp_time = rearrange(temp_time, '... ns -> ... ns 1')                 # [batch_size, seq_len, resolution, 1]
        temp_time = self.alpha * temp_time / aggregate_time                    # [batch_size, seq_len, resolution, num_events]

        intensity_all_events_pre_softplus = self.model.linear(history)         # [batch_size, seq_len, num_events]
        intensity_all_events_pre_softplus = repeat(intensity_all_events_pre_softplus, '... ne -> ... r ne', r = resolution)
                                                                               # [batch_size, seq_len, resolution, num_events]
        all_lambda = softplus_ext(intensity_all_events_pre_softplus + temp_time, F.softplus(self.beta))
                                                                               # [batch_size, seq_len, resolution, num_events]
        # all_lambda = F.softplus(intensity_all_events_pre_softplus + temp_time, beta = self.beta)
                                                                               # [batch_size, seq_len, resolution, num_events]
        lambda_mean = torch.mean(all_lambda, dim = -2)                         # [batch_size, seq_len, num_events]
    
        unbiased_integral_per_event = lambda_mean * rearrange(diff_time, '... -> ... 1')
                                                                               # [batch_size, seq_len, num_events]
        unbiased_integral = unbiased_integral_per_event.sum(dim = -1)          # [batch_size, seq_len]

        return unbiased_integral

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
        probabilty_expanded_events = torch.exp(-integral_sum_across_events.unsqueeze(dim = -1)) * expanded_intensity
                                                                               # [batch_size, seq_len, resolution, num_events]
        probability = expanded_time_gap.unsqueeze(dim = -1) * probabilty_expanded_events
                                                                               # [batch_size, seq_len, resolution, num_events]
        probability = probability.sum(dim = -2)                                # [batch_size, seq_len, num_events]
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
        
        mae_per_event_pure_predict = self.mean_absolute_error_per_event_worker(history, predicted_events, time_history, time_next,
                                                                               probability, resolution, mask_next, max_)
        mae_per_event = self.mean_absolute_error_per_event_worker(history, events_next, time_history, time_next, 
                                                                  probability, resolution, mask_next, max_)

        mae_per_event_pure_predict_avg = torch.sum(mae_per_event_pure_predict, dim = -1) / mask_next.sum(dim = -1)
        mae_per_event_avg = torch.sum(mae_per_event, dim = -1) / mask_next.sum(dim = -1)
        
        return f1, top_k_acc, probability_integral_sum, mask_next, (mae_per_event_pure_predict_avg, mae_per_event_avg), \
               (mae_per_event_pure_predict, mae_per_event)

    def evaluate_per_event(self, history, events_type_mask, time_next, tau, resolution):
        # Intensity and integral estimation
        time_multiplier = torch.linspace(0, 1, resolution, device = self.device)
        expanded_time = rearrange(tau, '... -> ... 1') * time_multiplier       # [batch_size, seq_len, resolution]
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
        probabilty_expanded_events = torch.exp(-integral_sum_across_events.unsqueeze(dim = -1)) * expanded_intensity
                                                                               # [batch_size, seq_len, resolution, num_events]
        probability = expanded_time_gap.unsqueeze(dim = -1) * probabilty_expanded_events
                                                                               # [batch_size, seq_len, resolution, num_events]
        probability = probability.sum(dim = -2)                                # [batch_size, seq_len, num_events]
        probability = (probability * events_type_mask).sum(dim = -1)           # [batch_size, seq_len]

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

            return p_gap
            
        def median_prediction(events_history, l, r):
            for _ in range(50):
                c = (l + r)/2
                v = bisect_target(events_history, c)
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
        self.model.eval()

        time, events, _, mask, _ = input_data[0]                               # 3 * [batch_size, seq_len + 1]
        _, time_next = self.divide_history_and_next(time)                      # [batch_size, seq_len]
        _, events_next = self.divide_history_and_next(events)                  # [batch_size, seq_len]
        _, mask_next = self.divide_history_and_next(mask)                      # [batch_size, seq_len]
        aggregate_time_next = time_next.cumsum(dim = -1).unsqueeze(-1)         # [batch_size, seq_len, 1]

        batch_size, seq_len = time.shape
        seq_len -= 1
        history = self.evaluate(time, events, mask)                            # [batch_size, seq_len, d_input]

        time_multiplier = torch.linspace(0, 1, resolution, device = self.device)
        expanded_time = time_next.unsqueeze(-1) * time_multiplier              # [batch_size, seq_len, resolution]
        expanded_time_gap = torch.diff(expanded_time, dim = -1).mean(dim = -1, keepdim = True)  
                                                                               # [batch_size, seq_len, 1]

        scaled_expanded_time = (expanded_time / aggregate_time_next).unsqueeze(-1)
                                                                               # [batch_size, seq_len, resolution, 1]
        intensity_for_each_event = self.model.linear(history)                  # [batch_size, seq_len, num_events]
        intensity_for_each_event = intensity_for_each_event.unsqueeze(dim = -2)# [batch_size, seq_len, 1, num_events]
        
        expanded_intensity = torch.sum(softplus_ext(self.alpha * scaled_expanded_time + intensity_for_each_event, F.softplus(self.beta)), dim =-1)
                                                                               # [batch_size, seq_len, resolution]
        
        # Only works when expanded_time_gap is tiny.
        expanded_integral_unbiased = (expanded_intensity * expanded_time_gap).cumsum(dim = -1)
                                                                               # [batch_size, seq_len, resolution]

        expanded_intensity = expanded_intensity.reshape(batch_size, -1)        # [batch_size, seq_len * resolution]
        expanded_integral_unbiased = expanded_integral_unbiased.reshape(batch_size, -1)
                                                                               # [batch_size, seq_len * resolution]

        # aggregated timestamp
        timestamp = torch.cat(
            (torch.zeros((batch_size, seq_len, 1), device = self.device), expanded_time.diff(dim = -1)),
            dim = -1)                                                          # [batch_size, seq_len, resolution]
        timestamp = timestamp.reshape(batch_size, seq_len * resolution)        # [batch_size, seq_len * resolution]
        
        return expanded_integral_unbiased, expanded_intensity, timestamp
    
    def model_prober(self, input_data, resolution):
        '''
        Probe the learned intensity function from the model.
        This task should be pretty easy for the explicit form of intensity functions.
        '''
        self.model.eval()

        time, events, _, mask, _ = input_data[0]                               # 3 * [batch_size, seq_len + 1]
        mean, var = input_data[1]

        _, time_next = self.divide_history_and_next(time)                      # [batch_size, seq_len]
        _, events_next = self.divide_history_and_next(events)                  # [batch_size, seq_len]
        _, mask_next = self.divide_history_and_next(mask)                      # [batch_size, seq_len]
        aggregate_time_next = time_next.cumsum(dim = -1).unsqueeze(-1)         # [batch_size, seq_len, 1]

        batch_size, seq_len = time.shape
        seq_len -= 1
        history = self.evaluate(time, events, mask)                            # [batch_size, seq_len, d_input]

        time_multiplier = torch.linspace(0, 1, resolution, device = self.device)
        expanded_time = time_next.unsqueeze(-1) * time_multiplier              # [batch_size, seq_len, resolution]
        expanded_time_gap = torch.diff(expanded_time, dim = -1).mean(dim = -1, keepdim = True)
        expanded_time_gap = rearrange(expanded_time_gap, '... -> ... 1')
                                                                               # [batch_size, seq_len, 1, 1]

        scaled_expanded_time = (expanded_time / aggregate_time_next).unsqueeze(-1)
                                                                               # [batch_size, seq_len, resolution, 1]
        intensity_for_each_event = self.model.linear(history)                  # [batch_size, seq_len, num_events]
        intensity_for_each_event = intensity_for_each_event.unsqueeze(dim = -2)# [batch_size, seq_len, 1, num_events]
        
        expand_intensity = softplus_ext(self.alpha * scaled_expanded_time + intensity_for_each_event, F.softplus(self.beta))
                                                                               # [batch_size, seq_len, resolution, num_events]

        # Only works when expanded_time_gap is tiny.
        expanded_integral_unbiased = (expand_intensity * expanded_time_gap).cumsum(dim = -1)
                                                                               # [batch_size, seq_len, resolution, num_events]

        # Tune tensors' shape
        expand_intensity = rearrange(expand_intensity, 'b s r ne -> b (s r) ne')
                                                                               # [batch_size, seq_len * resolution, num_events]
        expanded_integral_unbiased = rearrange(expanded_integral_unbiased, 'b s r ne -> b (s r) ne')
                                                                               # [batch_size, seq_len * resolution, num_events]

        intensity_and_integral_plot = {}
        expanded_intensity = torch.chunk(expand_intensity, self.num_events, dim = -1)
                                                                               # [batch_size, seq_len * resolution, 1] * num_events
        expanded_integral_unbiased = torch.chunk(expanded_integral_unbiased, self.num_events, dim = -1)
                                                                               # [batch_size, seq_len * resolution, 1] * num_events
        for idx, (intensity_per_event, integral_per_event) in enumerate(zip(expanded_intensity, expanded_integral_unbiased)):
            intensity_and_integral_plot[f'event_intensity_{idx}'] = rearrange(intensity_per_event, '... 1 -> ...')
                                                                               # [batch_size, seq_len * resolution]
            intensity_and_integral_plot[f'event_integral_{idx}'] = rearrange(integral_per_event, '... 1 -> ...')
                                                                               # [batch_size, seq_len * resolution]

        # aggregated timestamp
        timestamp = torch.cat(
            (torch.zeros((batch_size, seq_len, 1), device = self.device), expanded_time.diff(dim = -1)),
            dim = -1)                                                          # [batch_size, seq_len, resolution]
        timestamp = timestamp.reshape(batch_size, seq_len * resolution)        # [batch_size, seq_len * resolution]
        
        # Additional plots
        additional_plot = []
        f1, top_k, probability_sum, mask, maes_avg, maes \
                = self.mean_absolute_error_per_event(
                    input_time = time, input_events = events, mask = mask, mean = mean, var = var, fast = False
                )
        mae_per_event_pure_predict_avg, mae_per_event_avg = maes_avg
        mae_per_event_pure_predict, mae_per_event = maes

        probability_sum = probability_sum.detach().cpu().numpy()               # [batch_size, seq_len]
        mae_per_event_pure_predict_avg = mae_per_event_pure_predict_avg.detach().cpu().numpy()
                                                                               # [batch_size]
        mae_per_event_avg = mae_per_event_avg.detach().cpu().numpy()           # [batch_size]
        mae_per_event_pure_predict = mae_per_event_pure_predict.detach().cpu().numpy()
                                                                               # [batch_size, seq_len]
        mae_per_event = mae_per_event.detach().cpu().numpy()                   # [batch_size, seq_len]
        expand_intensity = expand_intensity.detach().cpu().numpy()             # [batch_size, seq_len * resolution, num_events]

        packed_values = zip(f1, top_k, probability_sum, mae_per_event_pure_predict, mae_per_event_pure_predict_avg, \
                            mae_per_event, mae_per_event_avg, expand_intensity, time_next, mask_next)

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
        tpp_loss, mark_loss, the_number_of_events = model(time, events, mask)
        loss = tpp_loss
        loss.backward()

        tpp_loss, mark_loss = tpp_loss.item(), mark_loss.item()
        fact = fact.sum()
    
        return tpp_loss / the_number_of_events , mark_loss / the_number_of_events, fact / the_number_of_events
    
    def evaluation_step(model, minibatch, device):
        ''' Epoch operation in evaluation phase '''
    
        model.eval()

        time, events, fact, mask = minibatch[0]                                 # 3 * [batch_size, seq_len + 1, 1] & [batch_size, seq_len, 1]
        tpp_loss, mark_loss, the_number_of_events = model(time, events, mask)

        tpp_loss, mark_loss = tpp_loss.item(), mark_loss.item()
        fact = fact.sum()

        return tpp_loss / the_number_of_events, mark_loss / the_number_of_events, fact / the_number_of_events

    def postprocess(input, procedure):
        return [input[0], input[0] - input[2], input[1]]


    def log_print_format(input, procedure):
        format_dict = {}
        format_dict['absolute_loss'] = input[0]
        format_dict['relative_loss'] = input[1]
        format_dict['events_loss'] = input[2]
        format_dict['num_format'] = {'absolute_loss': ':8.5f', 'relative_loss': ':8.5f', 'events_loss': ':8.5f'}
        return format_dict
        
    format_dict_length = 3
    

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