import torch
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
from sklearn.metrics import f1_score, top_k_accuracy_score, accuracy_score
import numpy as np

from .transformers import TransformerTPP
from ..utils import BasicModule
from .utils import *

class THP(BasicModule):
    def __init__(self, num_events, device, d_input = 64, d_rnn = 64, d_hidden = 256, n_layers = 3,
                 n_head = 3, d_qk = 64, d_v = 64, dropout = 0.1, beta = 1, mae_threshold = 2):
        super(THP, self).__init__()
        self.device = device
        self.num_events = num_events if num_events > 0 else 1
        self.mae_threshold = mae_threshold

        # parameter for the weight of time difference
        self.alpha = nn.Parameter(torch.tensor(0, dtype = torch.float32, device = self.device, requires_grad = True))

        # parameter for the softplus function
        self.beta = beta

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

        history = self.model(time_history, events_history, mask_history)       # [batch_size, seq_len, num_types]
        return history.detach()

    def divide_history_and_next(self, input, unsqueeze = False):
        input_history, input_next = input.clone()[:, :-1], input.clone()[:, 1:]
        if unsqueeze:
            input_history = input_history.unsqueeze(-1)                        # [batch_size, seq_len, 1]
            input_next = input_next.unsqueeze(-1)                              # [batch_size, seq_len, 1]
        return input_history, input_next

    '''
    Loss functions
    '''
    def log_likelihood(self, history, time, events, mask):
        """ Log-likelihood of sequence. """
            
        if events is not None:
            type_mask = F.one_hot(events.long(), num_classes = self.num_events)# [batch_size, seq_len, num_types]
        else:
            type_mask = torch.ones_like(history, device = history.device)      # [batch_size, seq_len, num_types]

        '''
        MTPP loss function
        '''
        aggregate_time = time.cumsum(dim = -1)                                 # [batch_size, seq_len]
        scaled_time = (time / aggregate_time).unsqueeze(dim = -1)              # [batch_size, seq_len, 1]
        intensity_all_event = F.softplus(self.model.linear(history) + self.alpha * scaled_time, beta = self.beta)
                                                                               # [batch_size, seq_len, num_types]
        intensity = torch.sum(intensity_all_event * type_mask, dim = -1)       # [batch_size, seq_len]

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
        event_prediction_probability = intensity_all_event / intensity_all_event.sum(dim = -1, keepdim = True)
                                                                               # [batch_size, seq_len, num_types]
        event_prediction_probability = rearrange(event_prediction_probability, 'b s n -> (b s) n')
                                                                               # [batch_size * seq_len, num_types]
        events = rearrange(events, 'b s -> (b s)')                             # [batch_size * seq_len]
        mask = rearrange(mask, 'b s -> (b s)')                                 # [batch_size * seq_len]
        event_loss = F.cross_entropy(input = event_prediction_probability, target = events.long(), reduction = 'none')
                                                                               # [batch_size, seq_len]
        event_loss = (event_loss * mask).sum()

        return mtpp_loss, event_loss
    
    def compute_integral_unbiased(self, history, time, non_pad_mask):
        """ Log-likelihood of non-events, using Monte Carlo integration. """
        num_samples = 100
    
        diff_time = time * non_pad_mask
        aggregate_time = diff_time.cumsum(dim = -1).unsqueeze(dim = -1).unsqueeze(dim = -1)
                                                                               # [batch_size, seq_len, 1, 1]
        temp_time = diff_time.unsqueeze(2) * \
                    torch.rand([*diff_time.size(), num_samples], device=history.device)
                                                                               # [batch_size, seq_len, num_samples]
        temp_time = temp_time.unsqueeze(dim = -2)                              # [batch_size, seq_len, 1, num_samples]

        intensity_all_event_pre_softplus = self.model.linear(history)          # [batch_size, seq_len, num_types]
        intensity_all_event_pre_softplus = intensity_all_event_pre_softplus.unsqueeze(dim = -1).repeat(1, 1, 1, num_samples)
                                                                               # [batch_size, seq_len, num_types, num_samples]

        all_lambda = F.softplus(intensity_all_event_pre_softplus + self.alpha * temp_time / aggregate_time, self.beta)
                                                                               # [batch_size, seq_len, num_types, num_samples]
        lambda_mean = torch.mean(all_lambda, dim = -1)                         # [batch_size, seq_len, num_types]
    
        unbiased_integral_per_event = lambda_mean * diff_time.unsqueeze(dim = -1)
                                                                               # [batch_size, seq_len, num_types]
        unbiased_integral = unbiased_integral_per_event.sum(dim = -1)          # [batch_size, seq_len]

        return unbiased_integral

    def mean_absolute_error_per_event(self, input_time, input_events, mask, mean, var, fast):
        '''
        The precedure resembles the compute_integral_unbiased() but the output of small step MC takes would
        be recorded as part of the output.
        '''
        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # [batch_size, seq_len]
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
        expanded_time = torch.ones_like(time_next, device = time_next.device) * max_
                                                                               # [batch_size, seq_len]
        expanded_time = expanded_time.unsqueeze(dim = -1) * time_multiplier    # [batch_size, seq_len, resolution]
        expanded_time_gap = torch.diff(expanded_time, dim = -1).mean(dim = -1, keepdim = True)
                                                                               # [batch_size, seq_len, 1]
        aggregated_time = torch.cumsum(time_next, dim = -1).unsqueeze(dim = -1)# [batch_size, seq_len, 1]
        scaled_expanded_time = expanded_time / aggregated_time                 # [batch_size, seq_len, resolution]
        scaled_expanded_time = scaled_expanded_time.unsqueeze(dim = -1)        # [batch_size, seq_len, resolution, 1]

        intensity_for_each_event = self.model.linear(history).detach()         # [batch_size, seq_len, num_types]
        intensity_for_each_event = intensity_for_each_event.unsqueeze(dim = -2)# [batch_size, seq_len, 1, num_types]
        
        expanded_intensity = F.softplus(self.alpha.detach() * scaled_expanded_time + intensity_for_each_event, self.beta)
                                                                               # [batch_size, seq_len, resolution, num_types]
        intensity_sum_across_events = torch.sum(expanded_intensity, dim = -1)  # [batch_size, seq_len, resolution]
        integral_sum_across_events = torch.cumsum(intensity_sum_across_events * expanded_time_gap, dim = -1)
                                                                               # [batch_size, seq_len, resolution]
        probabilty_expanded_event = integral_sum_across_events.unsqueeze(dim = -1) * expanded_intensity
                                                                               # [batch_size, seq_len, resolution, num_types]
        probability = expanded_time_gap.unsqueeze(dim = -1) * probabilty_expanded_event
                                                                               # [batch_size, seq_len, resolution, num_types]
        probability = probability.sum(dim = -2)                                # [batch_size, seq_len, num_types]
        predicted_event = torch.argmax(probability, dim = -1)                  # [batch_size, seq_len]

        # F1 value and top_k_acc are only avaliable when batch_size = 1
        f1 = f1_score(y_true = events_next.squeeze().detach().cpu(),
                      y_pred = predicted_event.squeeze().detach().cpu(), average = 'macro')
        
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
                        y_pred = predicted_event.squeeze().detach().cpu()
                    )
                )
                top_k_acc.append(1.0)
        
        # placeholder
        probability_integral_sum = 0
        mae_per_event_pure_predict_avg = torch.tensor(0)
        mae_per_event_avg = torch.tensor(0)
        mae_per_event_pure_predict = 0
        mae_per_event = 0

        if mean == 0:
            resolution = max(min(int(input_time.mean().item() * 200), 1000), 1)
        else:
            resolution = max(min(int(mean * 200), 1000), 1)
        
        mae_per_event_pure_predict = self.mean_absolute_error_per_event_worker(history, predicted_event, time_history, time_next,
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

        intensity_for_each_event = self.model.linear(history).detach()         # [batch_size, seq_len, num_types]
        intensity_for_each_event = intensity_for_each_event.unsqueeze(dim = -2)# [batch_size, seq_len, 1, num_types]
        
        expanded_intensity = F.softplus(self.alpha.detach() * scaled_expanded_time + intensity_for_each_event, self.beta)
                                                                               # [batch_size, seq_len, resolution, num_types]
        intensity_sum_across_events = torch.sum(expanded_intensity, dim = -1)  # [batch_size, seq_len, resolution]
        integral_sum_across_events = torch.cumsum(intensity_sum_across_events * expanded_time_gap, dim = -1)
                                                                               # [batch_size, seq_len, resolution]
        probabilty_expanded_event = integral_sum_across_events.unsqueeze(dim = -1) * expanded_intensity
                                                                               # [batch_size, seq_len, resolution, num_types]
        probability = expanded_time_gap.unsqueeze(dim = -1) * probabilty_expanded_event
                                                                               # [batch_size, seq_len, resolution, num_types]
        probability = probability.sum(dim = -2)                                # [batch_size, seq_len, num_types]
        probability = (probability * events_mask).sum(dim = -1)                # [batch_size, seq_len]

        return probability

    def mean_absolute_error_per_event_worker(self, history, events_next, 
        time_history, time_next, probability_integral, resolution, mask, max_val):
        '''
        The input should be the original minibatch
        MAE evaluation part, dwg and fullynn exclusive

        '''
        def bisect_target(events_history, taus):
            event_next_one_hot = torch.nn.functional.one_hot(events_next.long(), num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_event]
            p_xt = self.evaluate_per_event(events_history, event_next_one_hot , time_next, taus, resolution)
                                                                               # [batch_size, seq_len]
            p_x = torch.sum(probability_integral * event_next_one_hot, dim = -1)
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
        intensity_for_each_event = self.model.linear(history)                  # [batch_size, seq_len, num_types]
        intensity_for_each_event = intensity_for_each_event.unsqueeze(dim = -2)# [batch_size, seq_len, 1, num_types]
        
        expanded_intensity = torch.sum(F.softplus(self.alpha * scaled_expanded_time + intensity_for_each_event, self.beta), dim =-1)
                                                                               # [batch_size, seq_len, resolution]
        
        # Only works when expanded_time_gap is tiny.
        expanded_integral_unbiased = (expanded_intensity * expanded_time_gap).cumsum(dim = -1)
                                                                               # [batch_size, seq_len, resolution]

        expanded_intensity = expanded_intensity.reshape(batch_size, -1)        # [batch_size, seq_len * resolution]

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
        intensity_for_each_event = self.model.linear(history)                  # [batch_size, seq_len, num_types]
        intensity_for_each_event = intensity_for_each_event.unsqueeze(dim = -2)# [batch_size, seq_len, 1, num_types]
        
        expanded_intensity = torch.sum(F.softplus(self.alpha * scaled_expanded_time + intensity_for_each_event, self.beta), dim =-1)
                                                                               # [batch_size, seq_len, resolution]
        
        # Only works when expanded_time_gap is tiny.
        expanded_integral_unbiased = (expanded_intensity * expanded_time_gap).cumsum(dim = -1)
                                                                               # [batch_size, seq_len, resolution]

        expanded_intensity = expanded_intensity.reshape(batch_size, -1)        # [batch_size, seq_len * resolution]

        # aggregated timestamp
        timestamp = torch.cat(
            (torch.zeros((batch_size, seq_len, 1), device = self.device), expanded_time.diff(dim = -1)),
            dim = -1)                                                          # [batch_size, seq_len, resolution]
        timestamp = timestamp.reshape(batch_size, seq_len * resolution)        # [batch_size, seq_len * resolution]
        
        return expanded_integral_unbiased, expanded_intensity, timestamp

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
        time, event, fact, mask = minibatch[0]                                 # 3 * [batch_size, seq_len + 1, 1] & [batch_size, seq_len, 1]
        tpp_loss, mark_loss, the_number_of_events = model(time, event, mask)
        loss = tpp_loss
        loss.backward()

        tpp_loss, mark_loss = tpp_loss.item(), mark_loss.item()
        fact = fact.sum()
    
        return tpp_loss / the_number_of_events , mark_loss / the_number_of_events, fact / the_number_of_events
    
    def evaluation_step(model, minibatch, device):
        ''' Epoch operation in evaluation phase '''
    
        model.eval()

        time, event, fact, mask = minibatch[0]                                 # 3 * [batch_size, seq_len + 1, 1] & [batch_size, seq_len, 1]
        tpp_loss, mark_loss, the_number_of_events = model(time, event, mask)

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