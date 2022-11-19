from ..utils import BasicModule
from .rmtpp import RMTPPModule
from einops import rearrange, reduce, repeat
from sklearn.metrics import f1_score, top_k_accuracy_score, accuracy_score

import torch
import numpy as np

# We use syn dataloader for this model.
class RMTPP(BasicModule):
    def __init__(self, device, input_size, hidden_size, history_encoder_layers, dropout, num_events, output_size, limited_history_norm, 
                 original_mark_generation, mae_threshold = 2):
        super(RMTPP, self).__init__()
        self.device = device
        self.num_events = num_events
        self.limited_history_norm = limited_history_norm
        self.original_mark_generation = original_mark_generation
        self.mae_threshold = mae_threshold

        self.submodel = RMTPPModule(input_size = input_size, hidden_size = hidden_size, history_encoder_layers = history_encoder_layers, 
                                    dropout = dropout, num_events = num_events, output_size = output_size, 
                                    limited_history_norm = limited_history_norm, original_mark_generation = original_mark_generation, 
                                    device = device)

    def forward(self, event, time, mean, var, mask, evaluation = False):
        event_history, event_next = self.divide_history_and_next(event, unsqueeze = False)
                                                                               # [batch_size, seq_len]
        time_history, time_next = self.divide_history_and_next(time, unsqueeze = True)
                                                                               # [batch_size, seq_len, 1]
        _, mask_next = self.divide_history_and_next(mask, unsqueeze = False)   # [batch_size, seq_len]

        intensity, integral, mark, constant = self.submodel(event_history, time_history, time_next, mean, var)

        if torch.isnan(intensity).any():
            print(intensity)
        if torch.isnan(integral).any():
            print(integral)

        loss, time_loss, event_loss, the_number_of_events =\
                    self.loss_f(intensity, integral, mark, event_next, mask_next)
        
        mae = 0
        if evaluation:
            '''
            Calculating MAE here.
            '''
            mae = self.mean_absolute_error(event_history, time_history, time_next, mask_next, mean, var)
            
        return loss, mae, time_loss, event_loss, the_number_of_events, constant

    def evaluate(self, event_history, time_history, taus, mean, var):
        _, integral, _, _ = \
            self.submodel(event_history, time_history, taus, mean, var)
                                                                               # [batch_size, seq_len, num_events]
        
        return integral.sum(dim = -1, keepdim = True)

    def divide_history_and_next(self, input, unsqueeze):
        history, next = input.clone()[:, :-1], input.clone()[:, 1:]
        if unsqueeze:
            history = history.unsqueeze(-1)
            next = next.unsqueeze(-1)
        return history, next                                                   # [batch_size, seq_len, 1] or [batch_size, seq_len]

    def loss_f(self, intensity, integral, mark, event_next, mask_next):
        # temporal point process loss
        # intensity shape: [batch, seq_length]
        # so does tensor mask.

        loss = 0
        time_loss, event_loss = 0, 0
        if self.num_events > 1:
            event_loss = torch.nn.functional.cross_entropy(input = mark.transpose(1, 2), \
                                                    target = event_next.to(self.device).long(), \
                                                    reduction = 'none')
            event_loss = event_loss.clamp(max = 15) * mask_next
            event_loss = event_loss.sum()
        else:
            event_loss = torch.tensor(0., device = self.device)

        if self.original_mark_generation:
            time_loss = -torch.log(intensity + 1e-6) + integral                # [batch_size, seq_len, 1]
            time_loss = time_loss.clamp(max = 15).squeeze(dim = -1) * mask_next
            time_loss = time_loss.sum()

            loss = time_loss + event_loss
        else:
            event_mask = torch.nn.functional.one_hot(event_next.long(), num_class = self.num_events)
                                                                               # [batch_size, seq_len, num_events]
            intensity = (intensity * event_mask).sum(dim = -1)                 # [batch_size, seq_len]
            integral = integral.sum(dim = -1)                                  # [batch_size, seq_len]
            time_loss = - torch.log(intensity + 1e-6) + integral               # [batch_size, seq_len]
            time_loss = time_loss.clamp(max = 15) * mask_next                  # [batch_size, seq_len]
            time_loss = time_loss.sum()
            
            loss = time_loss

        return loss, time_loss, event_loss, mask_next.sum()

    def mean_absolute_error(self, event_history, time_history, time_next, mask_next, mean, var):
        '''
        The input should be the original minibatch
        MAE evaluation part, dwg and fullynn exclusive
        '''
        def bisect_target(event_history, time_history, taus, mean, var):
            return self.evaluate(event_history, time_history, taus, mean, var) - \
                   torch.log(torch.tensor(self.mae_threshold, device = time_history.device))
            
        def median_prediction(event_history, time_history, l, r, mean, var):
            for _ in range(50):
                c = (l + r)/2
                v = bisect_target(event_history, time_history, c, mean, var)
                l = torch.where(v < 0, c, l)
                r = torch.where(v >= 0, c, r)

            return (l + r)/2
        
        l = 0.0001*torch.ones_like(time_history, dtype = torch.float32)        # [batch_size, seq_len, 1]
        r = 1e6*torch.ones_like(time_history, dtype = torch.float32)           # [batch_size, seq_len, 1]
        tau_pred = median_prediction(event_history, time_history, l, r, mean, var)
        gap = (tau_pred - time_next).squeeze(-1) * mask_next
        gap_mean = torch.sum(torch.abs(gap)) / mask_next.sum()
        return gap_mean.item()

    def function_prober(self, data, resolution):
        (time, event, _, _, _), (mean, var) = data                               # 2 * [batch_size, seq_len + 1]
        event_history, _ = self.divide_history_and_next(event, unsqueeze = False)
                                                                               # [batch_size, seq_len]
        time_history, time_next = self.divide_history_and_next(time, unsqueeze = True)
                                                                               # [batch_size, seq_len, 1]
        intensity, integral, timestamp = self.submodel.intensity_integral(event_history, time_history, time_next, resolution, mean, var)
                                                                               # 3 * [batch_size, seq_len * resolution]
        
        return integral, intensity, timestamp


    def mean_absolute_error_per_event(self, input_time, input_events, mask, mean, var, fast):
        if self.original_mark_generation:
            raise Exception('Original RMTPP model is in fact a TPP model with dedicated event prediction module, so \
                             pe-MAE does not function here.')
        
        time_history, time_next = self.divide_history_and_next(input_time, unsqueeze = True)
                                                                               # [batch_size, seq_len, 1]
        events_history, events_next = self.divide_history_and_next(input_events, unsqueeze = False)
                                                                               # [batch_size, seq_len]
        _, mask_next = self.divide_history_and_next(mask, unsqueeze = False)   # [batch_size, seq_len]

        if mean == 0 and var == 1:
            '''
            This dataset does not apply normalisation, so we need to calculate the mean and variance here.
            '''
            mean = input_time.mean()
            var = input_time.var()
        
        # Use a relatively large number as the positive infinity.
        max_ = min(1e5, mean + 10 * var)
        resolution = min(int(max_ * 100), 5000)
        time_infinite = torch.ones_like(time_next, device = time_next.device) * max_
                                                                               # [batch_size, seq_len, 1]

        # First, we find the integral and intensity function that RMTPP estimates.
        # This part is only available when original_mark_generation is false as the original RMTPP model
        # is in fact a TPP model.
        intensity, integral, timestamp = \
                self.submodel.intensity_integral(events_history, time_history, time_infinite, resolution, mean, var, sum = False)
                                                                               # 2 * [batch_size, seq_len * resolution, num_event] + [batch_size, seq_len * resolution]
        probability_dist = intensity * torch.exp(-integral.sum(dim = -1, keepdim = True))
                                                                               # [batch_size, seq_len * resolution, num_event]
        cumulated_probability = probability_dist * timestamp.unsqueeze(dim = -1)
        cumulated_probability = rearrange(cumulated_probability, 'b (s r) ne -> b s r ne', r = resolution)
                                                                               # [batch_size, seq_len, resolution, num_event]
        probability = cumulated_probability.sum(dim = -2)                      # [batch_size, seq_len, num_event]
        probability_integral_sum = probability.sum(dim = -1)                   # [batch_size, seq_len]
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
        
        if mean == 0:
            resolution = max(min(int(input_time.mean().item() * 200), 1000), 1)
        else:
            resolution = max(min(int(mean * 200), 1000), 1)
        
        mae_per_event_pure_predict = self.mean_absolute_error_per_event_worker(events_history, predicted_event, time_history, time_next,
                                                                               probability, resolution, mask_next, max_, mean, var)
        mae_per_event = self.mean_absolute_error_per_event_worker(events_history, events_next, time_history, time_next, 
                                                                  probability, resolution, mask_next, max_, mean, var)

        mae_per_event_pure_predict_avg = torch.sum(mae_per_event_pure_predict) / mask_next.sum()
        mae_per_event_avg = torch.sum(mae_per_event) / mask_next.sum()
        
        return f1, top_k_acc, probability_integral_sum, (mae_per_event_pure_predict_avg.item(), mae_per_event_avg.item()), \
               (mae_per_event_pure_predict, mae_per_event)

    def evaluate_per_event(self, events_history, time_history, events_mask, tau, resolution, mean, var):
        intensity, integral, timestamp = \
                        self.submodel.intensity_integral(events_history, time_history, tau, resolution, mean, var, sum = False)
                                                                               # 2 * [batch_size, seq_len * resolution, num_events] + [batch_size, seq_len * resolution]
        probability_dist = intensity * torch.exp(integral.sum(dim = -1, keepdim = True))
                                                                               # [batch_size, seq_len * resolution, num_event]
        cumulative_probability = probability_dist * timestamp.unsqueeze(dim = -1)
                                                                               # [batch_size, seq_len * resolution, num_event]
        cumulative_probability = rearrange(cumulative_probability, 'b (s r) n -> b s r n', r = resolution)
                                                                               # [batch_size, seq_len, resolution, num_event]
        cumulative_probability = cumulative_probability.sum(dim = -2)          # [batch_size, seq_len, num_event]
        cumulative_probability = cumulative_probability * events_mask          # [batch_size, seq_len, num_event]
        probability = cumulative_probability.sum(dim = -1)                     # [batch_size, seq_len]

        return probability

    def mean_absolute_error_per_event_worker(self, events_history, events_next, 
        time_history, time_next, probability_integral, resolution, mask, max_val, mean, var):
        '''
        The input should be the original minibatch
        MAE evaluation part, dwg and fullynn exclusive

        '''
        def bisect_target(taus):
            event_next_one_hot = torch.nn.functional.one_hot(events_next.long(), num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_event]
            p_xt = self.evaluate_per_event(events_history, time_history, event_next_one_hot, taus, resolution, mean, var)
                                                                               # [batch_size, seq_len]
            p_x = torch.sum(probability_integral * event_next_one_hot, dim = -1)
                                                                               # [batch_size, seq_len]
            p_t_x = p_xt / p_x                                                 # [batch_size, seq_len]
            p_gap = p_t_x - (1 / self.mae_threshold)                           # [batch_size, seq_len]

            return p_gap.unsqueeze(dim = -1)
            
        def median_prediction(l, r):
            for _ in range(50):
                c = (l + r)/2
                v = bisect_target(c)
                l = torch.where(v < 0, c, l)
                r = torch.where(v >= 0, c, r)

            return (l + r)/2
        
        l = 0.0001*torch.ones_like(time_history, dtype = torch.float32)        # [batch_size, seq_len, 1]
        r = max_val*torch.ones_like(time_history, dtype = torch.float32)       # [batch_size, seq_len, 1]
        tau_pred = median_prediction(l, r)
        gap = (tau_pred - time_next).squeeze(-1) * mask
        gap = torch.abs(gap)

        return gap


    def train_step(model, minibatch, device):
        model.train()
        
        [time, event, score, mask], (mean, var) = minibatch                    # 4 * [batch_size, seq_len + 1]
        loss, mae, time_loss, event_loss, the_number_of_events, constant = model(event, time, mean, var, mask)

        loss.backward()

        time_loss = time_loss.item() / the_number_of_events
        event_loss = event_loss.item() / the_number_of_events
        fact = score.sum().item() / the_number_of_events
        constant_norm = torch.linalg.norm(constant).detach().item() / the_number_of_events

        return time_loss, event_loss, fact, constant_norm

    def evaluation_step(model, minibatch, device):
        model.eval()

        [time, event, score, mask], (mean, var) = minibatch                    # 4 * [batch_size, seq_len + 1]
        _, mae, time_loss, event_loss, the_number_of_events, constant\
            = model(event, time, mean, var, mask, evaluation = True)

        time_loss = time_loss.item() / the_number_of_events
        event_loss = event_loss.item() / the_number_of_events
        fact = score.sum().item() / the_number_of_events
        constant_norm = torch.linalg.norm(constant).detach().item() / the_number_of_events

        return time_loss, event_loss, fact, mae, constant_norm

    def postprocess(input, procedure):
        if procedure == 'Training':
            return [input[0], input[1], input[0] - input[2], input[3]]
        else:
            return [input[0], input[1], input[0] - input[2], input[3], input[4]]

    def log_print_format(input, procedure):
        def format_training(input):
            format_dict = {}
            format_dict['absolute_loss'] = input[0]
            format_dict['event_loss'] = input[1]
            format_dict['relative_loss'] = input[2]
            format_dict['constant_norm'] = input[3]
            format_dict['num_format'] = {'absolute_loss': ':8.5f', 'relative_loss': ':8.5f', 'event_loss': ':8.5f', 'constant_norm': ':8.5f'}
            return format_dict

        def format_eva_and_test(input):
            format_dict = {}
            format_dict['absolute_loss'] = input[0]
            format_dict['event_loss'] = input[1]
            format_dict['relative_loss'] = input[2]
            format_dict['MAE'] = input[3]
            format_dict['constant_norm'] = input[4]
            format_dict['num_format'] = {'absolute_loss': ':8.5f', 'relative_loss': ':8.5f', 'event_loss': ':8.5f', 'MAE': ':2.8f', 'constant_norm': ':8.5f'}
            return format_dict

        return format_training(input) if procedure == 'Training' else format_eva_and_test(input)
    
    format_dict_length = 5
    
    logfile_format = {'step': '', 'absolute loss': ':8.5f', 'relative loss': ':8.5f', 'event_loss': ':8.5f', 'constant_norm': ':8.5f'}

    def logfile_print_format(input):
        format_dict = {}
        format_dict['absolute loss'] = input[0]
        format_dict['event_loss'] = input[1]
        format_dict['relative loss'] = input[2]
        format_dict['constant_norm'] = input[3]
        return format_dict
    
    def choose_metric(evaluation_report, test_report):
        '''
        [relative loss on evaluation dataset, relative loss on test dataset]
        '''
        return [evaluation_report[3], test_report[3]]
    
    metric_number = 2 # metric number is the length of the output of choose_metric