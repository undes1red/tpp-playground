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

    def forward(self, events, time, mean, var, mask, evaluation = False):
        events_history, events_next = self.divide_history_and_next(events, unsqueeze = False)
                                                                               # [batch_size, seq_len]
        time_history, time_next = self.divide_history_and_next(time, unsqueeze = True)
                                                                               # [batch_size, seq_len, 1]
        _, mask_next = self.divide_history_and_next(mask, unsqueeze = False)   # [batch_size, seq_len]

        intensity, integral, mark, constant = self.submodel(events_history, time_history, time_next, mean, var)

        loss, time_loss, events_loss, the_number_of_events =\
                    self.loss_f(intensity, integral, mark, events_next, mask_next)
        
        mae = 0
        if evaluation:
            '''
            Calculating MAE here.
            '''
            mae = self.mean_absolute_error(events_history, time_history, time_next, mask_next, mean, var)
            
        return loss, mae, time_loss, events_loss, the_number_of_events, constant

    def evaluate(self, events_history, time_history, taus, mean, var):
        _, integral, _, _ = \
            self.submodel(events_history, time_history, taus, mean, var)
                                                                               # [batch_size, seq_len, num_events]
        
        return integral.sum(dim = -1, keepdim = True)

    def divide_history_and_next(self, input, unsqueeze):
        history, next = input.clone()[:, :-1], input.clone()[:, 1:]
        if unsqueeze:
            history = history.unsqueeze(-1)
            next = next.unsqueeze(-1)
        return history, next                                                   # [batch_size, seq_len, 1] or [batch_size, seq_len]

    def loss_f(self, intensity, integral, mark, events_next, mask_next):
        # temporal point process loss
        # intensity shape: [batch, seq_length]
        # so does tensor mask.

        loss = 0
        time_loss, events_loss = 0, 0
        if self.num_events > 1:
            events_loss = torch.nn.functional.cross_entropy(input = mark.transpose(1, 2), \
                                                    target = events_next.to(self.device).long(), \
                                                    reduction = 'none')
            events_loss = events_loss.clamp(max = 15) * mask_next
            events_loss = events_loss.sum()
        else:
            events_loss = torch.tensor(0., device = self.device)

        if self.original_mark_generation:
            time_loss = -torch.log(intensity + 1e-6) + integral                # [batch_size, seq_len, 1]
            time_loss = time_loss.clamp(max = 15).squeeze(dim = -1) * mask_next
            time_loss = time_loss.sum()

            loss = time_loss + events_loss
        else:
            events_mask = torch.nn.functional.one_hot(events_next.long(), num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]
            intensity = (intensity * events_mask).sum(dim = -1)                # [batch_size, seq_len]
            integral = integral.sum(dim = -1)                                  # [batch_size, seq_len]
            time_loss = - torch.log(intensity + 1e-6) + integral               # [batch_size, seq_len]
            time_loss = time_loss.clamp(max = 15) * mask_next                  # [batch_size, seq_len]
            time_loss = time_loss.sum()
            
            loss = time_loss

        return loss, time_loss, events_loss, mask_next.sum()

    def mean_absolute_error(self, events_history, time_history, time_next, mask_next, mean, var):
        '''
        The input should be the original minibatch
        MAE evaluation part, dwg and fullynn exclusive
        '''
        def bisect_target(events_history, time_history, taus, mean, var):
            return self.evaluate(events_history, time_history, taus, mean, var) - \
                   torch.log(torch.tensor(self.mae_threshold, device = time_history.device))
            
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
        gap = (tau_pred - time_next).squeeze(-1) * mask_next
        gap_mean = torch.sum(torch.abs(gap)) / mask_next.sum()
        return gap_mean.item()

    def function_prober(self, data, resolution):
        (time, events, _, _, _), (mean, var) = data                               # 2 * [batch_size, seq_len + 1]
        events_history, _ = self.divide_history_and_next(events, unsqueeze = False)
                                                                               # [batch_size, seq_len]
        time_history, time_next = self.divide_history_and_next(time, unsqueeze = True)
                                                                               # [batch_size, seq_len, 1]
        intensity, integral, timestamp = self.submodel.intensity_integral(events_history, time_history, time_next, resolution, mean, var)
                                                                               # 3 * [batch_size, seq_len * resolution]
        
        return integral, intensity, timestamp

    def model_prober(self, data, resolution):
        (time, events, _, _, _), (mean, var) = data                               # 2 * [batch_size, seq_len + 1]
        events_history, _ = self.divide_history_and_next(events, unsqueeze = False)
                                                                               # [batch_size, seq_len]
        time_history, time_next = self.divide_history_and_next(time, unsqueeze = True)
                                                                               # [batch_size, seq_len, 1]
        expand_intensity, expand_integral, timestamp = \
            self.submodel.intensity_integral(events_history, time_history,
                                             time_next, resolution, mean, var,
                                             sum = False)                      # 3 * [batch_size, seq_len * resolution, num_events]
        
        intensity_and_integral_plot = {}
        additional_plot = []
        expand_intensity = torch.chunk(expand_intensity, self.num_events, dim = -1)
        expand_integral = torch.chunk(expand_integral, self.num_events, dim = -1)
        for idx, (expand_intensity_per_seq, expand_integral_per_seq) in enumerate(zip(expand_intensity, expand_integral)):
            intensity_and_integral_plot[f'event_intensity_{idx}'] = expand_intensity_per_seq.squeeze(dim = -1)
            intensity_and_integral_plot[f'event_integral_{idx}'] = expand_integral_per_seq.squeeze(dim = -1)

        return (intensity_and_integral_plot, additional_plot), timestamp


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
                                                                               # 2 * [batch_size, seq_len * resolution, num_events] + [batch_size, seq_len * resolution]
        probability_dist = intensity * torch.exp(-integral.sum(dim = -1, keepdim = True))
                                                                               # [batch_size, seq_len * resolution, num_events]
        cumulated_probability = probability_dist * timestamp.unsqueeze(dim = -1)
        cumulated_probability = rearrange(cumulated_probability, 'b (s r) ne -> b s r ne', r = resolution)
                                                                               # [batch_size, seq_len, resolution, num_events]
        probability = cumulated_probability.sum(dim = -2)                      # [batch_size, seq_len, num_events]
        probability_integral_sum = probability.sum(dim = -1)                   # [batch_size, seq_len]
        predicted_events = torch.argmax(probability, dim = -1)                  # [batch_size, seq_len]

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
        
        mae_per_event_pure_predict = self.mean_absolute_error_per_event_worker(events_history, predicted_events, time_history, time_next,
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
                                                                               # [batch_size, seq_len * resolution, num_events]
        cumulative_probability = probability_dist * timestamp.unsqueeze(dim = -1)
                                                                               # [batch_size, seq_len * resolution, num_events]
        cumulative_probability = rearrange(cumulative_probability, 'b (s r) n -> b s r n', r = resolution)
                                                                               # [batch_size, seq_len, resolution, num_events]
        cumulative_probability = cumulative_probability.sum(dim = -2)          # [batch_size, seq_len, num_events]
        cumulative_probability = cumulative_probability * events_mask          # [batch_size, seq_len, num_events]
        probability = cumulative_probability.sum(dim = -1)                     # [batch_size, seq_len]

        return probability

    def mean_absolute_error_per_event_worker(self, events_history, events_next, 
        time_history, time_next, probability_integral, resolution, mask, max_val, mean, var):
        '''
        The input should be the original minibatch
        MAE evaluation part, dwg and fullynn exclusive

        '''
        def bisect_target(taus):
            events_next_one_hot = torch.nn.functional.one_hot(events_next.long(), num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]
            p_xt = self.evaluate_per_event(events_history, time_history, events_next_one_hot, taus, resolution, mean, var)
                                                                               # [batch_size, seq_len]
            p_x = torch.sum(probability_integral * events_next_one_hot, dim = -1)
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
        
        [time, events, score, mask], (mean, var) = minibatch                   # 4 * [batch_size, seq_len + 1]
        loss, mae, time_loss, events_loss, the_number_of_events, constant = model(events, time, mean, var, mask)

        loss.backward()

        time_loss = time_loss.item() / the_number_of_events
        events_loss = events_loss.item() / the_number_of_events
        fact = score.sum().item() / the_number_of_events
        constant_norm = torch.linalg.norm(constant).detach().item() / the_number_of_events

        return time_loss, events_loss, fact, constant_norm

    def evaluation_step(model, minibatch, device):
        model.eval()

        [time, events, score, mask], (mean, var) = minibatch                   # 4 * [batch_size, seq_len + 1]
        _, mae, time_loss, events_loss, the_number_of_events, constant\
            = model(events, time, mean, var, mask, evaluation = True)

        time_loss = time_loss.item() / the_number_of_events
        events_loss = events_loss.item() / the_number_of_events
        fact = score.sum().item() / the_number_of_events
        constant_norm = torch.linalg.norm(constant).detach().item() / the_number_of_events

        return time_loss, events_loss, fact, mae, constant_norm

    def postprocess(input, procedure):
        if procedure == 'Training':
            return [input[0], input[1], input[0] - input[2], input[3]]
        else:
            return [input[0], input[1], input[0] - input[2], input[3], input[4]]

    def log_print_format(input, procedure):
        def format_training(input):
            format_dict = {}
            format_dict['absolute_loss'] = input[0]
            format_dict['events_loss'] = input[1]
            format_dict['relative_loss'] = input[2]
            format_dict['constant_norm'] = input[3]
            format_dict['num_format'] = {'absolute_loss': ':8.5f', 'relative_loss': ':8.5f', 'events_loss': ':8.5f', 'constant_norm': ':8.5f'}
            return format_dict

        def format_eva_and_test(input):
            format_dict = {}
            format_dict['absolute_loss'] = input[0]
            format_dict['events_loss'] = input[1]
            format_dict['relative_loss'] = input[2]
            format_dict['MAE'] = input[3]
            format_dict['constant_norm'] = input[4]
            format_dict['num_format'] = {'absolute_loss': ':8.5f', 'relative_loss': ':8.5f', 'events_loss': ':8.5f', 'MAE': ':2.8f', 'constant_norm': ':8.5f'}
            return format_dict

        return format_training(input) if procedure == 'Training' else format_eva_and_test(input)
    
    format_dict_length = 5
    
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
        return [evaluation_report[3], test_report[3]]
    
    metric_number = 2 # metric number is the length of the output of choose_metric