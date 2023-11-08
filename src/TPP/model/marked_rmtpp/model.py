from sklearn.metrics import f1_score
import torch, copy
import torch.nn.functional as F
from sklearn.metrics import f1_score, top_k_accuracy_score, accuracy_score

from src.TPP.model.utils import *
from src.TPP.model.marked_rmtpp.rmtpp import MRMTPPModule
from src.TPP.model.marked_rmtpp.plot import *


class MRMTPP(BasicModule):
    def __init__(self, device, input_size, hidden_size, history_encoder_layers, dropout, info_dict, 
                 output_size, limited_history_norm, time_scalar_min = 1e-20, epsilon = 1e-20,
                 probability_threshold = 0.5, survival_loss_during_training = False):
        super(MRMTPP, self).__init__()
        self.device = device
        self.num_events = info_dict['num_events']
        self.start_time = info_dict['t_0']
        self.end_time = info_dict['T']
        self.limited_history_norm = limited_history_norm
        self.probability_threshold = probability_threshold
        self.epsilon = epsilon
        self.survival_loss_during_training = survival_loss_during_training
        self.sample_rate = 32
        self.bisect_early_stop_threshold = 1e-5

        self.model = MRMTPPModule(input_size = input_size, hidden_size = hidden_size, history_encoder_layers = history_encoder_layers, 
                                  dropout = dropout, num_events = self.num_events, output_size = output_size, 
                                  limited_history_norm = limited_history_norm, time_scalar_min = time_scalar_min, device = device)


    def forward(self, task_name, *args, **kwargs):
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
        task_mapper = {
            'train': self.train_procedure,
            'evaluate': self.evaluate_procedure,
            'spearman_and_l1': self.get_spearman_and_l1,
            'mae_and_f1': self.get_mae_and_f1,
            'mae_e_and_f1': self.get_mae_e_and_f1,
            'graph': self.plot,
        }

        return task_mapper[task_name](*args, **kwargs)


    def divide_history_and_next(self, input):
        history, next = input[:, :-1].clone(), input[:, 1:].clone()
        return history, next                                                   # [batch_size, seq_len, 1] or [batch_size, seq_len]


    def remove_dummy_event_from_mask(self, mask):
        '''
        Remove the probability of the dummy event by mask.
        '''
        mask_without_dummy = torch.zeros_like(mask)                            # [batch_size, seq_len - 1]
        for idx, mask_per_seq in enumerate(mask):
            dummy_index = mask_per_seq.sum() - 1
            mask_without_dummy_per_seq = copy.deepcopy(mask_per_seq.detach())
            mask_without_dummy_per_seq[dummy_index] = 0
            mask_without_dummy[idx] = mask_without_dummy_per_seq
        
        return mask_without_dummy


    def train_procedure(self, events, time, mask, mean, var):
        events_history, events_next = self.divide_history_and_next(events)
                                                                               # [batch_size, seq_len]
        time_history, time_next = self.divide_history_and_next(time)           # [batch_size, seq_len]
        _, mask_next = self.divide_history_and_next(mask)                      # [batch_size, seq_len]
        mask_next_without_dummy = self.remove_dummy_event_from_mask(mask_next) # [batch_size, seq_len]
        events_next_without_dummy = events_next * mask_next_without_dummy      # [batch_size, seq_len]
        the_number_of_events = mask_next_without_dummy.sum().item()

        integral, intensity, _ = self.model(events_history, time_history, time_next, mean, var)
                                                                               # [batch_size, seq_len, num_events] * 2

        check_tensor(intensity)
        check_tensor(integral)

        # loss_without_dummy = time_loss_without_dummy + events_loss_without_dummy
        # time_loss_without_dummy = \sum_{t_i}{\lambda^*(t_i)} + \int_{t_0}^{t_n}{\lambda^*(\tau)d\tau}
        # event_loss_without_dummy = \sum{x_i}{CrossEntropyLoss(\hat{x_i}, x_i)} for all real-world events.
        time_loss_without_dummy, events_loss_without_dummy = \
                   self.loss_function(intensity, integral, events_next_without_dummy, mask_next_without_dummy)
        
        probability_survival = 0
        if self.survival_loss_during_training:
            # survival loss: \int_{t_n}^{T}{\lambda^*(\tau)d\tau}
            dummy_event_index = mask_next.sum(dim = -1) - 1                    # [batch_size]
            probability_survival = integral.sum(dim = -1).gather(index = dummy_event_index.unsqueeze(dim = -1), dim = -1).sum()

        training_loss = time_loss_without_dummy + probability_survival

        return training_loss, time_loss_without_dummy, events_loss_without_dummy, the_number_of_events


    def evaluate_procedure(self, events, time, mask, mean, var):
        events_history, events_next = self.divide_history_and_next(events)     # [batch_size, seq_len]
        time_history, time_next = self.divide_history_and_next(time)           # [batch_size, seq_len]
        _, mask_next = self.divide_history_and_next(mask)                      # [batch_size, seq_len]
        mask_next_without_dummy = self.remove_dummy_event_from_mask(mask_next) # [batch_size, seq_len]
        events_next_without_dummy = events_next * mask_next_without_dummy      # [batch_size, seq_len]
        the_number_of_events = mask_next_without_dummy.sum().item()

        '''
        Calculating MAE here.
        '''
        mae, pred_time = self.mean_absolute_error(events_history, time_history, time_next, mask_next_without_dummy, mean, var)
                                                                               # [batch_size, seq_len] * 2

        integral_time_next, intensity_time_next, _ = self.model(events_history, time_history, time_next, mean, var)
                                                                               # [batch_size, seq_len, num_events]
        integral_pred_time, intensity_pred_time, _ = self.model(events_history, time_history, pred_time, mean, var)
                                                                               # [batch_size, seq_len, num_events]
        check_tensor(intensity_time_next)
        check_tensor(integral_time_next)
        check_tensor(intensity_pred_time)
        check_tensor(integral_pred_time)
        
        # NLL and event loss at time_next.
        time_loss_time_next_without_dummy, events_loss_time_next_without_dummy = \
                   self.loss_function(intensity_time_next, integral_time_next, events_next_without_dummy, mask_next_without_dummy)
        
        '''
        macro-F1 of evaluating event predictions.
        '''
        predicted_events = torch.argmax(intensity_pred_time, dim = -1)[mask_next_without_dummy == 1]
        events_true = events_next_without_dummy[mask_next_without_dummy == 1]
        predicted_events, events_true = move_from_tensor_to_ndarray(predicted_events, events_true)
                                                                       # [batch_size, seq_len] * 2
        f1 = f1_score(y_pred = predicted_events, y_true = events_true, average = 'macro')

        # survival loss: \int_{t_n}^{T}{\lambda^*(\tau)d\tau}
        dummy_event_index = mask_next.sum(dim = -1) - 1                        # [batch_size]
        time_loss_survival = integral_time_next.sum(dim = -1).gather(index = dummy_event_index.unsqueeze(dim = -1), dim = -1).mean()

        return time_loss_time_next_without_dummy, time_loss_survival, events_loss_time_next_without_dummy, \
               mae, f1, the_number_of_events


    def loss_function(self, intensity, integral, events_next, mask_next):
        # temporal point process loss
        # intensity shape: [batch, seq_length]
        # so does tensor mask.

        # Time loss, also the training loss.
        intensity_mask = torch.nn.functional.one_hot(events_next.long(), num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]
        selected_intensity = (intensity * intensity_mask).sum(dim = -1)        # [batch_size, seq_len]
        sum_of_integral = integral.sum(dim = -1)                               # [batch_size, seq_len]

        time_loss = -torch.log(selected_intensity + self.epsilon) + sum_of_integral
                                                                               # [batch_size, seq_len]
        time_loss = time_loss * mask_next
        time_loss = time_loss.sum()

        log_probability_for_each_event = torch.log(intensity + self.epsilon)   # [batch_size, seq_len, num_events]
        events_probability = torch.nn.functional.softmax(log_probability_for_each_event, dim = -1)
                                                                               # [batch_size, seq_len, num_events]
        events_loss = torch.nn.functional.cross_entropy(rearrange(events_probability, 'b s ne -> b ne s'), \
                                                        events_next.long(), reduction = 'none')
                                                                               # [batch_size, seq_len]
        events_loss = events_loss * mask_next                                  # [batch_size, seq_len]
        events_loss = events_loss.sum()

        return time_loss, events_loss


    def mean_absolute_error_and_f1(self, events_history, time_history, events_next, time_next, mask_history, mask_next, mean, var):
        mae, pred_time = self.mean_absolute_error(events_history, time_history, time_next, mask_next, mean, var)
        _, _, mark, _ = self.model(events_history, time_history, pred_time, mean, var)

        predicted_events = torch.argmax(mark, dim = -1)[mask_next == 1]        # [batch_size, seq_len]
        events_true = events_next[mask_next == 1]                              # [batch_size, seq_len]

        predicted_events, events_true = move_from_tensor_to_ndarray(predicted_events, events_true)
        f1 = f1_score(y_pred = predicted_events, y_true = events_true, average = 'macro')

        return mae, f1


    def mean_absolute_error(self, events_history, time_history, time_next, mask_next, mean, var):
        '''
        The input should be the original minibatch
        MAE evaluation part, dwg and fullynn exclusive
        '''
        dist = torch.distributions.uniform.Uniform(torch.tensor(0.0), torch.tensor(1.0))
        probability_threshold = dist.sample((self.sample_rate, *time_next.shape))
                                                                               # [sample_rate, batch_size, seq_len]
        probability_threshold = probability_threshold.to(self.device)

        def evaluate(taus):
            integral, _, _ = self.model(events_history, time_history, taus, mean, var)
                                                                               # [sample_rate, batch_size, seq_len, num_events]
            integral = integral.sum(dim = -1)                                  # [sample_rate, batch_size, seq_len]

            return integral

        def bisect_target(taus):
            return evaluate(taus) + torch.log(1 - probability_threshold)
            
        def median_prediction(l, r):
            index = 0
            while True:
                c = (l + r)/2
                v = bisect_target(c)
                l = torch.where(v < 0, c, l)
                r = torch.where(v >= 0, c, r)
                index += 1
                if (l - r).abs().max() < self.bisect_early_stop_threshold:
                    break
                if index > 50:
                    break

            return (l + r)/2
        
        l = 0.0001*torch.ones_like(probability_threshold, dtype = torch.float32)
                                                                               # [sample_rate, batch_size, seq_len]
        r = 1e6*torch.ones_like(probability_threshold, dtype = torch.float32)  # [sample_rate, batch_size, seq_len]
        tau_pred = median_prediction(l, r)                                     # [sample_rate, batch_size, seq_len]

        '''
        integral, intensity, _, _ = self.model(events_history, time_history, tau_pred, mean, var)
                                                                               # [sample_rate, batch_size, seq_len] * 2
        probability_of_each_event = intensity * torch.exp(-integral)           # [sample_rate, batch_size, seq_len]
        tau_pred = (tau_pred * probability_of_each_event).sum(dim = 0)         # [batch_size, seq_len]
        gap = torch.abs(tau_pred - time_next) * mask_next                      # [batch_size, seq_len]
        '''

        tau_pred = tau_pred.mean(dim = 0)                                      # [batch_size, seq_len]
        mae = torch.abs(tau_pred - time_next) * mask_next                      # [batch_size, seq_len]

        return mae, tau_pred


    def mean_absolute_error_e(self, time_history, time_next, events_history, events_next, mask_history, mask_next, mean, var):
        '''
        The precedure resembles the compute_integral_unbiased() but the output of small step MC takes would
        be recorded as part of the output.
        '''
        self.eval()

        '''
        set a relatively large number as the infinity and decide resolution based on this large value and
        the memory_ceiling.
        '''
        if mean == 0 and var == 1:
            max_ = time_next.mean() + 10 * time_next.var()
        else:
            max_ = mean + 10 * var

        if mean == 0:
            resolution_between_events = max(min(int(time_next.mean().item() // 0.005), 500), 10)
        else:
            resolution_between_events = max(min(int(mean // 0.005), 500), 10)
        
        max_ = min(1e6, max_)
        time_next_inf = torch.ones_like(time_history, device = self.device) * max_
                                                                               # [batch_size, seq_len]
        resolution_inf = max(int(max_ // 0.005), 100)
        
        expanded_integral_all_events_to_inf, expanded_intensity_all_events_to_inf, timestamp = \
            self.model.integral_intensity_time_next_2d(events_history, time_history, time_next_inf, mask_history, resolution_inf)
                                                                               # 2 * [batch_size, seq_len, resolution_inf, num_events]

        expanded_integral_sum_over_events_to_inf = expanded_integral_all_events_to_inf.sum(dim = -1, keepdim = True)
                                                                               # [batch_size, seq_len, resolution_inf, 1]
        expanded_probability_inf = expanded_intensity_all_events_to_inf * torch.exp(-expanded_integral_sum_over_events_to_inf)
                                                                               # [batch_size, seq_len, resolution_inf, num_events]
        probability_integral_to_inf = self.model.integration_estimator(expanded_probability_inf, timestamp, resolution_inf)[:, :, -1, :]
                                                                               # [batch_size, seq_len, num_events]
        probability_integral_sum = probability_integral_to_inf.sum(dim = -1)   # [batch_size, seq_len]
        predicted_events = torch.argmax(probability_integral_to_inf, dim = -1) # [batch_size, seq_len]

        f1 = []
        top_k_acc = []
        for (ground_truth_per_seq, probability_integral_per_seq) in zip(events_next, probability_integral_to_inf):
            ground_truth_per_seq, probability_integral_per_seq = \
                move_from_tensor_to_ndarray(ground_truth_per_seq, probability_integral_per_seq)

            f1.append(f1_score(y_true = ground_truth_per_seq,
                               y_pred = np.argmax(probability_integral_per_seq, axis = -1), average = 'macro'))
            
            top_k_acc_per_seq = []
            if self.num_events > 2:
                for k in range(1, self.num_events):
                    top_k_acc_per_seq.append(
                        top_k_accuracy_score(y_true = ground_truth_per_seq,
                                             y_score = probability_integral_per_seq,
                                             k = k,
                                             labels = np.arange(self.num_events))
                    )
            else:
                top_k_acc_per_seq.append(
                    accuracy_score(
                        y_true = ground_truth_per_seq,
                        y_pred = torch.argmax(probability_integral_per_seq, dim = -1)
                    )
                )
            top_k_acc.append(top_k_acc_per_seq)
        # F1:        [batch_size]
        # top_k_acc: [batch_size, num_events]

        tau_pred_all_event = self.prediction_with_all_event_types(events_history, time_history, \
                                                                  mask_history, probability_integral_to_inf, \
                                                                  resolution_between_events, max_, mean, var)
                                                                               # [batch_size, seq_len, num_events]

        predicted_event_mask = F.one_hot(predicted_events.long(), num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]
        event_next_mask = F.one_hot(events_next.long(), num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]

        mae_per_event_pure_predict = torch.abs((tau_pred_all_event * predicted_event_mask).sum(dim = -1) - time_next) * mask_next
                                                                               # [batch_size, seq_len]
        mae_per_event = torch.abs((tau_pred_all_event * event_next_mask).sum(dim = -1) - time_next) * mask_next
                                                                               # [batch_size, seq_len]

        mae_per_event_pure_predict_avg = torch.sum(mae_per_event_pure_predict, dim = -1) / mask_next.sum(dim = -1)
        mae_per_event_avg = torch.sum(mae_per_event, dim = -1) / mask_next.sum(dim = -1)
        
        return f1, top_k_acc, probability_integral_sum, tau_pred_all_event, (mae_per_event_pure_predict_avg, mae_per_event_avg), \
               (mae_per_event_pure_predict, mae_per_event)


    def prediction_with_all_event_types(self, events_history, time_history, mask_history, p_x, resolution, max_val, mean, var):
        '''
        The input should be the original minibatch
        MAE evaluation part, dwg and fullynn exclusive
        '''
        # Preprocess
        batch_size, seq_len = time_history.shape
        dist = torch.distributions.uniform.Uniform(torch.tensor(0.0), torch.tensor(1.0))
        probability_threshold = dist.sample((self.sample_rate, batch_size, seq_len, self.num_events))
                                                                               # [sample_rate, batch_size, seq_len, num_events]
        probability_threshold = probability_threshold.to(self.device)
        p_x = p_x.unsqueeze(dim = 0)                                           # [1, batch_size, seq_len, num_events]

        def evaluate_all_event(taus):
            expanded_integral_across_events, expanded_intensity_across_events, timestamp = \
                self.model.integral_intensity_time_next_3d(events_history, time_history, taus, mask_history, resolution, num_dimension_prior_batch = 1)
                                                                               # 2 * [sample_rate, batch_size, seq_len, num_events, resolution, num_events] + [sample_rate, batch_size, seq_len, num_events, resolution]
            expanded_integral_sum_across_events = expanded_integral_across_events.sum(dim = -1)
                                                                               # [sample_rate, batch_size, seq_len, num_events, resolution]
            intensity_event_mask = torch.diag(torch.ones(self.num_events, device = self.device))
                                                                               # [num_events, num_events]
            intensity_event_mask = rearrange(intensity_event_mask, f'ne ne1 -> {"() " * (len(expanded_intensity_across_events.shape) - 3)}ne () ne1')
                                                                               # [sample_rate, batch_size, seq_len, num_events, resolution, num_events]
            expanded_intensity_per_event = (expanded_intensity_across_events * intensity_event_mask).sum(dim = -1)
                                                                               # [sample_rate, batch_size, seq_len, num_events, resolution]
            expanded_probability_per_event = expanded_intensity_per_event * torch.exp(-expanded_integral_sum_across_events)
                                                                               # [sample_rate, batch_size, seq_len, num_events, resolution]
            probability = self.model.integration_probability_estimator(expanded_probability_per_event, timestamp, resolution)[..., -1]
                                                                               # [sample_rate, batch_size, seq_len, num_events]
            return probability
    
        def bisect_target(taus):
            p_xt = evaluate_all_event(taus)                                    # [sample_rate, batch_size, seq_len, num_events]
            p_t_x = p_xt / p_x                                                 # [sample_rate, batch_size, seq_len, num_events]
            p_gap = p_t_x - probability_threshold                              # [sample_rate, batch_size, seq_len, num_events]

            return p_gap
            
        def median_prediction(l, r):
            index = 0
            while True:
                c = (l + r)/2
                v = bisect_target(c)
                l = torch.where(v < 0, c, l)
                r = torch.where(v >= 0, c, r)
                index += 1
                if (l - r).abs().max() < self.bisect_early_stop_threshold:
                    break
                if index > 50:
                    break

            return (l + r)/2
        
        l = 0.0001*torch.ones((self.sample_rate, batch_size, seq_len, self.num_events), dtype = torch.float32, device = self.device)
                                                                               # [sample_rate, batch_size, seq_len, num_events]
        r = max_val*torch.ones((self.sample_rate, batch_size, seq_len, self.num_events), dtype = torch.float32, device = self.device)
                                                                               # [sample_rate, batch_size, seq_len, num_events]
        tau_pred = median_prediction(l, r)                                     # [sample_rate, batch_size, seq_len, num_events]

        '''
        integral_of_each_event, intensity_of_each_event, _ \
            = self.model.integral_intensity_time_next_3d(events_history, time_history, tau_pred, \
                                                         mask_history, resolution, num_dimension_prior_batch = 1)
                                                                               # 2 * [sample_rate, batch_size, seq_len, num_events, integration_sample_rate, num_events]
        integral_sum_of_each_event = integral_of_each_event.sum(dim = -1)
                                                                               # [sample_rate, batch_size, seq_len, num_events, resolution]
        intensity_event_mask = torch.diag(torch.ones(self.num_events, device = self.device))
                                                                               # [num_events, num_events]
        intensity_event_mask = rearrange(intensity_event_mask, f'ne ne1 -> {"() " * (len(intensity_of_each_event.shape) - 3)}ne () ne1')
                                                                               # [sample_rate, batch_size, seq_len, num_events, resolution, num_events]
        expanded_intensity_per_event = (intensity_of_each_event * intensity_event_mask).sum(dim = -1)
                                                                               # [sample_rate, batch_size, seq_len, num_events, resolution]
        expanded_probability_per_event = expanded_intensity_per_event * torch.exp(-integral_sum_of_each_event)
                                                                               # [sample_rate, batch_size, seq_len, num_events, resolution]
        probability_per_event = expanded_probability_per_event[..., -1]        # [sample_rate, batch_size, seq_len, num_events]
        tau_pred = (tau_pred * probability_per_event).sum(dim = 0)             # [batch_size, seq_len, num_events]
        '''
        tau_pred = tau_pred.mean(dim = 0)                                      # [batch_size, seq_len, num_events]
        
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

        data, timestamp = self.model.model_probe_function(events_history, time_history, time_next, \
                                                          mask_history, mask_next, opt.resolution)
        f1_2, top_k, probability_sum, tau_pred_all_event, maes_avg, maes \
            = self.mean_absolute_error_e(time_history, time_next, events_history, events_next, mask_history, mask_next, mean, var)

        '''
        Append additional info into the data dict.
        '''
        data['events_next'] = events_next
        data['time_next'] = time_next
        data['mask_next'] = mask_next
        data['f1_after_time_pred'] = f1_1
        data['mae_before_event'] = mae
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
        mae = move_from_tensor_to_ndarray(mae)

        return mae, f1_1


    def get_mae_e_and_f1(self, input_data, opt):
        input_time, input_events, input_intensity, mask, mean, var = self.extract_plot_data(input_data)
        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]

        f1_2, top_k, probability_sum, tau_pred_all_event, maes_avg, maes \
            = self.mean_absolute_error_e(time_history, time_next, events_history, \
                                         events_next, mask_history, mask_next, mean, var)
        
        _, maes, probability_sum = move_from_tensor_to_ndarray(*maes, probability_sum)

        return maes, f1_2, probability_sum


    def train_step(model, minibatch, device):
        model.train()
        
        [time, events, score, mask], (mean, var) = minibatch                   # 4 * [batch_size, seq_len + 1]
        loss, time_loss_without_dummy, events_loss_without_dummy, the_number_of_events \
            = model('train', events, time, mask, mean, var)

        loss.backward()

        time_loss_without_dummy = time_loss_without_dummy.item() / the_number_of_events
        events_loss_without_dummy = events_loss_without_dummy.item() / the_number_of_events
        fact = score.sum().item() / the_number_of_events

        return time_loss_without_dummy, fact, events_loss_without_dummy


    def evaluation_step(model, minibatch, device):
        model.eval()

        [time, events, score, mask], (mean, var) = minibatch                   # 4 * [batch_size, seq_len + 1]
        time_loss_time_next_without_dummy, time_loss_survival, events_loss_time_next_without_dummy, \
        mae, f1, the_number_of_events = model('evaluate', events, time, mask, mean, var)

        time_loss_time_next_without_dummy = time_loss_time_next_without_dummy.item() / the_number_of_events
        time_loss_survival = time_loss_survival.item()
        fact = score.sum().item() / the_number_of_events
        events_loss_time_next_without_dummy = events_loss_time_next_without_dummy.item() / the_number_of_events
        mae = mae.sum().item() / the_number_of_events

        return time_loss_time_next_without_dummy, time_loss_survival, fact, events_loss_time_next_without_dummy, mae, f1


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
            return [input[0], input[1], input[0] - input[2], input[3], input[4], input[5]]
        
        return (train_postprocess(input) if procedure == 'Training' else test_postprocess(input))


    format_dict_length = 6

    
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
            format_dict['absolute_NLL_loss'] = input[0]
            format_dict['avg_survival_loss'] = input[1]
            format_dict['relative_NLL_loss'] = input[2]
            format_dict['events_loss'] = input[3]
            format_dict['mae'] = input[4]
            format_dict['f1_pred_at_pred_time'] = input[5]
            format_dict['num_format'] = {'absolute_NLL_loss': ':6.5f', 'avg_survival_loss': ':6.5f', \
                                         'relative_NLL_loss': ':6.5f', 'events_loss': ':6.5f',
                                         'mae': ':2.8f', 'f1_pred_at_pred_time': ':6.5f'}
            return format_dict
        
        return (train_log_print_format(input) if procedure == 'Training' else test_log_print_format(input))
    

    def choose_metric(evaluation_report_format_dict, test_report_format_dict):
        '''
        [relative loss on evaluation dataset, relative loss on test dataset, event loss on test dataset]
        '''
        return [evaluation_report_format_dict['absolute_NLL_loss'], 
                test_report_format_dict['absolute_NLL_loss']], \
               ['evaluation_absolute_loss', 'test_absolute_loss']


    metric_number = 2 # metric number is the length of the output of choose_metric