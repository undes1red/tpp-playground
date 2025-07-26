import torch, copy
import torch.nn.functional as F
from einops import rearrange, repeat, reduce, pack
from sklearn.metrics import f1_score, roc_auc_score

from src.toolbox.misc import check_tensor, move_from_tensor_to_ndarray, pack_one_value_to_dict, argument_check
from src.toolbox.integration import approximate_integration
from src.toolbox.metrics import L1_distance_between_two_funcs

from src.TPP.model.thp.sample import sample_time
from src.TPP.model.basic_tpp_model import memory_ceiling, BasicModel
from src.TPP.model.thp.plot import *
from src.TPP.model.thp.submodel import THP
from src.TPP.model.utils import predict_event, decide_resolution_inf_and_resolution_between_events, get_f1_and_top_k_acc_in_mae_e


class THPWrapper(BasicModel):
    '''
    Transformer Hawkes Process (THP), proposed by Zuo et al. at ICML 2020.
    '''
    def __init__(self, opt, device, d_input = 64, d_rnn = 64, d_hidden = 256, n_layers = 3,
                 n_head = 3, d_qk = 64, d_v = 64, dropout = 0.1, sample_rate = 32,
                 integration_sample_rate = 100, epsilon = 1e-20, history_time_offset = 1.0, mae_step = 32, mae_e_step = 32,
                 survival_loss_during_training = True):
        '''
        This function creates a THP model.
        
        ### Args
            * ```int``` d_input
            The dimension of the Transformer input tensor.
            * ```int``` d_hidden
              The dimension of the FFN module in the Transformer.  
            * ```int``` n_layers
              The number of self attention + FFN layers in the Transformer.  
            * ```int``` n_head
              The number of head in self attention.
            * ```int``` d_qk
              The dimension of matrices Q and K.
            * ```int``` d_v
              The dimension of metrix V.
            * ```float``` dropout
              Dropout rate for the history encoder.
            * ```int``` d_rnn
              The dimension of RNN's hidden state.
            * ```float``` history_time_offset
              THP scales the input time by dividing it with the time interval from start to the latest event in history.
              This can cause issues when there is no event in history-the input time will be divided by 0.
              So, we add this offset to the time interval to avoid this divided-by-0 case.
            * ```namespace``` opt
              Model arguments.
            * ```torch.device``` device
              Running models on GPU or CPU?
            * ```float``` epsilon
              Shiftting the calculated intensity function and probability distribution by a little bit so that ```torch.log()``` won't fail.
            * ```int``` sample_rate
              This tells how many time samples from the time distribution are needed for one time prediction.
            * ```int``` mae_step
              This parameter controls how many samples are generated in one shot when sampling from p(t).
            * ```int``` mae_e_step
              This parameter controls how many samples are generated in one shot when sampling from all p(t|m)s at the same time.
              mae_step and mae_e_step are useful when you cannot get sample_rate time samples from time distributions because of insufficient GPU memory.
            * ```int``` integration_sample_rate
              The number of interpolated points in a time interval between two adjoint events for integration estimation.
              The number of interpolated points counts the start and end point of the interval.
            * ```bool``` survival_loss_during_training
              When true, the training loss includes the integral between the last observed event to the end time T. Most of time this argument should be true.
        '''
        super(THPWrapper, self).__init__()
        self.device = device
        self.compile_or_not = opt.compile
        self.num_events = opt.info_dict['num_events']
        self.start_time = opt.info_dict['t_0']
        self.end_time = opt.info_dict['T']
        self.epsilon = epsilon
        self.history_time_offset = history_time_offset
        self.survival_loss_during_training = survival_loss_during_training
        self.integration_sample_rate = integration_sample_rate
        self.sample_rate = sample_rate
        self.mae_step = mae_step
        self.mae_e_step = mae_e_step
        self.bisect_early_stop_threshold = 1e-4
        self.max_step = 50

        self.model = THP(num_events = self.num_events, d_input = d_input, d_rnn = d_rnn, d_hidden = d_hidden, \
                         n_layers = n_layers, n_head = n_head, d_qk = d_qk, d_v = d_v, dropout = dropout, \
                         integration_sample_rate = integration_sample_rate, device = device, \
                         history_time_offset = history_time_offset)


    def divide_history_and_next(self, input):
        '''
        Extract the history and prediction sequences from the input sequence.
        
        ### Args
            * ```torch.tensor``` input
              shape: [batch_size, seq_len + 1]
              The input sequence.
        
        ### Outputs
            * ```torch.tensor``` input_history
              shape: [batch_size, seq_len]
              The history sequence extracted from the original input.
            * ```torch.tensor``` input_next
              shape: [batch_size, seq_len]
              The history sequence extracted from the original input.
        '''
        input_history, input_next = input[:, :-1].clone(), input[:, 1:].clone()
        return input_history, input_next


    def remove_dummy_event_from_mask(self, mask):
        '''
        Remove the dummy event by altering the mask.

        ### Args
            * ```torch.tensor``` mask
              shape: [batch_size, seq_len]
              The input mask tensor.
        
        ### Outputs
            * ```torch.tensor``` mask_without_dummy
              shape: [batch_size, seq_len]
              The output mask tensor with the last unmask event in each sequence removed.
        '''
        mask_without_dummy = torch.zeros_like(mask)                            # [batch_size, seq_len - 1]
        for idx, mask_per_seq in enumerate(mask):
            dummy_index = mask_per_seq.sum() - 1
            mask_without_dummy_per_seq = copy.deepcopy(mask_per_seq.detach())
            mask_without_dummy_per_seq[dummy_index] = 0
            mask_without_dummy[idx] = mask_without_dummy_per_seq
        
        return mask_without_dummy


    def forward(self, task_name, *args, **kwargs):
        '''
        The entrance of the THP.
        
        ### Args
            * ```str``` task_name
              The name of the executed task.
        '''
        task_mapper = {
            'train': self.train_procedure,
            'evaluate': self.evaluate_procedure,
            'spearman_and_l1': self.get_spearman_and_l1,
            'mae_and_f1': self.get_mae_and_f1,
            'mae_e_and_f1': self.get_mae_e_and_f1,
            'which_event_occurs_first': self.get_which_event_first,
            'samples_from_et': self.samples_from_et,

            # Figure Drawing.
            'intensity': self.figure_intensity,
            'integral': self.figure_integral,
            'probability': self.figure_probability,
            'debug': self.figure_debug,

            # For CPPOD, should be used with the od_generic dataloader.
            'cppod_evaluation': self.cppod_evaluation,
            'cppod_commission_evaluation': self.cppod_commission_evaluation
        }

        return task_mapper[task_name](*args, **kwargs)


    def train_procedure(self, time, events, mask, mean, std):
        '''
        THP's forwardpropagation function for training.
        
        ### Args
            * ```torch.tensor``` time
              shape: ```[batch_size, seq_len + 1]```
              Time sequence for training.
            * ```torch.tensor``` events
              shape: ```[batch_size, seq_len + 1]```
              Event sequence for training.
            * ```torch.tensor``` mask
              shape: ```[batch_size,, seq_len + 1]```
              Mask sequence. Events whose corresponding mask is 0 are dummy events.
            * ```float``` mean
            * ```float``` std
              Used for input time scaling.

        ### Outputs
            * ```torch.tensor``` loss
              shape: ```[1]```
              The sum of NLL loss L = -log \\frac{\\partial \\Lambda^*(m, t)}{\\partial t} + \\Lambda^*(m, t) at each happened event (the dummy event at end time T included).
            * ```torch.tensor``` neg_log_likeli_loss_without_dummy
              shape: ```[1]```
              The sum of NLL loss L = -log \\frac{\\partial \\Lambda^*(m, t)}{\\partial t} + \\Lambda^*(m, t) at each happened event (the dummy event at end time T excluded).
            * ```torch.tensor``` marker_loss_without_dummy
              shape: ```[1]```
              The sum of the event loss: L = -log \\frac{\\lambda^*(m, t)}{\\sum_{n \\in M}{\\lambda^*(n, t)}} where m is the mark of the real event.
            * ```int``` the_number_of_events
              The number of legit events.
        '''
        time_history, time_next = self.divide_history_and_next(time)           # [batch_size, seq_len] * 2
        events_history, events_next = self.divide_history_and_next(events)     # [batch_size, seq_len] * 2
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len] * 2

        mask_next_without_dummy = self.remove_dummy_event_from_mask(mask_next) # [batch_size, seq_len]
        events_next_without_dummy = events_next * mask_next_without_dummy      # [batch_size, seq_len]
        the_number_of_events = mask_next_without_dummy.sum().item()

        integral_all_events, intensity_all_events = self.model(time_history, time_next, events_history, mask_history)
                                                                               # [batch_size, seq_len, num_events]
        # L = \\sum_{i}{\\lambda^_k*(t_i)} + \\int_{t_0}^{t_n}{\\sum_{k}{\\lambda^*_k(\\tau)}d\\tau}
        neg_log_likeli_loss_without_dummy, marker_loss_without_dummy = self.negative_log_likelihood_and_event_loss(
             intensity_all_events = intensity_all_events, integral_all_events = integral_all_events,\
             events_next = events_next_without_dummy, mask_next = mask_next_without_dummy
        )

        loss_survival = 0
        if self.survival_loss_during_training:
            # survival_loss = \\int_{t_n}^{T}{\\sum_{k}{\\lambda^*_k(\\tau)}d\\tau}
            dummy_event_index = mask_next.sum(dim = -1) - 1                        # [batch_size]
            integral_survival = integral_all_events.sum(dim = -1).gather(index = dummy_event_index.unsqueeze(dim = -1), dim = -1)
                                                                                   # [batch_size, 1]
            loss_survival = integral_survival.sum()
        
        loss = neg_log_likeli_loss_without_dummy + loss_survival

        return loss, neg_log_likeli_loss_without_dummy, marker_loss_without_dummy, the_number_of_events


    @torch.inference_mode()
    def evaluate_procedure(self, time, events, mask, mean, std):
        '''
        THP's forwardpropagation function for evaluation.
        
        ### Args
            * ```torch.tensor``` time
              shape: ```[batch_size, seq_len + 1]```
              Time sequencalculatesce for training.
            * ```torch.tensor``` events
              shape: ```[batch_size, seq_len + 1]```
              Event sequence for training.
            * ```torch.tensor``` mask
              shape: ```[batch_size,, seq_len + 1]```
              Mask sequence. Events whose corresponding mask is 0 are dummy events.
            * ```float``` mean
            * ```float``` std
              Used for input time scaling.

        ### Outputs
            * ```torch.tensor``` log_likeli_loss_time_next_without_dummy
              shape: ```[1]```
              The sum of NLL loss L = -log \\frac{\\partial \\Lambda^*(m, t)}{\\partial t} + \\Lambda^*(m, t) at each happened event.
            * ```torch.tensor``` loss_survival
              shape: ```[1]```
              The sum of the integration \\Lambda^*(m, t) from the last observed event to the end time T.
            * ```torch.tensor``` marker_loss_time_next_without_dummy
              shape: ```[1]```
              The sum of the event loss: L = -log \\frac{\\lambda^*(m, t)}{\\sum_{n \\in M}{\\lambda^*(n, t)}} where m is the mark of the real event.
            * ```float``` mae
              The average error between predicted time and real time.
            * ```float``` f1
              The prediction accuracy of predicted marks.
            * ```int``` the_number_of_events
              The number of legit events.
        '''
        time_history, time_next = self.divide_history_and_next(time)           # [batch_size, seq_len] * 2
        events_history, events_next = self.divide_history_and_next(events)     # [batch_size, seq_len] * 2
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]

        mask_next_without_dummy = self.remove_dummy_event_from_mask(mask_next) # [batch_size, seq_len]
        events_next_without_dummy = events_next * mask_next_without_dummy      # [batch_size, seq_len]
        the_number_of_events = mask_next_without_dummy.sum().item()

        # Event loss. This loss should not be counted into the backward loss
        mae, f1, _ = self.mean_absolute_error_and_f1(events_history, time_history, 
                                                     events_next, time_next, 
                                                     mask_history, mask_next_without_dummy, mean, std)
        mae = mae.sum().item() / the_number_of_events

        integral_all_events_time_next, intensity_all_events_time_next = self.model(time_history, time_next, events_history, mask_history)
                                                                               # 2 * [batch_size, seq_len, num_events]

        # L = \\sum_{i}{\\lambda^_k*(t_i)} + \\int_{t_0}^{t_n}{\\sum_{k}{\\lambda^*_k(\\tau)}d\\tau}
        log_likeli_loss_time_next_without_dummy, marker_loss_time_next_without_dummy = self.negative_log_likelihood_and_event_loss(
             intensity_all_events = intensity_all_events_time_next, integral_all_events = integral_all_events_time_next,\
             events_next = events_next_without_dummy, mask_next = mask_next_without_dummy
        )

        # survival_loss = \\int_{t_n}^{T}{\\sum_{k}{\\lambda^*_k(\\tau)}d\\tau}
        dummy_event_index = mask_next.sum(dim = -1) - 1                        # [batch_size]
        integral_survival = integral_all_events_time_next.sum(dim = -1).gather(index = dummy_event_index.unsqueeze(dim = -1), dim = -1)
                                                                               # [batch_size, 1]
        loss_survival = integral_survival.sum()

        return log_likeli_loss_time_next_without_dummy, loss_survival, marker_loss_time_next_without_dummy, \
               mae, f1, the_number_of_events


    def negative_log_likelihood_and_event_loss(self, intensity_all_events, integral_all_events, events_next, mask_next):
        '''
        This function computes the NLL loss at each legit event in events_next.
    
        ### Args
            * ```torch.tensor``` intensity_all_events
              shape: ```[batch_size, seq_len, num_events]```
              intensity values at t_i.
            * ```torch.tensor``` integral_all_events
              shape: ```[batch_size, seq_len, num_events]```
              intensity integral from t_{i - 1} to t_{i} (t_0 = 0).
            * ```torch.tensor``` events_next
              shape: ```[batch_size, seq_len]```
              The mark of the events that we need to predict.
            * ```torch.tensor``` mask_next
              shape: ```[batch_size, seq_len]```
              Needed mask to mask out unneeded loss values.
        
        ### Outputs
            * ```torch.tensor``` mtpp_nll_loss, 
              shape: ```[1]```
              The sum of NLL loss on all event.
            * ```torch.tensor``` events_loss
              shape: ```[1]```
              The sum of the event loss: L = -log \\frac{\\lambda^*(m, t)}{\\sum_{n \\in M}{\\lambda^*(n, t)}} where m is the mark of the real event.
        '''
        type_mask = F.one_hot(events_next.long(), num_classes = self.num_events)

        # MTPP loss function
        intensity = torch.sum(intensity_all_events * type_mask, dim = -1)      # [batch_size, seq_len]
        log_intensity = torch.log(intensity + self.epsilon) * mask_next        # [batch_size, seq_len]
        intensity_integral = integral_all_events.sum(dim = -1)                 # [batch_size, seq_len]
        ll = -log_intensity + intensity_integral                               # [batch_size, seq_len]
        mtpp_nll_loss = torch.sum(ll)

        # Event loss function. Only for evaluation, do not use this loss as a part of the training loss.
        events_prediction_probability = torch.log(intensity_all_events + self.epsilon)
                                                                               # [batch_size, seq_len, num_events]
        events_prediction_probability = F.softmax(events_prediction_probability, dim = -1)
                                                                               # [batch_size, seq_len, num_events]
        events_prediction_probability = rearrange(events_prediction_probability, 'b s ne -> b ne s')
                                                                               # [batch_size, num_events, seq_len]
        events_loss = F.cross_entropy(input = events_prediction_probability, target = events_next.long(), reduction = 'none')
                                                                               # [batch_size, seq_len]
        events_loss = (events_loss * mask_next).sum()

        return mtpp_nll_loss, events_loss


    sample_time = sample_time


    @torch.inference_mode()
    def evaluate_f1(self, intensity_all_events, events_next, mask_next):
        '''
        Evaluate the performance of mark prediction made from the given intensity values.
        ### Args:
          * ```torch.tensor``` intensity_all_events
            shape: [batch_size, seq_len, num_events]
            The intensity values used for mark prediction.
          * ```torch.tensor``` events_next
            shape: [batch_size, seq_len]
            The real mark of next events.
          * ```torch.tensor``` mask_next
            shape: [batch_size, seq_len]
            The mask tensor tells which events in events_next are padding events and which are not.
        ### Outputs:
          * ```float``` f1
            The macro-F1 of mark predictions.
        '''
        events_prediction_probability = torch.log(intensity_all_events + self.epsilon)
                                                                               # [batch_size, seq_len, num_events]
        events_prediction_probability = F.softmax(events_prediction_probability, dim = -1)
                                                                               # [batch_size, seq_len, num_events]

        pred_events = predict_event(events_prediction_probability)[mask_next == 1]
        true_events = events_next[mask_next == 1]
        pred_events, true_events = move_from_tensor_to_ndarray(pred_events, true_events)

        f1 = f1_score(y_pred = pred_events, y_true = true_events, average = 'macro')
        
        return f1


    @torch.inference_mode()
    def mean_absolute_error_and_f1(self, events_history, time_history, events_next, time_next, mask_history, mask_next,
                                   mean, std, opt = None):
        '''
        Called by evaluate_procedure(), debug() and get_mae_and_f1(), this function computed the MAE and macro-F1 of one minibatch.

        ### Args
            * ```torch.tensor``` events_history
              shape: ```[batch_size, seq_len]```
            * ```torch.tensor``` time_history
              shape: ```[batch_size, seq_len]```
              The event history \\mathcal{H}_{t_l}. We use these history info and time history for \\(\\lambda^*(m, t)\\) and \\(\\Lambda^*(m, t)\\).
            * ```torch.tensor``` events_next
              shape: ```[batch_size, seq_len]```
            * ```torch.tensor``` time_next
              shape: ```[batch_size, seq_len]```
            * ```torch.tensor``` mask_history
              shape: ```[batch_size, seq_len]```
            * ```torch.tensor``` mask_next
              shape: ```[batch_size, seq_len]```
              The real-world event sequence. We use events in this sequence to evaluate the predicted events.
            * ```float``` mean
            * ```float``` std
              Used for input time scaling.

        ### Outputs
            * ```torch.tensor``` mae
              shape: ```[batch_size, seq_len]```
              Mean Absolute Error(MAE) between predicted times \\(t_p\\) and ground truths \\(t_i\\). MAE = |t_p - t_i|.
            * ```float``` f1
              macro-F1 value between events predicted at \\(t_p\\) and the ground truths.
            * ```torch.tensor``` mark_distribution
              shape: ```[batch_size, seq_len, num_events]```
              The mark distribution at the real time.
        '''
        pred_time = self.sample_time(sampling_approach = 'its', task = 'tm',
                                     events_history = events_history, time_history = time_history, mask_history = mask_history,
                                     number_of_total_samples = self.sample_rate if opt is None else opt.sample_rate, 
                                     step = self.mae_step if opt is None else opt.mae_step, 
                                     mean = mean, std = std)                   # [sample_rate, batch_size, seq_len]
        pred_time = pred_time.mean(dim = 0)                                    # [batch_size, seq_len]
        mae = torch.abs(time_next - pred_time) * mask_next                     # [batch_size, seq_len]
        
        _, intensity_all_events_pred = self.model(time_history, time_next, events_history, mask_history)
                                                                               # 2 * [batch_size, seq_len, num_events]
        mark_distribution = intensity_all_events_pred / intensity_all_events_pred.sum(dim = -1, keepdim = True)
                                                                               # [batch_size, seq_len, num_events]
        f1_pred = self.evaluate_f1(intensity_all_events_pred, events_next, mask_next)
        
        return mae, f1_pred, mark_distribution


    @torch.inference_mode()
    def mean_absolute_error_e(self, time_history, time_next, events_history, events_next, mask_history, mask_next,
                              mean, std, return_mean = True, opt = None):
        '''
        Called by debug() and get_mae_e_and_f1(), this function computed the MAE-E and macro-F1 of one minibatch.

        ### Args
            * ```torch.tensor``` events_history
              shape: ```[batch_size, seq_len]```
              Historical event sequences. Commonly, this sequence is a slice of the original event sequence from 0 to seq_len - 1(included).
            * ```torch.tensor``` events_next
              shape: ```[batch_size, seq_len]```
              The mark of the events that we need to predict.
            * ```torch.tensor``` time_history
              shape: ```[batch_size, seq_len]```
              Historical time sequences. Similar to events_history, we always generate this sequence as a slice of the original time sequence from 0 to seq_len - 1(included).
            * ```torch.tensor``` time_next
              shape: ```[batch_size, seq_len, num_events]```
              When the next event actually happens. 
            * ```torch.tensor``` mask_history
              shape: ```[batch_size, seq_len]```
            * ```torch.tensor``` mask_next
              shape: ```[batch_size, seq_len]```
              Needed mask to mask out unneeded loss values.
            * ```float``` mean
            * ```float``` std
              Used for input time scaling.
            * ```bool``` return_mean
              If true, we compute the mean of mae_per_event_with_predict_index and mae_per_event_with_event_next on all events in the minibatch.
              If false, we compute the mean of mae_per_event_with_predict_index and mae_per_event_with_event_next per sequence.
            * ```namespace``` opt
              One may bring custom settings into this function through this argument during evaluation. Please refers to
              debug() and get_mae_e_and_f1() for more information about what custom settings are available.

        ### Outputs
            * ```float``` f1
              macro-F1 value between events predicted and the ground truths.
            * ```list``` top_k_acc
              top-1 to top-N accuracy value between events predicted and the ground truths.
            * ```torch.tensor``` probability_integral_sum
              shape: ```[batch_size, seq_len]```
              The sum of p(m) over m.
            * ```torch.tensor``` p_m
              shape: ```[batch_size, seq_len, num_events]```
              The value of p(m) over the different mark m.
            * ```torch.tensor``` tau_pred_all_event
              shape: ```[batch_size, seq_len, num_events]```
              Time predicted by p(t|m) over all marks m.
            * ```torch.tensor``` mae_per_event_with_predict_index_avg
              shape: ```[batch_size]``` if return_mean else ```[1]```
              The average of MAE-E when we pick predicted times using predicted marks.
            * ```torch.tensor``` mae_per_event_with_event_next_avg
              shape: ```[batch_size]``` if return_mean else ```[1]```
              The average of MAE-E when we pick predicted times using real marks.
            * ```torch.tensor``` mae_per_event_with_predict_index
              shape: ```[batch_size, seq_len]```
              The MAE-E values when we pick predicted times using predicted marks.
            * ```torch.tensor``` mae_per_event_with_event_next
              shape: ```[batch_size, seq_len]```
              The MAE-E values when we pick predicted times using real marks.
        '''
        inf_val, resolution_inf, resolution_between_events \
            = decide_resolution_inf_and_resolution_between_events(time_next, memory_ceiling, self.num_events, mean, std)
        time_next_inf = torch.ones_like(time_history, device = self.device) * inf_val
                                                                               # [batch_size, seq_len]
        expanded_integral_all_events_to_inf, expanded_intensity_all_events_to_inf, timestamp = \
            self.model.integral_intensity_time_next_2d(events_history, time_history, time_next_inf, mask_history, resolution_inf)
                                                                               # 2 * [batch_size, seq_len, resolution, num_events]
        expanded_probability_inf = \
            torch.exp(-expanded_integral_all_events_to_inf.sum(dim = -1, keepdim = True)) * expanded_intensity_all_events_to_inf
                                                                               # [batch_size, seq_len, resolution, num_events]
        probability_integral_to_inf = approximate_integration(expanded_probability_inf, timestamp, dim = -2, only_integral = True)
                                                                               # [batch_size, seq_len, num_events]
        probability_integral_sum = probability_integral_to_inf.sum(dim = -1)   # [batch_size, seq_len]
        predicted_events = torch.argmax(probability_integral_to_inf, dim = -1) # [batch_size, seq_len]

        # F1 value and top_k_acc are only avaliable when batch_size = 1
        f1, top_k_acc = get_f1_and_top_k_acc_in_mae_e(events_next, probability_integral_to_inf, mask_next, self.num_events)

        # F1:        [batch_size]
        # top_k_acc: [batch_size, num_events]
        tau_pred_all_event = self.sample_time('its', 'mt', 
                                              events_history = events_history, time_history = time_history, mask_history = mask_history,
                                              p_m = probability_integral_to_inf, resolution = resolution_between_events, number_of_total_samples = self.sample_rate, step = self.mae_e_step, inf_val = inf_val, 
                                              mean = mean, std = std)          # [sample_rate, batch_size, seq_len, num_events]
        predicted_event_mask = F.one_hot(predicted_events.long(), num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]
        event_next_mask = F.one_hot(events_next.long(), num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]

        if return_mean:
            tau_pred_all_event = tau_pred_all_event.mean(dim = 0)              # [batch_size, seq_len, num_events]
            mae_per_event_with_predict_index = torch.abs((tau_pred_all_event * predicted_event_mask).sum(dim = -1) - time_next) * mask_next
                                                                               # [batch_size, seq_len]
            mae_per_event_with_event_next = torch.abs((tau_pred_all_event * event_next_mask).sum(dim = -1) - time_next) * mask_next
                                                                               # [batch_size, seq_len]
    
            mae_per_event_with_predict_index_avg = torch.sum(mae_per_event_with_predict_index, dim = -1) / mask_next.sum(dim = -1)
            mae_per_event_with_event_next_avg = torch.sum(mae_per_event_with_event_next, dim = -1) / mask_next.sum(dim = -1)
        else:
            mae_per_event_with_predict_index = torch.abs((tau_pred_all_event * predicted_event_mask.unsqueeze(dim = 0)).sum(dim = -1) - time_next) * mask_next.unsqueeze(dim = 0)
                                                                               # [sample_rate, batch_size, seq_len]
            mae_per_event_with_event_next = torch.abs((tau_pred_all_event * event_next_mask.unsqueeze(dim = 0)).sum(dim = -1) - time_next) * mask_next.unsqueeze(dim = 0)
                                                                               # [sample_rate, batch_size, seq_len]
    
            mae_per_event_with_predict_index_avg = torch.sum(mae_per_event_with_predict_index, dim = -1) / mask_next.sum(dim = -1)
                                                                               # [sample_rate, batch_size]
            mae_per_event_with_event_next_avg = torch.sum(mae_per_event_with_event_next, dim = -1) / mask_next.sum(dim = -1)
                                                                               # [sample_rate, batch_size]
            
            # Calculate mean
            mae_per_event_with_predict_index = mae_per_event_with_predict_index.mean(dim = 0)
                                                                               # [batch_size, seq_len]
            mae_per_event_with_event_next = mae_per_event_with_event_next.mean(dim = 0)
                                                                               # [batch_size, seq_len]
            mae_per_event_with_predict_index_avg = mae_per_event_with_predict_index_avg.mean(dim = 0)
                                                                               # [batch_size]
            mae_per_event_with_event_next_avg = mae_per_event_with_event_next_avg.mean(dim = 0)
                                                                               # [batch_size]
        
        return f1, top_k_acc, probability_integral_sum, \
               probability_integral_to_inf, \
               tau_pred_all_event, (mae_per_event_with_predict_index_avg, mae_per_event_with_event_next_avg), \
               (mae_per_event_with_predict_index, mae_per_event_with_event_next)


    def extract_plot_data(self, minibatch):
        '''
        This function extracts input_time, input_events, input_intensity, mask, mean, and std from the minibatch.

        ### Args
            * ```list``` minibatch
              shape: ```[[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]```
              data structure: [[input_time, input_events, score, mask], (mean, std)]
              data type: [```torch.tensor```, ```torch.tensor```, ```torch.tensor```, ```torch.tensor```, (```float```, ```float```)]
              The input minibatch.
              
        ### Outputs
            * ```torch.tensor``` input_time
              shape: ```[batch_size, seq_len + 1]```
              Raw event timestamp sequence.
            * ```torch.tensor``` input_events
              shape: ```[batch_size, seq_len + 1]```
              Raw event marks sequence.
            * ```torch.tensor``` mask
              shape: ```[batch_size, seq_len + 1]```
              Raw mask sequence.
            * ```int``` mean
              The mean of all $ t_i - t_{i - 1} $ in the entire dataset. Dataloader is responsible to provide this value if needed.
            * ```int``` std
              The mean of all $ t_i - t_{i - 1} $ in the entire dataset. Dataloader is responsible to provide this value if needed.
        '''
        input_time, input_events, _, mask, input_intensity = minibatch[0]
        mean, std = minibatch[1]

        return input_time, input_events, input_intensity, mask, mean, std
    

    @torch.inference_mode()
    def figure_intensity(self, input_data, opt):
        '''
        Function prober, used by evaluator to draw plots of the intensity function.

        You should declare the following arguments in your config file:
        1. ```int``` resolution: The number of interpolated points in a time interval between two adjoint events for integration estimation.
                                 The number of interpolated points counts the start and end point of the interval.

        ### Args
            * ```list``` input_data
              shape: ```[[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]```
              data structure: [[input_time, input_events, score, mask], (mean, std)]
              data type: [```torch.tensor```, ```torch.tensor```, ```torch.tensor```, ```torch.tensor```, (```float```, ```float```)]
              The input minibatch.
            * ```namespace``` opt
              plot and model configs
        '''
        argument_check(opt, **{'resolution': int})
        
        input_time, input_events, input_intensity, mask, mean, std = self.extract_plot_data(input_data)
        
        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]

        expand_integral, expand_intensity, timestamp = \
            self.model.integral_intensity_time_next_2d(events_history, time_history, time_next, mask_history, opt.resolution)
                                                                               # 3 * [batch_size, seq_len, resolution, num_events]
        check_tensor(expand_integral)
        check_tensor(expand_intensity)
        assert expand_intensity.shape == expand_integral.shape

        data = {
            'time_next': time_next,
            'events_next': events_next,
            'mask_next': mask_next,
            'expand_intensity': expand_intensity,
            'input_intensity': input_intensity, 
            'timestamp': timestamp
            }
        generate_intensity_figure(data, opt)


    @torch.inference_mode()
    def figure_integral(self, input_data, opt):
        '''
        Function prober, used by evaluator to draw plots of the integral of the intensity function.
        
        You should declare the following arguments in your config file:
        1. ```int``` resolution: The number of interpolated points in a time interval between two adjoint events for integration estimation.
                                 The number of interpolated points counts the start and end point of the interval.
        
        ### Args
            * ```list``` input_data
              shape: ```[[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]```
              data structure: [[input_time, input_events, score, mask], (mean, std)]
              data type: [```torch.tensor```, ```torch.tensor```, ```torch.tensor```, ```torch.tensor```, (```float```, ```float```)]
              The input minibatch.
            * ```namespace``` opt
              plot and model configs
        '''
        argument_check(opt, **{'resolution': int})
        
        input_time, input_events, input_intensity, mask, mean, std = self.extract_plot_data(input_data)
        
        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]

        expand_integral, expand_intensity, timestamp = \
            self.model.integral_intensity_time_next_2d(events_history, time_history, time_next, mask_history, opt.resolution)
                                                                               # 3 * [batch_size, seq_len, resolution, num_events]
        check_tensor(expand_integral)
        check_tensor(expand_intensity)
        assert expand_intensity.shape == expand_integral.shape

        data = {
            'time_next': time_next,
            'events_next': events_next,
            'mask_next': mask_next,
            'expand_integral': expand_integral,
            'input_intensity': input_intensity, 
            'timestamp': timestamp
        }
        
        generate_integral_figure(data, opt)


    @torch.inference_mode()
    def figure_probability(self, input_data, opt):
        '''
        Function prober, used by evaluator to draw plots of the probability distribution.
        
        You should declare the following arguments in your config file:
        1. ```int``` resolution: The number of interpolated points in a time interval between two adjoint events for integration estimation.
                                 The number of interpolated points counts the start and end point of the interval.
        
        ### Args
            * ```list``` input_data
              shape: ```[[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]```
              data structure: [[input_time, input_events, score, mask], (mean, std)]
              data type: [```torch.tensor```, ```torch.tensor```, ```torch.tensor```, ```torch.tensor```, (```float```, ```float```)]
              The input minibatch.
            * ```namespace``` opt
              plot and model configs
        '''
        argument_check(opt, **{'resolution': int})
        
        input_time, input_events, input_intensity, mask, mean, std = self.extract_plot_data(input_data)
        
        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]

        expand_integral, expand_intensity, timestamp = \
            self.model.integral_intensity_time_next_2d(events_history, time_history, time_next, mask_history, opt.resolution)
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
            'input_intensity': input_intensity, 
            'timestamp': timestamp
        }
        
        generate_probability_figure(data, opt)


    @torch.inference_mode()
    def figure_debug(self, input_data, opt):
        '''
        Function prober, used by evaluator to draw plots for deeper insight of intensity functions and other metrics.
        
        You should declare the following arguments in your config file:
        1. ```int``` resolution: The number of interpolated points in a time interval between two adjoint events for integration estimation.
                                 The number of interpolated points counts the start and end point of the interval.
        2. ```int``` sample_rate: The number of interpolated points in a time interval between two adjoint events for integration estimation.
                                  The number of interpolated points counts the start and end point of the interval.
        3. ```int``` mae_step: This parameter controls how many samples are generated in one shot when sampling from p(t).
        4. ```int``` mae_e_step: This parameter controls how many samples are generated in one shot when sampling from all p(t|m)s at the same time.
        
        ### Args
            * ```list``` input_data
              shape: ```[[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]```
              data structure: [[input_time, input_events, score, mask], (mean, std)]
              data type: [```torch.tensor```, ```torch.tensor```, ```torch.tensor```, ```torch.tensor```, (```float```, ```float```)]
              The input minibatch.
            * ```namespace``` opt
              plot and model configs
        '''
        argument_check(opt, **{'resolution': int, 'sample_rate': int, 'mae_step': int, 'mae_e_step': int})
        
        input_time, input_events, input_intensity, mask, mean, std = self.extract_plot_data(input_data)

        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]

        mae, f1_1, _ = self.mean_absolute_error_and_f1(events_history, time_history, events_next, time_next, mask_history, mask_next,
                                                       mean, std, opt = opt)   # [batch_size, seq_len]
        data, timestamp = self.model.model_probe_function(events_history, time_history, time_next, mask_history, mask_next, opt.resolution)
        f1_2, top_k, probability_sum, _, tau_pred_all_event, maes_avg, maes \
            = self.mean_absolute_error_e(time_history, time_next, events_history, events_next, mask_history, mask_next,
                                         mean, std, return_mean = False, opt = opt)

        # Append additional info into the data dict.
        data.update({
            'events_next': events_next,
            'time_next': time_next,
            'mask_next': mask_next,
            'f1_after_time_pred': f1_1,
            'mae_before_event': mae,
            'f1_before_time_pred': f1_2,
            'top_k': top_k,
            'probability_sum': probability_sum,
            'tau_pred_all_event': tau_pred_all_event,
            'maes_after_event_avg': maes_avg,
            'maes_after_event': maes,
            'timestamp': timestamp
        })

        generate_debug_figure(data, opt)


    # Evaluation over the entire dataset.
    @torch.inference_mode()
    def get_spearman_and_l1(self, input_data, opt):
        '''
        Used by evaluator to calculate the average gap between the predicted and real distribution using L1 distance and spearman coefficient.
        
        You should declare the following arguments in your config file:
        1. ```int``` resolution: The number of interpolated points in a time interval between two adjoint events for integration estimation.
                                 The number of interpolated points counts the start and end point of the interval.
        
        ### Args
            * ```list``` input_data
              shape: ```[[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]```
              data structure: [[input_time, input_events, score, mask], (mean, std)]
              data type: [```torch.tensor```, ```torch.tensor```, ```torch.tensor```, ```torch.tensor```, (```float```, ```float```)]
              The input minibatch.
            * ```namespace``` opt
              plot and model configs
        
        ### Outputs:
            * ```float``` spearman
              The spearman coefficient between the predicted and real distribution.
            * ```float``` l1
              The l1 distance between the predicted and real distribution.
        '''
        argument_check(opt, **{'resolution', int})
        
        input_time, input_events, input_intensity, mask, mean, std = self.extract_plot_data(input_data)
        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]

        expand_integral, expand_intensity, timestamp = \
            self.model.integral_intensity_time_next_2d(events_history, time_history, time_next, mask_history, opt.resolution)
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

            l1_per_seq = L1_distance_between_two_funcs(x = true_probability_per_seq[:seq_len, :], y = expand_probability_per_seq[:seq_len, :], \
                                                       timestamp = timestamp_per_seq)
            spearman += spearman_per_seq
            l1 += l1_per_seq

        batch_size = mask_next.shape[0]
        spearman /= batch_size
        l1 /= batch_size

        return spearman, l1
    

    @torch.inference_mode()
    def get_mae_and_f1(self, input_data, opt):
        '''
        Used by evaluator to evaluate the performance of predicted time from p(t) and mark from p(m|t).
        
        You should declare the following arguments in your config file:
        1. ```int``` sample_rate: The number of interpolated points in a time interval between two adjoint events for integration estimation.
                                  The number of interpolated points counts the start and end point of the interval.
        2. ```int``` mae_step: This parameter controls how many samples are generated in one shot when sampling from p(t).
        
        ### Args
            * ```list``` input_data
              shape: ```[[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]```
              data structure: [[input_time, input_events, score, mask], (mean, std)]
              data type: [```torch.tensor```, ```torch.tensor```, ```torch.tensor```, ```torch.tensor```, (```float```, ```float```)]
              The input minibatch.
            * ```namespace``` opt
              plot and model configs
        
        ### Outputs:
            * ```np.ndarray``` mae
              shape: ```[batch_size, seq_len]```
              The MAE value, which is the time gap between each predicted and real event.
            * ```float``` f1_1
              The f1 value shows the accuracy of the predicted marks.
            * ```np.ndarray``` dist
              shape: ```[batch_size, seq_len]```
              Predicted mark distribution at when an event is observed.
            * ```np.ndarray``` events_next
              shape: ```[batch_size, seq_len]```
              Real marks of observed events.
        '''
        argument_check(opt, **{'sample_rate': int, 'mae_step': int})
        
        input_time, input_events, input_intensity, mask, mean, std = self.extract_plot_data(input_data)
        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]

        mae, f1_1, dist = self.mean_absolute_error_and_f1(events_history, time_history, events_next, \
                                                          time_next, mask_history, mask_next, mean, std, opt = opt)
                                                                               # [batch_size, seq_len]
        mae, events_next, dist = move_from_tensor_to_ndarray(mae, events_next, dist)

        return mae, f1_1, dist, events_next

    
    @torch.inference_mode()
    def get_mae_e_and_f1(self, input_data, opt):
        '''
        Used by evaluator to evaluate the performance of predicted time from p(m) and mark from p(t|m).
        
        You should declare the following arguments in your config file:
        1. ```int``` sample_rate: The number of interpolated points in a time interval between two adjoint events for integration estimation.
                                  The number of interpolated points counts the start and end point of the interval.
        2. ```int``` mae_e_step: This parameter controls how many samples are generated in one shot when sampling from p(t|m).
        
        ### Args
            * ```list``` input_data
              shape: ```[[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]```
              data structure: [[input_time, input_events, score, mask], (mean, std)]
              data type: [```torch.tensor```, ```torch.tensor```, ```torch.tensor```, ```torch.tensor```, (```float```, ```float```)]
              The input minibatch.
            * ```namespace``` opt
              plot and model configs

        ### Outputs:
            * ```np.ndarray``` maes
              shape: ```[batch_size, seq_len]```
              The MAE-E values when we pick predicted times using real marks.
            * ```float``` f1_2
              The f1 value shows the accuracy of the predicted marks.
            * ```np.ndarray``` probability_sum
              shape: ```[batch_size, seq_len]```
              The sum of calculated p(m) over all marks.
            * ```np.adarray``` p_m
              shape: ```[batch_size, seq_len, num_events]```
              The value of calculated p(m).
            * ```np.ndarray``` tau_pred_all_event
              shape: ```[batch_size, seq_len, num_events]```
              The predicted time for each mark using p(t|m).
            * ```np.ndarray``` time_next
              shape: ```[batch_size, seq_len]```
              Real time of observed events.
            * ```np.ndarray``` events_next
              shape: ```[batch_size, seq_len]```
              Real marks of observed events.
        '''
        argument_check(opt, **{'sample_rate': int, 'mae_e_step': int})
        
        input_time, input_events, input_intensity, mask, mean, std = self.extract_plot_data(input_data)
        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]

        f1_2, top_k, probability_sum, probability_integral_to_inf, tau_pred_all_event, maes_avg, maes \
            = self.mean_absolute_error_e(time_history, time_next, events_history, \
                                         events_next, mask_history, mask_next, mean, std, opt = opt)
        
        _, maes, probability_sum, probability_integral_to_inf, events_next = move_from_tensor_to_ndarray(*maes, probability_sum, probability_integral_to_inf, events_next)

        return maes, f1_2, probability_sum, probability_integral_to_inf, tau_pred_all_event, time_next, events_next


    @torch.inference_mode()
    def get_which_event_first(self, input_data, opt):
        '''
        Used by evaluator to evaluate the performance of predicted time from p(m) and mark from p(t|m).
        Instead of picking the most probable event, we pick the event predicted to happen first.
        
        You should declare the following arguments in your config file:
        1. ```int``` sample_rate: how many time samples from the time distribution are needed.
        2. ```int``` which_event_first_step: This parameter controls how many samples are generated in one shot when sampling from p(t|m).
        
        ### Args
            * ```list``` input_data
              shape: ```[[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]```
              data structure: [[input_time, input_events, score, mask], (mean, std)]
              data type: [```torch.tensor```, ```torch.tensor```, ```torch.tensor```, ```torch.tensor```, (```float```, ```float```)]
              The input minibatch.
            * ```namespace``` opt
              plot and model configs

        ### Outputs:
            * ```np.ndarray``` maes
              shape: ```[batch_size, seq_len]```
              The MAE values when we pick predicted times using real marks.
            * ```float``` f1
              The f1 value shows the accuracy of the predicted marks.
        '''
        argument_check(opt, **{'sample_rate': int, 'which_event_first_step': int})

        input_time, input_events, input_intensity, mask, mean, std = self.extract_plot_data(input_data)
        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]

        inf_val, resolution_inf, resolution_between_events \
            = decide_resolution_inf_and_resolution_between_events(time_next, memory_ceiling, self.num_events, mean, std)
        time_next_inf = torch.ones_like(time_history, device = self.device) * inf_val
                                                                               # [batch_size, seq_len]
        expanded_integral_all_events_to_inf, expanded_intensity_all_events_to_inf, timestamp = \
            self.model.integral_intensity_time_next_2d(events_history, time_history, time_next_inf, mask_history, resolution_inf)
                                                                               # 2 * [batch_size, seq_len, resolution, num_events]
        expanded_probability_inf = \
            torch.exp(-expanded_integral_all_events_to_inf.sum(dim = -1, keepdim = True)) * expanded_intensity_all_events_to_inf
                                                                               # [batch_size, seq_len, resolution, num_events]
        probability_integral_to_inf = approximate_integration(expanded_probability_inf, timestamp, dim = -2, only_integral = True)
                                                                               # [batch_size, seq_len, num_events] 
        # step 2: get the time prediction for that kind of event
        tau_pred_all_event = self.sample_time(sampling_approach = 'its', task = 'mt', 
                                              events_history = events_history, time_history = time_history, mask_history = mask_history,
                                              p_m = probability_integral_to_inf, resolution = resolution_between_events,
                                              number_of_total_samples = opt.sample_rate, step = opt.which_event_first_step,
                                              inf_val = inf_val, 
                                              mean = mean, std = std)          # [sample_rate, batch_size, seq_len, num_events]

        sampled_times_mean = tau_pred_all_event.mean(dim = 0)                  # [batch_size, seq_len, num_events]
        predicted_time, predicted_mark = sampled_times_mean.min(dim = -1)      # [batch_size, seq_len] + [batch_size, seq_len]
        maes = torch.abs(time_next - predicted_time) * mask_next               # [batch_size, seq_len]

        events_pred_index = predicted_mark[mask_next == 1]
        events_true = events_next[mask_next == 1]
        events_true, events_pred_index = move_from_tensor_to_ndarray(events_true, events_pred_index)
        f1 = f1_score(y_true = events_true, y_pred = events_pred_index, average = 'macro')

        maes = move_from_tensor_to_ndarray(maes)

        return maes, f1


    def samples_from_et(self, input_data, opt):
        '''
        This function samples from the distribution p(m, t) by sampling the mark first from p(m) then time from p(t|m).
        All samples can later be used to draw the distribution plot.
        
        You should declare the following arguments in your config file:
        1. ```int``` sample_rate: how many time samples from the time distribution are needed.
        2. ```int``` sample_substep: This parameter controls how many samples are generated in one shot when sampling from p(t|m).
        
        ### Args
            * ```list``` input_data
              shape: ```[[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]```
              data structure: [[input_time, input_events, score, mask], (mean, std)]
              data type: [```torch.tensor```, ```torch.tensor```, ```torch.tensor```, ```torch.tensor```, (```float```, ```float```)]
              The input minibatch.
            * ```namespace``` opt
              plot and model configs

        ### Outputs:
            * ```np.ndarray``` tau_pred_all_event
              shape: ```[batch_size, seq_len, num_events]```
              Predicted time for all marks using p(t|m)
            * ```np.ndarray``` probability_integral_to_inf
              shape: ```[batch_size, seq_len, num_events]```
              The value of p(m).
        '''
        argument_check(opt, **{'sample_rate': int, 'sample_substep': int})
        
        input_time, input_events, input_intensity, mask, mean, std = self.extract_plot_data(input_data)
        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # [batch_size, seq_len]

        input_time, input_events, input_intensity, mask, mean, std = self.extract_plot_data(input_data)
        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]

        inf_val, resolution_inf, resolution_between_events \
            = decide_resolution_inf_and_resolution_between_events(time_next, memory_ceiling, self.num_events, mean, std)
        time_next_inf = torch.ones_like(time_history, device = self.device) * inf_val
                                                                               # [batch_size, seq_len]
        expanded_integral_all_events_to_inf, expanded_intensity_all_events_to_inf, timestamp = \
            self.model.integral_intensity_time_next_2d(events_history, time_history, time_next_inf, mask_history, resolution_inf)
                                                                               # 2 * [batch_size, seq_len, resolution, num_events]
        expanded_probability_inf = \
            torch.exp(-expanded_integral_all_events_to_inf.sum(dim = -1, keepdim = True)) * expanded_intensity_all_events_to_inf
                                                                               # [batch_size, seq_len, resolution, num_events]
        probability_integral_to_inf = approximate_integration(expanded_probability_inf, timestamp, dim = -2, only_integral = True)
                                                                               # [batch_size, seq_len, num_events]   
        # step 2: get the time prediction for that kind of event
        tau_pred_all_event = self.sample_time(sampling_approach = 'its', task = 'mt', 
                                              events_history = events_history, time_history = time_history, mask_history = mask_history,
                                              p_m = probability_integral_to_inf, resolution = resolution_between_events,
                                              number_of_total_samples = opt.sample_rate, step = opt.sample_substep,
                                              inf_val = inf_val, 
                                              mean = mean, std = std)          # [sample_rate, batch_size, seq_len, num_events]

        return tau_pred_all_event, probability_integral_to_inf


    def convert_missing_mask_to_gap_mask(self, missing_mask):
        # input shape: [num_samples, seq_len]
        
        masks = []
        for missing_mask_per_seq in missing_mask:
            current_in_missing = False
            mask_current_seq = []
            for item in missing_mask_per_seq[1:]:
                if item == 1 and not current_in_missing:
                    mask_current_seq.append(1)
                elif item == 1 and current_in_missing:
                    current_in_missing = False
                elif item == 0 and not current_in_missing:
                    mask_current_seq.append(0)
                    current_in_missing = True
                else:
                    continue
            
            masks.append(mask_current_seq)
        
        return masks


    def cppod_evaluation(self, input_data, opt):
        '''
        Take care. This function only evaluates the omission outlier.
        Interestingly, the original CPPOD code seems only focusing on omission too as only omission scores are recorded in model.detect_outlier().
        '''
        forward_complete_data, backward_complete_data, padded_obs_data, padded_backward_obs_event_seq, (mean, std) \
            = input_data
        
        roc_result = []
        for obs_time_for_one_seq, obs_events_for_one_seq, obs_mask_for_one_seq, missing_mask_for_one_seq, _ in padded_obs_data:
            obs_time_history_for_one_seq, obs_time_next_for_one_seq = self.divide_history_and_next(obs_time_for_one_seq)
                                                                               # [batch_size, seq_len] * 2
            obs_events_history_for_one_seq, obs_events_next_for_one_seq = self.divide_history_and_next(obs_events_for_one_seq)
                                                                               # [batch_size, seq_len] * 2
            obs_mask_history_for_one_seq, obs_mask_next_for_one_seq = self.divide_history_and_next(obs_mask_for_one_seq)
                                                                               # [batch_size, seq_len]
            
            missing_mask_for_one_seq = self.convert_missing_mask_to_gap_mask(missing_mask_for_one_seq)
                                                                               # [num_samples, ...]
            integral_all_events, intensity_all_events \
                = self.model(obs_time_history_for_one_seq.float(), obs_time_next_for_one_seq.float(), \
                             obs_events_history_for_one_seq, obs_mask_history_for_one_seq)
                                                                               # [num_samples, seq_len, num_events]
            
            integral_sum = integral_all_events.sum(dim = -1)                   # [num_samples, seq_len]
            intensity_sum = intensity_all_events.sum(dim = -1)                 # [num_samples, seq_len]
            
            all_roauc_area = []
            for integral_sum_per_seq_per_sample, missing_mask_for_one_seq_per_sample in \
                zip(integral_sum, missing_mask_for_one_seq):
                
                sample_len = len(missing_mask_for_one_seq_per_sample)
                selected_integral_sum_per_seq_per_sample = move_from_tensor_to_ndarray(integral_sum_per_seq_per_sample[:sample_len])
                
                roauc_area = roc_auc_score(y_true = np.array(missing_mask_for_one_seq_per_sample) ^ 1, y_score = selected_integral_sum_per_seq_per_sample)
                all_roauc_area.append(roauc_area)
            
            roc_result.append(np.mean(all_roauc_area))
        
        roc_result = np.array(roc_result)
        return roc_result


    def cppod_commission_evaluation(self, input_data, opt):
        (time_seq, events, commission, mask), (mean, std) = input_data
        
        time_history, time_next = self.divide_history_and_next(time_seq)       # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(events)     # [batch_size, seq_len]
        _, commission_next = self.divide_history_and_next(commission)          # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]
        time_history = time_history.float()
        time_next = time_next.float()
        
        _, intensity_all_events = self.model(time_history, time_next, events_history, mask_history)
                                                                               # 2 * [batch_size, seq_len, num_events]
        
        intensity_sum_from_tl_to_time_next = intensity_all_events.sum(dim = -1)
                                                                               # [batch_size, seq_len]
        score = -intensity_sum_from_tl_to_time_next                            # [batch_size, seq_len]
        
        packed_data = zip(score, commission_next, mask_next)
        all_roauc_area = []

        for score_per_seq, commission_next_per_seq, mask_next_per_seq in packed_data:
            available_score = score_per_seq[mask_next_per_seq]
            available_commission_label = commission_next_per_seq[mask_next_per_seq]
            
            available_score, available_commission_label = move_from_tensor_to_ndarray(available_score, available_commission_label)
            roauc_area = roc_auc_score(y_true = available_commission_label, y_score = available_score)
            all_roauc_area.append(roauc_area)
        
        all_roauc_area = np.array(all_roauc_area)
        return all_roauc_area


    def train_step(model, minibatch, device):
        '''
        This function unpacks the minibatch, calls the train_procedure() to calculate the loss, and do the backpropagation.

        ### Args
            * ```torch.nn.Module``` model
              The MTPP model that we train.
            * ```list``` minibatch
              shape: ```[[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]```
              data structure: [[input_time, input_events, score, mask], (mean, std)]
              data type: [```torch.tensor```, ```torch.tensor```, ```torch.tensor```, ```torch.tensor```, (```float```, ```float```)]
              The input minibatch.
            * ```torch.device``` device
              where we train the model.

        ### Outputs:
            * ```float``` tpp_loss
              The average NLL loss without dummy events, specifically the start and the end event.
            * ```float``` fact
              The average NLL loss with the real distribution. This value only makes sense for synthetic datasets.
            * ```float``` mark_loss
              The average cross-entropy loss of the event prediction distribution. The value is only for performance measure porpose.
              The training loss does not and should not include this value.
        '''
        model.train()

        (time, events, fact, mask), (mean, std) = minibatch                    # 3 * [batch_size, seq_len + 1, 1] & [batch_size, seq_len, 1] & 2 numbers
        loss, neg_log_likeli_loss, marker_loss, the_number_of_events = model('train', time, events, mask, mean, std)
        loss.backward()

        tpp_loss, mark_loss = neg_log_likeli_loss.item() / the_number_of_events, marker_loss.item() / the_number_of_events
        fact = fact.sum() / the_number_of_events
    
        return tpp_loss, fact, mark_loss


    def evaluation_step(model, minibatch, device):
        '''
        This function unpacks the minibatch, calls the evaluation_procedure() to compute the metrics.

        ### Args
            * ```torch.nn.Module``` model
              The MTPP model that we train.
            * ```list``` minibatch
              shape: ```[[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]```
              data structure: [[input_time, input_events, score, mask], (mean, std)]
              data type: [```torch.tensor```, ```torch.tensor```, ```torch.tensor```, ```torch.tensor```, (```float```, ```float```)]
              The input minibatch.
            * ```torch.device``` device
              where we train the model.

        ### Outputs:
            * ```float``` time_loss
              The average NLL loss without dummy events, specifically the start and the end event.
            * ```float``` loss_survival
              The average NLL loss of the end event, which is the integral of the intensity function from the last occurred event to the end time.
            * ```float``` fact
              The average NLL loss with the real distribution. This value only makes sense for synthetic datasets.
            * ```float``` events_loss
              The average cross-entropy loss of the event prediction distribution. The value is only for performance measure porpose.
            * ```float``` mae
              The average error between predicted time and real time.
            * ```float``` f1
              The prediction accuracy of predicted marks.
        '''
        model.eval()

        (time, events, fact, mask), (mean, std) = minibatch                    # 3 * [batch_size, seq_len + 1, 1] & [batch_size, seq_len, 1] & 2 numbers
        time_loss, loss_survival, events_loss, mae, f1, the_number_of_events = model('evaluate', time, events, mask, mean, std)

        time_loss = time_loss.item() / the_number_of_events
        loss_survival = loss_survival.item()
        events_loss = events_loss.item() / the_number_of_events
        fact = fact.sum().item() / the_number_of_events
        
        return time_loss, loss_survival, fact, events_loss, mae, f1


    def postprocess(input, procedure):
        '''
        This function makes some modifications to the output of training_step() and evaluation_step().

        ### Args
            * ```list``` input
              The output of either training_step() or evaluation_step().
            * ```str``` procedure
              This string tells the function which function the input comes from.

        ### Outputs:
            * ```list```
              The postprocessed outputs.
        '''
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
    
    
    def log_print_format(input, procedure):
        '''
        This function packs the procedure input into a dict that can be handled by trainer and evaluator for logging.

        ### Args
            * ```list``` input
              The output of either training_step() or evaluation_step().
            * ```str``` procedure
              This string tells the function which function the input comes from.

        ### Outputs:
            * ```dict``` format_dict
              format: {..., <variable name>: {'data': <value>, 'num_format': <num_format>, 'suffix': <suffix>}, ...}
              example: {..., 'memory': {'data': 12.123456, 'num_format': ':2.4f', 'suffix': 'GiB'}, ...}
              The formated results.
        '''
        def train_log_print_format(input):
            format_dict = {}
            format_dict['absolute_loss'] = pack_one_value_to_dict(input[0])
            format_dict['relative_loss'] = pack_one_value_to_dict(input[1])
            format_dict['events_loss'] = pack_one_value_to_dict(input[2])
            return format_dict

        def test_log_print_format(input):
            format_dict = {}
            format_dict['absolute_NLL_loss'] = pack_one_value_to_dict(input[0])
            format_dict['avg_survival_loss'] = pack_one_value_to_dict(input[1])
            format_dict['relative_NLL_loss'] = pack_one_value_to_dict(input[2])
            format_dict['events_loss'] = pack_one_value_to_dict(input[3])
            format_dict['mae'] = pack_one_value_to_dict(input[4], '2.8f')
            format_dict['f1_pred_at_pred_time'] = pack_one_value_to_dict(input[5], '2.8f')
            return format_dict
        
        return (train_log_print_format(input) if procedure == 'Training' else test_log_print_format(input))


    '''
    The maximum length of the format_dict in different procedures.
    '''
    format_dict_length = 6

    
    def choose_metric(evaluation_report_format_dict, test_report_format_dict):
        '''
        This function helps the trainer to pick the best checkpoint based on several metrics.

        ### Args
            * ```dict``` evaluation_report_format_dict
            * ```dict``` test_report_format_dict
              The formated output of training_step() and evaluation_step().

        ### Outputs:
            * ```list```
              The picked metrics used for model select.
            * ```list```
              The name of these metrics.
        '''
        return [evaluation_report_format_dict['absolute_NLL_loss'], 
                test_report_format_dict['absolute_NLL_loss']], \
               ['evaluation_absolute_loss', 'test_absolute_loss']

    '''
    metric number is the length of the output of choose_metric
    '''
    metric_number = 2 # metric number is the length of the output of choose_metric