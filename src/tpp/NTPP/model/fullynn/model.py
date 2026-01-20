import torch, copy
from sklearn.metrics import f1_score, roc_auc_score
from einops import rearrange, repeat, reduce
from scipy.stats import spearmanr

from src.toolbox.misc import check_tensor, move_from_tensor_to_ndarray, conditional_decorator, pack_one_value_to_dict, argument_check
from src.toolbox.integration import approximate_integration

from src.tpp.tpp_models.fenn.sample import sample_time
from src.tpp.tpp_models.basic_tpp_model import BasicModel, memory_ceiling
from src.tpp.tpp_models.fullynn.submodel import FullyNN
from src.tpp.tpp_models.utils import predict_event, decide_resolution_inf_and_resolution_between_events, get_f1_and_top_k_acc_in_mae_e, pick_log_probability
from src.tpp.tpp_models.fullynn.plot import *


class FullyNNModel(BasicModel):
    '''
    The original FullyNN model, proposed by Omi et al. at NeurIPS 2019.
    '''
    def __init__(self, d_history,
                 d_intensity,
                 dropout,
                 history_module_layers,
                 mlp_layers,
                 opt,
                 device,
                 epsilon = 1e-20,
                 history_module = 'LSTM', survival_loss_during_training = True,
                 mae_step = 16, mae_e_step = 16, sample_rate = 32):
        '''
        This function creates a FullyNN model.
        
        ### Args
            * ```str``` history_module
              Which RNN model do we use to encode the history? Default is LSTM. We don't recommend to change it to something else.
            * ```int``` d_history
              The dimension of the history representation.
            * ```float``` dropout
              Dropout rate for the history encoder. Only works when history_module_layers > 1.
            * ```int``` history_module_layers
              How many layer of RNN our model will have?
            * ```int``` d_intensity
              The dimension of the cumulative hazard function network.
            * ```int``` mlp_layers
              The number of layers in the cumulative hazard function network.
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
            * ```bool``` survival_loss_during_training
              When true, the training loss includes the integral between the last observed event to the end time T. Most of time this argument should be true.
        '''
        super(FullyNNModel, self).__init__()
        self.device = device
        self.compile_or_not = opt.compile
        self.num_events = opt.info_dict['num_events']
        self.start_time = opt.info_dict['t_0']
        self.end_time = opt.info_dict['T']
        self.epsilon = epsilon
        self.survival_loss_during_training = survival_loss_during_training
        self.sample_rate = sample_rate
        self.mae_step = mae_step
        self.mae_e_step = mae_e_step
        self.bisect_early_stop_threshold = 1e-4
        self.max_step = 50

        self.model = FullyNN(d_history = d_history, d_intensity = d_intensity, num_events = self.num_events,
                             dropout = dropout, history_module = history_module, history_module_layers = history_module_layers,
                             mlp_layers = mlp_layers, device = device)


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
        Remove the probability of the dummy event from the mask.

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
        The entrance of the FullyNN.
        
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


    def train_procedure(self, input_time, input_events, mask, mean, std):
        '''
        FullyNN's forwardpropagation function for training.
        
        ### Args
            * ```torch.tensor``` input_time
              shape: ```[batch_size, seq_len + 1]```
              Time sequence for training.
            * ```torch.tensor``` input_events
              shape: ```[batch_size, seq_len + 1]```
              Event sequence for training.
            * ```torch.tensor``` mask
              shape: ```[batch_size,, seq_len + 1]```
              Mask sequence. Events whose corresponding mask is 0 are dummy events.
            * ```float``` mean
            * ```float``` std
              Used for input time scaling.

        ### Outputs
            * ```torch.tensor``` time_loss
              shape: ```[1]```
              The sum of NLL loss L = -log \\frac{\\partial \\Lambda^*(m, t)}{\\partial t} + \\Lambda^*(m, t) at each happened event (the dummy event at end time T included).
            * ```torch.tensor``` time_loss_without_dummy
              shape: ```[1]```
              The sum of NLL loss L = -log \\frac{\\partial \\Lambda^*(m, t)}{\\partial t} + \\Lambda^*(m, t) at each happened event (the dummy event at end time T excluded).
            * ```torch.tensor``` events_loss
              shape: ```[1]```
              The sum of the event loss: L = -log \\frac{\\lambda^*(m, t)}{\\sum_{n \\in M}{\\lambda^*(n, t)}} where m is the mark of the real event.
            * ```int``` the_number_of_events
              The number of legit events.
        '''
        time_history, time_next = self.divide_history_and_next(input_time)     # 2 * [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # 2 * [batch_size, seq_len]
        _, mask_next = self.divide_history_and_next(mask)                      # [batch_size, seq_len]

        # preparing for multi-event training when needed
        time_next = repeat(time_next, 'b s -> b s ne', ne = self.num_events)   # [batch_size, seq_len, num_events]
        time_next.requires_grad = True
        integral_for_each_event = self.model(events_history, time_history, time_next, mean = mean, std = std)
                                                                               # [batch_size, seq_len, num_events]
        # Obtains intensity values.
        intensity_for_each_event = torch.autograd.grad(
            outputs = integral_for_each_event,
            inputs = time_next,
            grad_outputs = torch.ones_like(integral_for_each_event),
            create_graph = True,
        )[0]
        check_tensor(intensity_for_each_event)                                 # [batch_size, seq_len, num_events]
        assert intensity_for_each_event.shape == integral_for_each_event.shape
        time_next.requires_grad = False

        mask_next_without_dummy = self.remove_dummy_event_from_mask(mask_next) # [batch_size, seq_len]
        events_next_without_dummy = (events_next * mask_next_without_dummy).long()
                                                                               # [batch_size, seq_len]
        the_number_of_events = mask_next_without_dummy.sum().item()

        # Calculate the event loss, macro-F1, and other possible metrics measuring event prediction accuracy.
        events_loss = torch.tensor(0., dtype = torch.float32)
        probability_for_each_event = torch.log(intensity_for_each_event + self.epsilon)
                                                                               # [batch_size, seq_len, num_events]
        events_probability = torch.nn.functional.softmax(probability_for_each_event, dim = -1)
                                                                               # [batch_size, seq_len, num_events]
        events_loss = torch.nn.functional.cross_entropy(rearrange(events_probability, 'b s ne -> b ne s'), \
                                                                  events_next_without_dummy, reduction = 'none')
                                                                               # [batch_size, seq_len]
        events_loss = events_loss * mask_next_without_dummy                    # [batch_size, seq_len]
        events_loss = events_loss.sum()

        # Calculate the NLL loss of p^*(m, t).
        # L = -log \\frac{\\partial \\Lambda^*(m, t)}{\\partial t} + \\Lambda^*(m, t)
        time_loss_without_dummy = self.nll_loss(intensity = intensity_for_each_event, events_next = events_next_without_dummy, \
                                                intensity_integral = integral_for_each_event, mask_next = mask_next_without_dummy)
        # Survival probability: \\int_{t_N}^{T}{\\sum_{k}\\lambda_k^(\\tau)d\\tau}
        loss_survival = 0
        if self.survival_loss_during_training:
            dummy_event_index = mask_next.sum(dim = -1) - 1                    # [batch_size]
            integral_survival = integral_for_each_event.sum(dim = -1).gather(index = dummy_event_index.unsqueeze(dim = -1), dim = -1)
                                                                               # [batch_size, 1]
            loss_survival = integral_survival.sum()

        loss = time_loss_without_dummy + loss_survival

        return loss, time_loss_without_dummy, events_loss, the_number_of_events


    def evaluate_procedure(self, input_time, input_events, mask, mean, std):
        '''
        FullyNN's forwardpropagation function for evaluation.
        
        ### Args
            * ```torch.tensor``` input_time
              shape: ```[batch_size, seq_len + 1]```
              Time sequencalculatesce for training.
            * ```torch.tensor``` input_events
              shape: ```[batch_size, seq_len + 1]```
              Event sequence for training.
            * ```torch.tensor``` mask
              shape: ```[batch_size,, seq_len + 1]```
              Mask sequence. Events whose corresponding mask is 0 are dummy events.
            * ```float``` mean
            * ```float``` std
              Used for input time scaling.

        ### Outputs
            * ```torch.tensor``` time_loss
              shape: ```[1]```
              The sum of NLL loss L = -log \\frac{\\partial \\Lambda^*(m, t)}{\\partial t} + \\Lambda^*(m, t) at each happened event.
            * ```torch.tensor``` loss_survival
              shape: ```[1]```
              The sum of the integration \\Lambda^*(m, t) from the last observed event to the end time T.
            * ```torch.tensor``` events_loss
              shape: ```[1]```
              The sum of the event loss: L = -log \\frac{\\lambda^*(m, t)}{\\sum_{n \\in M}{\\lambda^*(n, t)}} where m is the mark of the real event.
            * ```float``` mae
              The average error between predicted time and real time.
            * ```float``` f1
              The prediction accuracy of predicted marks.
            * ```int``` the_number_of_events
              The number of legit events.
        '''
        time_history, time_next = self.divide_history_and_next(input_time)     # 2 * [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # 2 * [batch_size, seq_len]
        _, mask_next = self.divide_history_and_next(mask)                      # [batch_size, seq_len]

        mask_next_without_dummy = self.remove_dummy_event_from_mask(mask_next) # [batch_size, seq_len]
        events_next_without_dummy = (events_next * mask_next_without_dummy).long()
                                                                               # [batch_size, seq_len]
        the_number_of_events = mask_next_without_dummy.sum().item()

        mae, f1, _ = self.mean_absolute_error_and_f1(events_history = events_history, time_history = time_history,\
                                                         events_next = events_next, time_next = time_next, 
                                                         mask_next = mask_next_without_dummy, mean = mean, std = std)
                                                                               # 2 * [batch_size, seq_len]
        mae = mae.sum().item() / the_number_of_events

        time_next = repeat(time_next, 'b s -> b s ne', ne = self.num_events)   # [batch_size, seq_len, num_events]

        # preparing for multi-event training when needed
        # Caution: We calculate the absolute and relative time loss at event_next, not pred_time.
        time_next.requires_grad = True
        integral_for_each_event_from_tl_to_time_next = self.model(events_history, time_history, time_next, mean = mean, std = std)
                                                                               # [batch_size, seq_len, num_events]
        # Obtains intensity values.
        intensity_for_each_event_from_tl_to_time_next = torch.autograd.grad(
            outputs = integral_for_each_event_from_tl_to_time_next,
            inputs = time_next,
            grad_outputs = torch.ones_like(integral_for_each_event_from_tl_to_time_next),
        )[0]                                                                   # [batch_size, seq_len, num_events]
        time_next.requires_grad = False
        check_tensor(intensity_for_each_event_from_tl_to_time_next)            # [batch_size, seq_len, num_events]
        assert intensity_for_each_event_from_tl_to_time_next.shape == integral_for_each_event_from_tl_to_time_next.shape

        # Calculate the event loss, macro-F1, and other possible metrics measuring event prediction accuracy.
        probability_for_each_event = torch.log(intensity_for_each_event_from_tl_to_time_next + self.epsilon)
                                                                               # [batch_size, seq_len, num_events]
        events_probability = torch.nn.functional.softmax(probability_for_each_event, dim = -1)
                                                                               # [batch_size, seq_len, num_events]
        events_loss = torch.nn.functional.cross_entropy(rearrange(events_probability, 'b s ne -> b ne s'), \
                                                                  events_next_without_dummy, reduction = 'none')
                                                                               # [batch_size, seq_len]
        events_loss = events_loss * mask_next_without_dummy                    # [batch_size, seq_len]
        events_loss = events_loss.sum()

        # Calculate the NLL loss of p^*(m, t).
        # L = -log \\frac{\\partial \\Lambda^*(m, t)}{\\partial t} + \\Lambda^*(m, t)
        time_loss = self.nll_loss(intensity = intensity_for_each_event_from_tl_to_time_next, events_next = events_next_without_dummy, \
                                  intensity_integral = integral_for_each_event_from_tl_to_time_next, mask_next = mask_next_without_dummy)
        # Survival probability: \\int_{t_N}^{T}{\\sum_{k}\\lambda_k^(\\tau)d\\tau}
        dummy_event_index = mask_next.sum(dim = -1) - 1                        # [batch_size]
        integral_survival = integral_for_each_event_from_tl_to_time_next.sum(dim = -1).gather(index = dummy_event_index.unsqueeze(dim = -1), dim = -1)
                                                                               # [batch_size, 1]
        loss_survival = integral_survival.mean()

        return time_loss, loss_survival, events_loss, mae, f1, the_number_of_events
    

    def nll_loss(self, intensity, intensity_integral, events_next, mask_next):
        '''
        This function computes the NLL loss at each legit event in events_next.
    
        ### Args
            * ```torch.tensor``` intensity
              shape: ```[batch_size, seq_len, num_events]```
              intensity values at t_i.
            * ```torch.tensor``` intensity_integral
              shape: ```[batch_size, seq_len, num_events]```
              intensity integral from t_{i - 1} to t_{i} (t_0 = 0).
            * ```torch.tensor``` events_next
              shape: ```[batch_size, seq_len]```
              The mark of the events that we need to predict.
            * ```torch.tensor``` mask_next
              shape: ```[batch_size, seq_len]```
              Needed mask to mask out unneeded loss values.
        
        ### Outputs
            * ```torch.tensor``` loss
              shape: ```[1]```
              The sum of NLL loss on all event.
        '''
        intensity_mask = torch.nn.functional.one_hot(events_next, num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]
        log_intensity = torch.log(intensity + self.epsilon) * intensity_mask
        log_intensity = reduce(log_intensity, '... ne -> ...', 'sum')          # [batch_size, seq_len]
        intensity_integral = reduce(intensity_integral, '... ne -> ...', 'sum')# [batch_size, seq_len]
        nll_p = -log_intensity + intensity_integral                            # [batch_size, seq_len]
        loss = nll_p * mask_next
        loss = torch.sum(loss)

        return loss


    sample_time = sample_time


    def mean_absolute_error_and_f1(self, events_history, time_history, events_next, time_next, mask_next, mean, std, opt = None):
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
            * ```torch.tensor``` events_probability
              shape: ```[batch_size, seq_len, num_events]```
              The mark distribution at the real time.
        '''
        pred_time = self.sample_time(sampling_approach = 'its', task = 'tm',
                                     events_history = events_history, time_history = time_history,
                                     number_of_total_samples = self.sample_rate if opt is None else opt.sample_rate,
                                     step = self.mae_step if opt is None else opt.mae_step,
                                     mean = mean, std = std)                   # [sample_rate, batch_size, seq_len]
        pred_time = pred_time.mean(dim = 0)                                    # [batch_size, seq_len]
        mae = torch.abs(pred_time - time_next) * mask_next                     # [batch_size, seq_len]

        time_next = repeat(time_next, 'b s -> b s ne', ne = self.num_events)   # [batch_size, seq_len, num_events]

        # preparing for multi-event training when needed
        time_next.requires_grad = True
        integral_for_each_event = self.model(events_history, time_history, time_next, mean = mean, std = std)
                                                                               # [batch_size, seq_len, num_events]
        # Obtains intensity values.
        intensity_for_each_event = torch.autograd.grad(
            outputs = integral_for_each_event,
            inputs = time_next,
            grad_outputs = torch.ones_like(integral_for_each_event),
        )[0]
        check_tensor(intensity_for_each_event)                                 # [batch_size, seq_len, num_events]
        time_next.requires_grad = False
        assert intensity_for_each_event.shape == integral_for_each_event.shape

        # Calculate the event loss, macro-F1, and other possible metrics measuring event prediction accuracy.
        f1 = 0
        probability_for_each_event = torch.log(intensity_for_each_event + self.epsilon)
                                                                               # [batch_size, seq_len, num_events]
        events_probability = torch.nn.functional.softmax(probability_for_each_event, dim = -1)
                                                                               # [batch_size, seq_len, num_events]

        events_pred_index = predict_event(events_probability)[mask_next == 1]
        events_true = events_next[mask_next == 1]
        events_pred_index, events_true = move_from_tensor_to_ndarray(events_pred_index, events_true)
        f1 = f1_score(y_true = events_true, y_pred = events_pred_index, average = 'macro')

        return mae, f1, events_probability


    def mean_absolute_error_e(self, events_history, events_next, time_history, time_next, mask_next, mean, std,
                              return_mean = True, opt = None):
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
        
        # set a relatively large number as the infinity and decide resolution based on this large value and
        # the memory_ceiling.
        inf_val, resolution_inf, resolution_between_events = \
            decide_resolution_inf_and_resolution_between_events(time_next, memory_ceiling, self.num_events, mean, std)
        time_next_inf = torch.ones_like(time_history) * inf_val                # [batch_size, seq_len]

        # Step 1: obtain p^*(m) = \\int_{t_l}^{+infty}{p(m, t)\\dt}
        expand_integral_to_inf, expand_intensity_to_inf, time_interval \
                = self.model.integral_intensity_time_next_2d(events_history, time_history, time_next_inf, resolution_inf, mean, std)
                                                                               # [batch_size, seq_len, resolution, num_events]
        # Step 2: provide event predictions
        expand_probability_per_event = expand_intensity_to_inf * torch.exp(-expand_integral_to_inf.sum(dim = -1, keepdim = True))
                                                                               # [batch_size, seq_len, resolution, num_events]
        p_m = approximate_integration(expand_probability_per_event, time_interval, dim = -2, only_integral = True)
                                                                               # [batch_size, seq_len, num_events]
        probability_integral_sum = reduce(p_m, 'b s ne -> b s', 'sum')         # [batch_size, seq_len]
        predict_index = torch.argmax(p_m, dim = -1)                            # [batch_size, seq_len]

        # Step 3: calculate macro-F1 and top-K accuracy
        f1, top_k_acc = get_f1_and_top_k_acc_in_mae_e(events_next, p_m, mask_next, self.num_events)

        predict_index_one_hot_mask = torch.nn.functional.one_hot(predict_index.long(), num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]
        events_next_one_hot_mask = torch.nn.functional.one_hot(events_next.long(), num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]
        
        # Step 4: get the time prediction for all, predicted, and real events.
        tau_pred_all_event = self.sample_time(sampling_approach = 'its', task = 'mt', 
                                              events_history = events_history, time_history = time_history, 
                                              p_m = p_m, resolution = resolution_between_events,
                                              number_of_total_samples = self.sample_rate if opt is None else opt.sample_rate,
                                              step = self.mae_e_step if opt is None else opt.mae_e_step, 
                                              inf_val = inf_val, mean = mean, std = std)
                                                                               # [sample_rate, batch_size, seq_len, num_events]
        if return_mean:
            tau_pred_all_event = tau_pred_all_event.mean(dim = 0)              # [batch_size, seq_len, num_events]
            mae_per_event_with_predict_index = torch.abs((tau_pred_all_event * predict_index_one_hot_mask).sum(dim = -1) - time_next) * mask_next
                                                                               # [batch_size, seq_len]
            mae_per_event_with_event_next = torch.abs((tau_pred_all_event * events_next_one_hot_mask).sum(dim = -1) - time_next) * mask_next
                                                                               # [batch_size, seq_len]
    
            mae_per_event_with_predict_index_avg = torch.sum(mae_per_event_with_predict_index, dim = -1) / mask_next.sum(dim = -1)
            mae_per_event_with_event_next_avg = torch.sum(mae_per_event_with_event_next, dim = -1) / mask_next.sum(dim = -1)
        else:
            mae_per_event_with_predict_index = torch.abs((tau_pred_all_event * predict_index_one_hot_mask.unsqueeze(dim = 0)).sum(dim = -1) - time_next) * mask_next.unsqueeze(dim = 0)
                                                                               # [sample_rate, batch_size, seq_len]
            mae_per_event_with_event_next = torch.abs((tau_pred_all_event * events_next_one_hot_mask.unsqueeze(dim = 0)).sum(dim = -1) - time_next) * mask_next.unsqueeze(dim = 0)
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

        return f1, top_k_acc, probability_integral_sum, p_m, \
               tau_pred_all_event, \
               (mae_per_event_with_predict_index_avg, mae_per_event_with_event_next_avg), \
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
            self.model.integral_intensity_time_next_2d(events_history, time_history, time_next, opt.resolution, mean, std)
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
            'timestamp': timestamp}

        generate_intensity_figure(data, opt)


    def figure_integral(self, input_data, opt):
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
            self.model.integral_intensity_time_next_2d(events_history, time_history, time_next, opt.resolution, mean, std)
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
            'timestamp': timestamp}
        
        generate_integral_figure(data, opt)


    def figure_probability(self, input_data, opt):
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
            self.model.integral_intensity_time_next_2d(events_history, time_history, time_next, opt.resolution, mean, std)
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
            'timestamp': timestamp}
        
        generate_probability_figure(data, opt)


    def figure_debug(self, input_data, opt):
        '''
        Function prober, used by evaluator to draw plots for deeper insight of intensity functions and other metrics.
        
        You should declare the following arguments in your config file:
        1. ```int``` resolution: The number of interpolated points in a time interval between two adjoint events for integration estimation.
                                 The number of interpolated points counts the start and end point of the interval.
        2. ```int``` sample_rate: how many time samples from the time distribution are needed.
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

        mae, f1_1, _ = self.mean_absolute_error_and_f1(events_history, time_history, events_next, \
                                                       time_next, mask_next, mean, std, opt = opt)
                                                                               # [batch_size, seq_len]
        data, timestamp = self.model.model_probe_function(events_history, time_history, time_next, mask_next, \
                                                          opt.resolution, mean, std)
        f1_2, top_k, probability_sum, _, tau_pred_all_event, maes_avg, maes \
            = self.mean_absolute_error_e(events_history, events_next, time_history, time_next, mask_next, mean, std,
                                         return_mean = False, opt = opt)

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
        _, mask_next = self.divide_history_and_next(mask)                      # [batch_size, seq_len]

        expand_integral, expand_intensity, timestamp = \
            self.model.integral_intensity_time_next_2d(events_history, time_history, time_next, opt.resolution, mean, std)
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

            l1_per_seq = \
                L1_distance_between_two_funcs(x = true_probability_per_seq[:seq_len, :], y = expand_probability_per_seq[:seq_len, :], \
                                              timestamp = timestamp_per_seq)
            spearman += spearman_per_seq
            l1 += l1_per_seq

        batch_size = mask_next.shape[0]
        spearman /= batch_size
        l1 /= batch_size

        return spearman, l1


    def get_mae_and_f1(self, input_data, opt):
        '''
        Used by evaluator to evaluate the performance of predicted time from p(t) and mark from p(m|t).
        
        You should declare the following arguments in your config file:
        1. ```int``` sample_rate: how many time samples from the time distribution are needed.
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
            * ```torch.tensor``` dist
              shape: ```[batch_size, seq_len, num_events]```
              The mark distribution at when the real event happens.
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
                                                    time_next, mask_next, mean, std, opt = opt)
                                                                               # [batch_size, seq_len]
        mae, events_next, dist = move_from_tensor_to_ndarray(mae, events_next, dist)

        return mae, f1_1, dist, events_next
    

    def get_mae_e_and_f1(self, input_data, opt):
        '''
        Used by evaluator to evaluate the performance of predicted time from p(m) and mark from p(t|m).
        
        You should declare the following arguments in your config file:
        1. ```int``` sample_rate: how many time samples from the time distribution are needed.
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

        f1_2, top_k, probability_sum, p_m, tau_pred_all_event, maes_avg, maes \
            = self.mean_absolute_error_e(events_history, events_next, time_history, time_next, mask_next, mean, std, opt = opt)
        
        _, maes, probability_sum, p_m, tau_pred_all_event, time_next, events_next \
          = move_from_tensor_to_ndarray(*maes, probability_sum, p_m, tau_pred_all_event, time_next, events_next)

        return maes, f1_2, probability_sum, p_m, tau_pred_all_event, time_next, events_next


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
    
        inf_val, resolution_inf, resolution_between_events = \
            decide_resolution_inf_and_resolution_between_events(time_next, memory_ceiling, self.num_events, mean, std)
        time_next_inf = torch.ones_like(time_history) * inf_val                # [batch_size, seq_len]

        expand_integral_to_inf, expand_intensity_to_inf, time_interval \
                = self.model.integral_intensity_time_next_2d(events_history, time_history, time_next_inf, resolution_inf, mean, std)
                                                                               # [batch_size, seq_len, resolution, num_events]
        expand_probability_per_event = expand_intensity_to_inf * torch.exp(-expand_integral_to_inf.sum(dim = -1, keepdim = True))
                                                                               # [batch_size, seq_len, resolution, num_events]
        p_m = approximate_integration(expand_probability_per_event, time_interval, dim = -2, only_integral = True)
                                                                               # [batch_size, seq_len, num_events]
        # step 2: get the time prediction for that kind of event
        tau_pred_all_event = self.sample_time(sampling_approach = 'its', task = 'mt', \
                                              events_history = events_history, time_history = time_history, p_m = p_m, \
                                              number_of_total_samples = opt.sample_rate, step = opt.which_event_first_step, \
                                              resolution = resolution_between_events, inf_val = inf_val, mean = mean, std = std)
                                                                               # [sample_rate, batch_size, seq_len, num_events]

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
            * ```np.ndarray``` p_m
              shape: ```[batch_size, seq_len, num_events]```
              The value of p(m).
        '''
        argument_check(opt, **{'sample_rate': int, 'sample_substep': int})
        
        input_time, input_events, input_intensity, mask, mean, std = self.extract_plot_data(input_data)
        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # [batch_size, seq_len]

        inf_val, resolution_inf, resolution_between_events = \
            decide_resolution_inf_and_resolution_between_events(time_next, memory_ceiling, self.num_events, mean, std)
        time_next_inf = torch.ones_like(time_history) * inf_val                # [batch_size, seq_len]

        expand_integral_to_inf, expand_intensity_to_inf, time_interval \
                = self.model.integral_intensity_time_next_2d(events_history, time_history, time_next_inf, resolution_inf, mean, std)
                                                                               # [batch_size, seq_len, resolution, num_events]
        expand_probability_per_event = expand_intensity_to_inf * torch.exp(-expand_integral_to_inf.sum(dim = -1, keepdim = True))
                                                                               # [batch_size, seq_len, resolution, num_events]
        p_m = approximate_integration(expand_probability_per_event, time_interval, dim = -2, only_integral = True)
                                                                               # [batch_size, seq_len, num_events]  
        # step 2: get the time prediction for that kind of event
        tau_pred_all_event = self.sample_time(sampling_approach = 'its', task = 'mt', \
                                              events_history = events_history, time_history = time_history, p_m = p_m, \
                                              number_of_total_samples = opt.sample_rate, step = opt.sample_substep, 
                                              resolution = resolution_between_events, inf_val = inf_val, mean = mean, std = std)
                                                                               # [sample_rate, batch_size, seq_len, num_events]

        tau_pred_all_event, probability_integral_from_zero_to_infinite = \
            move_from_tensor_to_ndarray(tau_pred_all_event, probability_integral_from_zero_to_infinite)

        return tau_pred_all_event, p_m


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
        Paired with the od_genetic dataloader.
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
            
            obs_time_next_for_one_seq = repeat(obs_time_next_for_one_seq, 'b s -> b s ne', ne = self.num_events)
                                                                               # [batch_size, seq_len, num_events]
            missing_mask_for_one_seq = self.convert_missing_mask_to_gap_mask(missing_mask_for_one_seq)
                                                                               # [num_samples, ...]
            integral_all_events = self.model(obs_events_history_for_one_seq, obs_time_history_for_one_seq.float(), obs_time_next_for_one_seq.float(), mean, std)
                                                                               # [num_samples, seq_len, num_events]
            
            integral_sum = integral_all_events.sum(dim = -1)                   # [num_samples, seq_len]
            
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
        
        time_next = repeat(time_next, 'b s -> b s ne', ne = self.num_events)   # [batch_size, seq_len, num_events]
        '''
        preparing for multi-event training when needed
        '''
        time_next.requires_grad = True
        integral_for_each_event_from_tl_to_time_next = self.model(events_history, time_history, time_next, mean = mean, std = std)
                                                                               # [batch_size, seq_len, num_events]
        '''
        Obtains intensity values.
        '''
        intensity_for_each_event_from_tl_to_time_next = torch.autograd.grad(
            outputs = integral_for_each_event_from_tl_to_time_next,
            inputs = time_next,
            grad_outputs = torch.ones_like(integral_for_each_event_from_tl_to_time_next),
        )[0]                                                                   # [batch_size, seq_len, num_events]
        time_next.requires_grad = False
        check_tensor(intensity_for_each_event_from_tl_to_time_next)            # [batch_size, seq_len, num_events]
        assert intensity_for_each_event_from_tl_to_time_next.shape == integral_for_each_event_from_tl_to_time_next.shape
        
        intensity_sum_from_tl_to_time_next = intensity_for_each_event_from_tl_to_time_next.sum(dim = -1)
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
            * ```float``` time_loss_without_dummy
              The average NLL loss without dummy events, specifically the start and the end event.
            * ```float``` fact
              The average NLL loss with the real distribution. This value only makes sense for synthetic datasets.
            * ```float``` events_loss
              The average cross-entropy loss of the event prediction distribution. The value is only for performance measure porpose.
              The training loss does not and should not include this value.
        '''
        model.train()
    
        [time_seq, event_seq, score, mask], (mean, std) = minibatch
        loss, time_loss_without_dummy, events_loss, the_number_of_events = model(         
                task_name = 'train', input_time = time_seq, input_events = event_seq, \
                mask = mask, mean = mean, std = std
        )

        loss.backward()
    
        time_loss_without_dummy = time_loss_without_dummy.item() / the_number_of_events
        events_loss = events_loss.item() / the_number_of_events
        fact = score.sum().item() / the_number_of_events
        
        return time_loss_without_dummy, fact, events_loss


    def evaluation_step(model, minibatch, device):
        '''
        This function unpacks the minibatch, calls the evaluation_procedure() to calculate the metrics.

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
    
        [time_seq, event_seq, score, mask], (mean, std) = minibatch
        time_loss, loss_survival, events_loss, mae, f1, the_number_of_events = model(
                task_name = 'evaluate', input_time = time_seq, input_events = event_seq, \
                mask = mask, mean = mean, std = std
        )
    
        time_loss = time_loss.item() / the_number_of_events
        loss_survival = loss_survival.item()
        events_loss = events_loss.item() / the_number_of_events
        fact = score.sum().item() / the_number_of_events
        
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
    metric_number = 2

    '''
    True: The lower the metric is, the better the model is.
    False: The higher the metric is, the better the model is.
    '''
    smaller_is_better = [True, True]