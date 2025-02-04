import torch, copy
from sklearn.metrics import f1_score, roc_auc_score
from einops import rearrange, repeat, reduce, pack
from scipy.stats import spearmanr
import torch.nn.functional as F

from src.toolbox.misc import check_tensor, move_from_tensor_to_ndarray, pack_one_value_to_dict, conditional_decorator
from src.toolbox.metrics import L1_distance_between_two_funcs

from src.TPP.model.basic_tpp_model import BasicModel
from src.TPP.model.ifib_c.submodel import IFIBC
from src.TPP.model.ifib_c.sample import sample_time, sample_time_event, sample_event_time
from src.TPP.model.ifib_c.plot import *
from src.TPP.model.utils import predict_event, get_f1_and_top_k_acc_in_mae_e, step_split


class IFIBCModel(BasicModel):
    def __init__(self, d_history,
                 d_intensity,
                 dropout,
                 history_module_layers,
                 mlp_layers,
                 opt,
                 device,
                 removes_tail, tanh_parameter,
                 history_module = 'LSTM', survival_loss_during_training = True,
                 epsilon = 0.0, sample_rate = 32, mae_step = 32, mae_e_step = 32):
        super(IFIBCModel, self).__init__()
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

        self.model = IFIBC(d_history = d_history, d_intensity = d_intensity, num_events = self.num_events,
                           dropout = dropout, history_module = history_module, history_module_layers = history_module_layers,
                           mlp_layers = mlp_layers, removes_tail = removes_tail, tanh_parameter = tanh_parameter, 
                           epsilon = epsilon, device = device)


    def divide_history_and_next(self, input):
        input_history, input_next = input[:, :-1].clone(), input[:, 1:].clone()
        return input_history, input_next


    def forward(self, task_name, *args, **kwargs):
        '''
        The entrance of the IFIB-C wrapper.
        
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
        * std           type: float shape: N/A
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
            'mae_e_and_f1_by_time_event': self.get_mae_e_and_f1_by_time_event,
            'mae_e_and_f1': self.get_mae_e_and_f1,
            'which_event_occurs_first': self.get_which_event_first,
            'samples_from_et': self.samples_from_et,
            'generate_hypro_dataset': self.generate_hypro_dataset,

            # Figure Drawing.
            'intensity': self.figure_intensity,
            'integral': self.figure_integral,
            'probability': self.figure_probability,
            'debug': self.figure_debug,

            # For CPPOD, should be used with the od_generic dataloader.
            'cppod_evaluation': self.cppod_evaluation
        }

        return task_mapper[task_name](*args, **kwargs)
    

    def remove_dummy_event_from_mask(self, mask):
        '''
        Flip the mask of the end dummy event from 1 to 0.
        '''
        mask_without_dummy = torch.zeros_like(mask)                            # [batch_size, seq_len - 1]
        for idx, mask_per_seq in enumerate(mask):
            dummy_index = mask_per_seq.sum() - 1
            mask_without_dummy_per_seq = copy.deepcopy(mask_per_seq.detach())
            mask_without_dummy_per_seq[dummy_index] = 0
            mask_without_dummy[idx] = mask_without_dummy_per_seq
        
        return mask_without_dummy


    def train_procedure(self, input_time, input_events, mask, mean, std):
        time_history, time_next = self.divide_history_and_next(input_time)     # 2 * [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # 2 * [batch_size, seq_len]
        _, mask_next = self.divide_history_and_next(mask)                      # [batch_size, seq_len]

        time_next = repeat(time_next, 'b s -> b s ne', ne = self.num_events)   # [batch_size, seq_len, num_events]
        time_next.requires_grad = True

        '''
        \\int_{t}^{+\\inf}{p(m, \\tau|\\mathcal{H})d\\tau}
        '''
        probability_integral_from_t_to_infinite = self.model('default_forward', events_history, time_history, time_next, mean = mean, std = std)
                                                                               # [batch_size, seq_len, num_events]

        '''
        the value of probability distribution at t, or p(m, t|\\mathcal{H})
        '''
        probability_for_each_event = - torch.autograd.grad(
            outputs = probability_integral_from_t_to_infinite,
            inputs = time_next,
            grad_outputs = torch.ones_like(probability_integral_from_t_to_infinite),
            create_graph = True
        )[0]                                                                   # [batch_size, seq_len, num_events]
        time_next.requires_grad = False
        check_tensor(probability_for_each_event)                               # [batch_size, seq_len, num_events]
        check_tensor(probability_integral_from_t_to_infinite)                  # [batch_size, seq_len, num_events]

        '''
        Remove the probability of the dummy event by mask.
        '''
        mask_next_without_dummy = self.remove_dummy_event_from_mask(mask_next) # [batch_size, seq_len]
        events_next_without_dummy = events_next * mask_next_without_dummy      # [batch_size, seq_len]
        the_number_of_events = mask_next_without_dummy.sum().item()

        '''
        cross entropy loss between p_{real} and p_{pred}.
        '''
        log_probability_for_each_event_without_dummy = torch.log(probability_for_each_event + self.epsilon)
                                                                               # [batch_size, seq_len, num_events]
        events_probability_without_dummy = torch.nn.functional.softmax(log_probability_for_each_event_without_dummy, dim = -1)
                                                                               # [batch_size, seq_len, num_events]
        events_loss_without_dummy = torch.nn.functional.cross_entropy(rearrange(events_probability_without_dummy, 'b s ne -> b ne s'), \
                                                                                events_next_without_dummy.long(), reduction = 'none')
                                                                               # [batch_size, seq_len]
        events_loss_without_dummy = events_loss_without_dummy * mask_next_without_dummy
                                                                               # [batch_size, seq_len]
        events_loss = events_loss_without_dummy.sum()

        # Time loss: -log p(t) = \\sum_{i = 1}^{N}{\\lambda_{k}(t_i)} + \\int_{t_0}^{t_N}{\\sum_{k}\\lambda_k^(\\tau)d\\tau}
        # time_loss_without_dummy = self.nll_loss(probability = probability_for_each_event, \
        #                                         mask_next = mask_next_without_dummy, events_next = events_next_without_dummy)
        time_loss_without_dummy = self.nll_loss_oversample(probability = probability_for_each_event, \
                                                           mask_next = mask_next_without_dummy, \
                                                           events_next_without_dummy = events_next_without_dummy, \
                                                           events_next = events_next_without_dummy)
        time_loss_survival = 0
        if self.survival_loss_during_training:
            # Survival probability: \\int_{t_N}^{T}{\\sum_{k}\\lambda_k^(\\tau)d\\tau} = -\\log(1 - P(t)) = -log(IFIB-C(t)).
            dummy_event_index = mask_next.sum(dim = -1) - 1                    # [batch_size]
            probability_survival = probability_integral_from_t_to_infinite.sum(dim = -1).gather(index = dummy_event_index.unsqueeze(dim = -1), dim = -1)
                                                                               # [batch_size, 1]
            # The experiment result shows that the existence of probability_survival could significantly damage the performance on the synthetic dataset.
            # Given other models are not affected, it is highly possible that I calculate the wrong survival loss.
            # However, I have no idea why I am wrong and what the correct one should be.
            time_loss_survival = -torch.log(probability_survival).sum()

        loss = time_loss_without_dummy + time_loss_survival

        # we need time_loss_without_dummy to compare our distribution against the ground truth.
        return loss, time_loss_without_dummy, events_loss, the_number_of_events
    

    def evaluate_procedure(self, input_time, input_events, mask, mean, std):
        time_history, time_next = self.divide_history_and_next(input_time)     # 2 * [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # 2 * [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)           # 2 * [batch_size, seq_len]
        
        '''
        Remove the probability of the dummy event by mask.
        '''
        mask_next_without_dummy = self.remove_dummy_event_from_mask(mask_next) # [batch_size, seq_len]
        events_next_without_dummy = events_next * mask_next_without_dummy      # [batch_size, seq_len]
        the_number_of_events = mask_next_without_dummy.sum().item()
        
        mae, f1 = self.mean_absolute_error_and_f1(events_history = events_history, time_history = time_history,\
                                                  events_next = events_next, time_next = time_next, \
                                                  mask_history = mask_history, mask_next = mask_next_without_dummy, \
                                                  mean = mean, std = std)      # 2 * [batch_size, seq_len]
        mae = mae.sum().item() / the_number_of_events

        time_next = repeat(time_next, 'b s -> b s ne', ne = self.num_events)   # [batch_size, seq_len, num_events]
        time_next.requires_grad = True                                         # [batch_size, seq_len, num_events]
        probability_integral_from_time_next_to_infinite = self.model('default_forward', events_history, time_history, time_next, mean = mean, std = std)
                                                                               # [batch_size, seq_len, num_events]
        probability_for_each_event_at_time_next = - torch.autograd.grad(
            outputs = probability_integral_from_time_next_to_infinite,
            inputs = time_next,
            grad_outputs = torch.ones_like(probability_integral_from_time_next_to_infinite)
        )[0]                                                                   # [batch_size, seq_len, num_events]
        time_next.requires_grad = False

        '''
        Event loss. Event predictions are made with time predictions at time_next.
        '''
        log_probability_for_each_event_at_time_next = torch.log(probability_for_each_event_at_time_next + self.epsilon)
                                                                               # [batch_size, seq_len, num_events]
        events_probability = torch.nn.functional.softmax(log_probability_for_each_event_at_time_next, dim = -1)
                                                                               # [batch_size, seq_len, num_events]
        events_loss = torch.nn.functional.cross_entropy(rearrange(events_probability, 'b s ne -> b ne s'), \
                                                                  events_next_without_dummy.long(), reduction = 'none')
                                                                               # [batch_size, seq_len]
        events_loss = events_loss * mask_next_without_dummy                    # [batch_size, seq_len]
        events_loss = events_loss.sum()

        # Time loss: -log p(t) = \\sum_{i = 1}^{N}{\\lambda_{k}(t_i)} + \\int_{t_0}^{t_N}{\\sum_{k}\\lambda_k^(\\tau)d\\tau}
        # time_loss_without_dummy = self.nll_loss(probability = probability_for_each_event_at_time_next, mask_next = mask_next_without_dummy, events_next = events_next_without_dummy)
        time_loss_without_dummy = self.nll_loss_oversample(probability = probability_for_each_event_at_time_next, \
                                                           mask_next = mask_next_without_dummy, \
                                                           events_next_without_dummy = events_next_without_dummy, \
                                                           events_next = events_next_without_dummy)
        # Survival probability: \\int_{t_N}^{T}{\\sum_{k}\\lambda_k^(\\tau)d\\tau} = -\\log(1 - P(t)) = -log(\\sum_{m}{IFIB-C(m, t)}).
        dummy_event_index = mask_next.sum(dim = -1) - 1                        # [batch_size]
        probability_survival = probability_integral_from_time_next_to_infinite.sum(dim = -1).gather(index = dummy_event_index.unsqueeze(dim = -1), dim = -1)
                                                                               # [batch_size, 1]
        time_loss_survival = -torch.log(probability_survival + self.epsilon).mean()

        return time_loss_without_dummy, time_loss_survival, events_loss, mae, f1, the_number_of_events


    def nll_loss(self, probability, events_next, mask_next):
        '''
        The definition of loss.
    
        Args:
            probability:        [batch_size, seq_len, num_events]
            events_next:        [batch_size, seq_len]
            mask_next:          [batch_size, seq_len]
        '''
        probability_mask = torch.nn.functional.one_hot(events_next.long(), num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]
        log_probability = - torch.log(probability + self.epsilon) * probability_mask
        log_probability = reduce(log_probability, '... ne -> ...', 'sum')      # [batch_size, seq_len]

        loss = log_probability * mask_next                                     # [batch_size, seq_len]
        loss = torch.sum(loss)

        return loss


    def nll_loss_oversample(self, probability, events_next, events_next_without_dummy, mask_next):
        '''
        The definition of loss.
    
        Args:
            probability:        [batch_size, seq_len, num_events]
            events_next:        [batch_size, seq_len]
            mask_next:          [batch_size, seq_len]
        '''
        # Balance the loss of different marks.
        marks, counts = torch.unique(events_next, return_counts = True)
        marks_counts_with_unobserved = {}
        for mark in range(self.num_events):
            if mark in marks:
                marks_counts_with_unobserved[mark] = counts[marks == mark]
        
        min_num = min(marks_counts_with_unobserved.values())
        for mark, num in marks_counts_with_unobserved.items():
            marks_counts_with_unobserved[mark] = min_num/num
        
        probability_mask = torch.nn.functional.one_hot(events_next_without_dummy.long(), num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]
        log_probability = - torch.log(probability + self.epsilon) * probability_mask
        log_probability = reduce(log_probability, '... ne -> ...', 'sum')      # [batch_size, seq_len]
        
        loss = 0
        for mark in marks_counts_with_unobserved.keys():
            selected_loss = log_probability[events_next == mark] * marks_counts_with_unobserved[mark]
            loss += selected_loss.sum()
        
        # loss = log_probability * mask_next                                   # [batch_size, seq_len]
        # loss = torch.sum(loss)

        return loss


    '''
    Loss functions
    '''
    def loss_function_undersample(self, integral_all_events, intensity_all_events, events_next_without_dummy, events_next, mask_next):
        
        # Balance the loss of different marks.
        marks, counts = torch.unique(events_next, return_counts = True)
        marks_counts_with_unobserved = {}
        for mark in range(self.num_events):
            if mark in marks:
                marks_counts_with_unobserved[mark] = counts[marks == mark]
            else:
                marks_counts_with_unobserved[mark] = 0
        
        min_num = max(min(marks_counts_with_unobserved.values()), 1)
        
        """ Log-likelihood of sequence. """
        type_mask = F.one_hot(events_next_without_dummy, num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]
        '''
        MTPP loss function
        '''
        selected_intensity = (intensity_all_events * type_mask).sum(dim = -1)  # [batch_size, seq_len]
        log_intensity = torch.log(selected_intensity + self.epsilon)           # [batch_size, seq_len]
        nll = -log_intensity + integral_all_events.sum(dim = -1)               # [batch_size, seq_len]
    
        mtpp_loss = 0
        for mark in marks_counts_with_unobserved.keys():
            selected_loss = nll[events_next == mark][:min_num]
            mtpp_loss += selected_loss.sum()
    
        # mtpp_loss = torch.sum(nll * mask_next)

        '''
        Event loss function. Only for evaluation, do NOT use this loss as a part of the training loss.
        '''
        events_prediction_probability = torch.log(intensity_all_events + self.epsilon)
                                                                               # [batch_size, seq_len, num_events]
        events_prediction_probability = F.softmax(events_prediction_probability, dim = -1)
                                                                               # [batch_size, seq_len, num_events]
        reshaped_events_prediction_probability = rearrange(events_prediction_probability, 'b s ne -> b ne s')
                                                                               # [batch_size, num_events, seq_len]
        events_loss = F.cross_entropy(input = reshaped_events_prediction_probability, target = events_next_without_dummy, reduction = 'none')
                                                                               # [batch_size, seq_len]
        events_loss = (events_loss * mask_next).sum()

        return mtpp_loss, events_loss
    
    
    def sample_time(self, *args, **kwargs):
        return conditional_decorator(torch.compile, False, sample_time)(self, *args, **kwargs)


    sample_time_event = sample_time_event
    sample_event_time = sample_event_time
    
    
    def mean_absolute_error_and_f1(self, events_history, time_history, events_next, time_next, mask_history, mask_next, mean, std):
        pred_time = self.sample_time(sampling_approach = 'its', task = 'tm',
                                     events_history = events_history, time_history = time_history,
                                     number_of_total_samples = self.sample_rate, step = self.mae_step, mean = mean, std = std)
                                                                               # [sample_rate, batch_size, seq_len] * 2
        
        pred_time = pred_time.mean(dim = 0)                                    # [batch_size, seq_len]
        mae = torch.abs(pred_time - time_next) * mask_next                     # [batch_size, seq_len]

        time_next_pred = repeat(pred_time, 'b s -> b s ne', ne = self.num_events)
                                                                               # [batch_size, seq_len, num_events]
        time_next_pred.requires_grad = True                                    # [batch_size, seq_len, num_events]
        probability_integral_from_pred_to_infinite = self.model('default_forward', events_history, time_history, time_next_pred, mean = mean, std = std)
                                                                               # [batch_size, seq_len, num_events]
        probability_for_each_event = - torch.autograd.grad(
            outputs = probability_integral_from_pred_to_infinite,
            inputs = time_next_pred,
            grad_outputs = torch.ones_like(probability_integral_from_pred_to_infinite)
        )[0]                                                                   # [batch_size, seq_len, num_events]
        time_next_pred.requires_grad = False                                   # [batch_size, seq_len, num_events]

        events_pred_index = predict_event(probability_for_each_event)[mask_next == 1]
        events_true = events_next[mask_next == 1]
        events_pred_index, events_true = move_from_tensor_to_ndarray(events_pred_index, events_true)
        f1 = f1_score(y_true = events_true, y_pred = events_pred_index, average = 'macro')
        
        return mae, f1


    def mean_absolute_error_e(self, events_history, events_next, time_history, time_next, mask_next, mean, std, return_mean = True):
        '''
        Well...We will do something totally different by performing event-wise MAE.
        First, predict the event types by \\int_{t_i}^{+\\infty}{\\lambda^*_i(t)\\exp(-\\int_{t_0}^{\\tau}{\\lambda^*_i(t)dt})d\\tau}
        Next, given time predictions. (Expectation? or probability bigger than 0.5?)
        '''
        time_zero = torch.zeros_like(time_next)                                # [batch_size, seq_len]
        # preparing for multi-event training when needed
        time_zero = repeat(time_zero, 'b s -> b s ne', ne = self.num_events)   # [batch_size, seq_len, num_events]

        probability_integral_from_zero_to_infinite = self.model('default_forward', events_history, time_history, time_zero, mean = mean, std = std)
                                                                               # [batch_size, seq_len, num_events]

        probability_integral_sum = reduce(probability_integral_from_zero_to_infinite, 'b s ne -> b s', 'sum')
                                                                               # [batch_size, seq_len]
        predict_index = torch.argmax(probability_integral_from_zero_to_infinite, dim = -1)
                                                                               # [batch_size, seq_len]
        f1, top_k_acc = get_f1_and_top_k_acc_in_mae_e(events_next, probability_integral_from_zero_to_infinite, mask_next, self.num_events)

        predict_index_one_hot = torch.nn.functional.one_hot(predict_index.long(), num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]
        events_next_one_hot = torch.nn.functional.one_hot(events_next.long(), num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]
        # step 2: get the time prediction for that kind of event
        tau_pred_all_event = self.sample_time(sampling_approach = 'its', task = 'mt', \
                                              events_history = events_history, time_history = time_history, \
                                              p_m = probability_integral_from_zero_to_infinite, \
                                              number_of_total_samples = self.sample_rate, step = self.mae_e_step, \
                                              inf_val = 1e6, mean = mean, std = std)
                                                                               # [sample_rate, batch_size, seq_len, num_events]
        
        if return_mean:
            tau_pred_all_event = tau_pred_all_event.mean(dim = 0)              # [batch_size, seq_len, num_events]
            mae_per_event_pure_predict = torch.abs((tau_pred_all_event * predict_index_one_hot).sum(dim = -1) - time_next) * mask_next
                                                                               # [batch_size, seq_len]
            mae_per_event = torch.abs((tau_pred_all_event * events_next_one_hot).sum(dim = -1) - time_next) * mask_next
                                                                               # [batch_size, seq_len]
    
            mae_per_event_pure_predict_avg = torch.sum(mae_per_event_pure_predict, dim = -1) / mask_next.sum(dim = -1)
            mae_per_event_avg = torch.sum(mae_per_event, dim = -1) / mask_next.sum(dim = -1)
        else:
            mae_per_event_pure_predict = torch.abs((tau_pred_all_event * predict_index_one_hot.unsqueeze(dim = 0)).sum(dim = -1) - time_next) * mask_next.unsqueeze(dim = 0)
                                                                               # [sample_rate, batch_size, seq_len]
            mae_per_event = torch.abs((tau_pred_all_event * events_next_one_hot.unsqueeze(dim = 0)).sum(dim = -1) - time_next) * mask_next.unsqueeze(dim = 0)
                                                                               # [sample_rate, batch_size, seq_len]
    
            mae_per_event_pure_predict_avg = torch.sum(mae_per_event_pure_predict, dim = -1) / mask_next.sum(dim = -1)
                                                                               # [sample_rate, batch_size]
            mae_per_event_avg = torch.sum(mae_per_event, dim = -1) / mask_next.sum(dim = -1)
                                                                               # [sample_rate, batch_size]
            
            # Calculate mean
            mae_per_event_pure_predict = mae_per_event_pure_predict.mean(dim = 0)
                                                                               # [batch_size, seq_len]
            mae_per_event = mae_per_event.mean(dim = 0)                        # [batch_size, seq_len]
            mae_per_event_pure_predict_avg = mae_per_event_pure_predict_avg.mean(dim = 0)
                                                                               # [batch_size]
            mae_per_event_avg = mae_per_event_avg.mean(dim = 0)                # [batch_size]


        return f1, top_k_acc, probability_integral_sum, probability_integral_from_zero_to_infinite, \
               tau_pred_all_event, (mae_per_event_pure_predict_avg, mae_per_event_avg), \
               (mae_per_event_pure_predict, mae_per_event)
    

    def mean_absolute_error_e_and_f1_by_time_event(self, events_history, time_history, events_next, time_next, mask_history, mask_next, mean, std):
        sample_rate = 10000
        sample_step = 1000
        
        pred_time = self.sample_time(sampling_approach = 'its', task = 'tm', \
                                     events_history = events_history, time_history = time_history,
                                     number_of_total_samples = sample_rate, step = sample_step, mean = mean, std = std)
                                                                               # [sample_rate, batch_size, seq_len]
        # Preprocess
        sample_rate_list = step_split(sample_rate, sample_step)
        probability_for_each_event = []

        time_next_pred = repeat(pred_time, '... b s -> ... b s ne', ne = self.num_events)
                                                                               # [sample_rate, batch_size, seq_len, num_events]
        time_next_pred.requires_grad = True                                    # [sample_rate, batch_size, seq_len, num_events]
        for idx, samples in enumerate(sample_rate_list):
            selected_time_next_pred = time_next_pred[idx * sample_step:idx * sample_step + samples]
            probability_integral_from_pred_to_infinite = self.model('default_forward', events_history, time_history, \
                                                                    selected_time_next_pred, mean = mean, std = std)
                                                                               # [sample_rate, batch_size, seq_len, num_events]
            probability_for_each_event.append(- torch.autograd.grad(
                outputs = probability_integral_from_pred_to_infinite,
                inputs = selected_time_next_pred,
                grad_outputs = torch.ones_like(probability_integral_from_pred_to_infinite)
            )[0])                                                              # [sample_rate, batch_size, seq_len, num_events]
        time_next_pred.requires_grad = False                                   # [sample_rate, batch_size, seq_len, num_events]
        probability_for_each_event = torch.concat(probability_for_each_event)  # [sample_rate, batch_size, seq_len, num_events]

        events_pred_index = torch.argmax(probability_for_each_event, dim = -1) # [sample_rate, batch_size, seq_len]
        pred_events = torch.mode(events_pred_index, dim = 0).values            # [batch_size, seq_len]
        pred_events = pred_events[mask_next == 1]
        events_true = events_next[mask_next == 1]

        # f1
        pred_events, events_true = move_from_tensor_to_ndarray(pred_events, events_true)
        f1 = f1_score(y_true = events_true, y_pred = pred_events, average = 'macro')
        
        # MAE-E
        select_mask = events_pred_index == events_next.unsqueeze(dim = 0)      # [sample_rate, batch_size, seq_len]
        
        picked_time = pred_time * select_mask.int()                            # [sample_rate, batch_size, seq_len]
        predicted_time = picked_time.sum(dim = 0) / select_mask.int().sum(dim = 0)
                                                                               # [batch_size, seq_len]
        
        mae = torch.abs(predicted_time - time_next)                            # [batch_size, seq_len]
        mae = torch.nan_to_num(mae, 1e6)                                       # [batch_size, seq_len]

        return mae, f1, events_pred_index, events_next


    def extract_plot_data(self, minibatch):
        '''
        This function extracts input_time, input_events, input_intensity, mask, mean, and std from the minibatch.
        Caution: dataloader won't add the end dummy event during evaluation!

        Args:
        * minibatch  type: list shape: [[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]
                     data structure: [[input_time, input_events, score, mask], (mean, std)]
        
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
        * std           type: int shape: N/A
                        The mean of all $ t_i - t_{i - 1} $ in the entire dataset. Dataloader is responsible to provide
                        this value if needed.
        '''
        input_time, input_events, _, mask, input_intensity = minibatch[0]
        mean, std = minibatch[1]

        return input_time, input_events, input_intensity, mask, mean, std
    

    def figure_intensity(self, input_data, opt):
        '''
        Function prober, used by tpp_ploter to draw plots.

        Args:
        * input_data  type: list shape: [[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]
                      The original minibatch. Detailed information is available in extract_plot_data()
        * resolution  type: int shape: N/A
                      How many interpretive numbers we have between an event interval?
        '''

        return NotImplementedError('IFIB is intensity-free. Therefore, it can not provide the plot for the intensity function.')


    def figure_integral(self, input_data, opt):
        '''
        Function prober, used by tpp_ploter to draw plots.

        Args:
        * input_data  type: list shape: [[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]
                      The original minibatch. Detailed information is available in extract_plot_data()
        * resolution  type: int shape: N/A
                      How many interpretive numbers we have between an event interval?
        '''
        return NotImplementedError('IFIB is intensity-free. Therefore, it can not provide the plot for the intensity integral.')


    def figure_probability(self, input_data, opt):
        '''
        Function prober, used by tpp_ploter to draw plots.

        Args:
        * input_data  type: list shape: [[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]
                      The original minibatch. Detailed information is available in extract_plot_data()
        * resolution  type: int shape: N/A
                      How many interpretive numbers we have between an event interval?
        '''
        input_time, input_events, input_intensity, mask, mean, std = self.extract_plot_data(input_data)
        
        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # [batch_size, seq_len]
        _, mask_next = self.divide_history_and_next(mask)                      # [batch_size, seq_len]

        expand_probability, timestamp = \
            self.model.probability(events_history, time_history, time_next, opt.resolution, mean, std)
                                                                               # [batch_size, seq_len, resolution, num_events]
        data = {
            'time_next': time_next,
            'events_next': events_next,
            'mask_next': mask_next,
            'expand_probability': expand_probability,
            'input_intensity': input_intensity
            }
        
        generate_probability_figure(data, timestamp, opt)


    def figure_debug(self, input_data, opt):
        '''
        Args:
        time: [batch_size(always 1), seq_len + 1]
              The original dataset records. 
        resolution: int
              How many interpretive numbers we have between an event interval?
        '''
        input_time, input_events, input_intensity, mask, mean, std = self.extract_plot_data(input_data)

        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]

        mae, f1_1 = self.mean_absolute_error_and_f1(events_history, time_history, events_next, \
                                                    time_next, mask_history, mask_next, mean, std)
                                                                               # [batch_size, seq_len]
        data, timestamp = self.model.model_probe_function(events_history, time_history, time_next, opt.resolution, mean, std, mask_next)

        f1_2, top_k, probability_sum, _, tau_pred_all_event, maes_avg, maes \
            = self.mean_absolute_error_e(events_history, events_next, time_history, time_next, mask_next, mean, std, return_mean = False)

        '''
        We show how porobability distribution goes on two sampled sequences, one following the event-time routine, and the other following
        the time-event routine.
        '''
        time_history_for_sampling_event_time, events_history_for_sampling_event_time, sampled_mask_event_time \
            = self.sample_event_time(None, None, mean, std, end_sampling_requirement = 'time_and_event_num', \
                                     number_of_sampled_sequences = 1, end_time = self.end_time - self.start_time, max_seq_len = 250)
                                                                               # 3 * [number_of_sampled_sequences, length_of_sampled_sequences]

        sampled_time_history_event_time, sampled_time_next_event_time = self.divide_history_and_next(time_history_for_sampling_event_time)
                                                                               # 2 * [batch_size, seq_len]
        sampled_events_history_event_time, sampled_events_next_event_time = self.divide_history_and_next(events_history_for_sampling_event_time)
                                                                               # 2 * [batch_size, seq_len]
        _, sampled_mask_next_event_time = self.divide_history_and_next(sampled_mask_event_time)
                                                                               # 2 * [batch_size, seq_len]
#
        sampled_data_event_time, sampled_timestamp_event_time \
            = self.model.model_probe_function(sampled_events_history_event_time, sampled_time_history_event_time, \
                                              sampled_time_next_event_time, opt.resolution, mean, std, sampled_mask_next_event_time)


        time_history_for_sampling_time_event, events_history_for_sampling_time_event, sampled_mask_time_event \
            = self.sample_time_event(None, None, mean, std, end_sampling_requirement = 'time_and_event_num', \
                                     number_of_sampled_sequences = 1, end_time = self.end_time - self.start_time, max_seq_len = 250)
                                                                               # 3 * [number_of_sampled_sequences, length_of_sampled_sequences]

        sampled_time_history_time_event, sampled_time_next_time_event = self.divide_history_and_next(time_history_for_sampling_time_event)
                                                                               # 2 * [batch_size, seq_len]
        sampled_events_history_time_event, sampled_events_next_time_event = self.divide_history_and_next(events_history_for_sampling_time_event)
                                                                               # 2 * [batch_size, seq_len]
        _, sampled_mask_next_time_event = self.divide_history_and_next(sampled_mask_time_event)
                                                                               # 2 * [batch_size, seq_len]

        sampled_data_time_event, sampled_timestamp_time_event \
            = self.model.model_probe_function(sampled_events_history_time_event, sampled_time_history_time_event, \
                                              sampled_time_next_time_event, opt.resolution, mean, std, sampled_mask_next_time_event)
    
        '''
        Append additional info into the data dict.
        '''
        data['events_next'] = events_next
        data['time_next'] = time_next
        data['mask_next'] = mask_next
        data['f1_after_time_pred'] = f1_1
        data['f1_before_time_pred'] = f1_2
        data['top_k'] = top_k
        data['probability_sum'] = probability_sum
        data['tau_pred_all_event'] = tau_pred_all_event
        data['mae_before_event'] = mae
        data['maes_after_event_avg'] = maes_avg
        data['maes_after_event'] = maes
        
        '''
        Show the event sequence sampled from p(t) and p(m|t)
        '''
        data['sampled_events_next_event_time'] = sampled_events_next_event_time
        data['sampled_time_next_event_time'] = sampled_time_next_event_time
        data['sampled_mask_next_event_time'] = sampled_mask_next_event_time
        data['sampled_timestamp_event_time'] = sampled_timestamp_event_time
        data['sampled_subprobability_event_time'] = sampled_data_event_time['expand_probability_for_each_event']
        '''
        Show the event sequence sampled from p(m) and p(t|m)
        '''
        data['sampled_events_next_time_event'] = sampled_events_next_time_event
        data['sampled_time_next_time_event'] = sampled_time_next_time_event
        data['sampled_mask_next_time_event'] = sampled_mask_next_time_event
        data['sampled_timestamp_time_event'] = sampled_timestamp_time_event
        data['sampled_subprobability_time_event'] = sampled_data_time_event['expand_probability_for_each_event']

        generate_debug_figure(data, timestamp, opt)


    '''
    Evaluation over the entire dataset.
    '''
    def get_spearman_and_l1(self, input_data, opt):
        input_time, input_events, input_intensity, mask, mean, std = self.extract_plot_data(input_data)
        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history, _ = self.divide_history_and_next(input_events)         # [batch_size, seq_len]
        _, mask_next = self.divide_history_and_next(mask)                      # [batch_size, seq_len]

        expand_probability, timestamp = \
            self.model.probability(events_history, time_history, time_next, opt.resolution, mean, std)
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
        input_time, input_events, input_intensity, mask, mean, std = self.extract_plot_data(input_data)
        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]

        mae, f1_1 = self.mean_absolute_error_and_f1(events_history, time_history, events_next, \
                                                    time_next, mask_history, mask_next, mean, std)
                                                                               # [batch_size, seq_len]
        mae = move_from_tensor_to_ndarray(mae)

        return mae, f1_1, events_next

    
    def get_mae_e_and_f1(self, input_data, opt):
        input_time, input_events, input_intensity, mask, mean, std = self.extract_plot_data(input_data)
        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]

        f1_2, top_k, probability_sum, probability_integral_from_zero_to_infinite, tau_pred_all_event, maes_avg, maes \
            = self.mean_absolute_error_e(events_history, events_next, time_history, time_next, mask_next, mean, std)
        
        _, maes, probability_sum, probability_integral_from_zero_to_infinite, events_next = move_from_tensor_to_ndarray(*maes, probability_sum, probability_integral_from_zero_to_infinite, events_next)

        return maes, f1_2, probability_sum, probability_integral_from_zero_to_infinite, events_next


    def get_mae_e_and_f1_by_time_event(self, input_data, opt):
        input_time, input_events, input_intensity, mask, mean, std = self.extract_plot_data(input_data)
        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]

        maes, f1_2, events_pred_index, events_next \
            = self.mean_absolute_error_e_and_f1_by_time_event(events_history, time_history, events_next, \
                                                              time_next, mask_history, mask_next, mean, std)
        
        maes, events_pred_index, events_next = move_from_tensor_to_ndarray(maes, events_pred_index, events_next)

        return maes, f1_2, events_pred_index, events_next


    def get_which_event_first(self, input_data, opt):
        '''
        Hyperparameters
        '''
        the_number_of_samples = 10000
        substep = 10000

        input_time, input_events, input_intensity, mask, mean, std = self.extract_plot_data(input_data)
        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]
    
        time_zero = torch.zeros_like(time_next)                                # [batch_size, seq_len]
        # preparing for multi-event training when needed
        time_zero = repeat(time_zero, 'b s -> b s ne', ne = self.num_events)   # [batch_size, seq_len, num_events]
        probability_integral_from_zero_to_infinite = self.model('default_forward', events_history, time_history, time_zero, mean = mean, std = std)
                                                                               # [batch_size, seq_len, num_events]    
        # step 2: get the time prediction for that kind of event
        tau_pred_all_event = self.sample_time(sampling_approach = 'its', task = 'mt', \
                                              events_history = events_history, time_history = time_history, \
                                              p_m = probability_integral_from_zero_to_infinite, \
                                              number_of_total_samples = the_number_of_samples, step = substep, \
                                              inf_val = 1e6, mean = mean, std = std)
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
        input_time, input_events, input_intensity, mask, mean, std = self.extract_plot_data(input_data)
        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # [batch_size, seq_len]
        the_number_of_samples = 3000
        substep = 3000

        time_zero = torch.zeros_like(time_next)                                # [batch_size, seq_len]
        # preparing for multi-event training when needed
        time_zero = repeat(time_zero, 'b s -> b s ne', ne = self.num_events)   # [batch_size, seq_len, num_events]
    
        probability_integral_from_zero_to_infinite = self.model('default_forward', events_history, time_history, time_zero, mean = mean, std = std)
                                                                               # [batch_size, seq_len, num_events]    
        # step 2: get the time prediction for that kind of event
        tau_pred_all_event = self.sample_time(sampling_approach = 'its', task = 'mt', \
                                              events_history = events_history, time_history = time_history, \
                                              p_m = probability_integral_from_zero_to_infinite, \
                                              number_of_total_samples = the_number_of_samples, step = substep, \
                                              inf_val = 1e6, mean = mean, std = std)
                                                                               # [sample_rate, batch_size, seq_len, num_events]

        return tau_pred_all_event, probability_integral_from_zero_to_infinite


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
            # CAUTION: this is the integral of p(m, t) from t to the positive infinity, or 1 - F(m, t)
            # 1 - F(t) = \Gamma(t) = exp(-\Lambda(t))
            # <=> \Lambda(t) = - ln \Gamma(t)
            obtained_integral = self.model('default_forward', obs_events_history_for_one_seq, obs_time_history_for_one_seq.float(), obs_time_next_for_one_seq.float(), mean, std)
                                                                               # [num_samples, seq_len, num_events]
            
            integral_sum = - torch.log(obtained_integral).sum(dim = -1)        # [num_samples, seq_len]
            
            all_roauc_area = []
            for integral_sum_per_seq_per_sample, missing_mask_for_one_seq_per_sample in \
                zip(integral_sum, missing_mask_for_one_seq):
                
                sample_len = len(missing_mask_for_one_seq_per_sample)
                selected_integral_sum_per_seq_per_sample = move_from_tensor_to_ndarray(integral_sum_per_seq_per_sample[:sample_len])
                
                roauc_area = roc_auc_score(y_true = missing_mask_for_one_seq_per_sample, y_score = selected_integral_sum_per_seq_per_sample)
                all_roauc_area.append(roauc_area)
            
            roc_result.append(np.mean(all_roauc_area))
        
        roc_result = np.array(roc_result)
        return roc_result


    def generate_hypro_dataset(self, input_data, opt):
        # CAUTION: Only works when batch_size = 1.
        
        input_time, input_events, input_intensity, mask, mean, std = self.extract_plot_data(input_data)        
        
        if mask.sum(dim = -1) <= opt.number_of_events_hypro:
            '''
            Sequence too short to perform HYPRO. Considering to make the number_of_events_hypro lower to avoid this.
            '''
            return None
        
        time_history_for_sampling = repeat(input_time[..., :-opt.number_of_events_hypro], '() ... -> nns ...', nns = opt.number_of_negative_samples)
                                                                               # [number_of_negative_samples, seq_len - opt.number_of_events_hypro]
        event_history_for_sampling = repeat(input_events[..., :-opt.number_of_events_hypro], '() ... -> nns ...', nns = opt.number_of_negative_samples)
                                                                               # [number_of_negative_samples, seq_len - opt.number_of_events_hypro]
        
        tau_sampled, events_sampled, _, \
            = self.sample_event_time(time_history_for_sampling, event_history_for_sampling, mean, std, \
                                     end_sampling_requirement = 'event_num', max_seq_len = mask.sum(dim = -1))
                                                                               # [number_of_negative_samples, seq_len]
        
        input_time, input_events, tau_sampled, events_sampled = move_from_tensor_to_ndarray(input_time, input_events, tau_sampled, events_sampled)
        
        return input_time, input_events, tau_sampled, events_sampled


    '''
    All static methods
    '''
    def train_step(model, minibatch, device):
        ''' 
        Epoch operation in training phase.
        The input minibatch comprise time sequences.

        Args:
            minibatch: [batch_size, seq_len]
                       contains [time_seq, event_seq, score, mask]
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
        ''' Epoch operation in evaluation phase '''
        model.eval()
        
        [time_seq, event_seq, score, mask], (mean, std) = minibatch
        time_loss_without_dummy, time_loss_survival, events_loss, mae, f1, the_number_of_events \
            = model(task_name = 'evaluate', input_time = time_seq, input_events = event_seq, 
                    mask = mask, mean = mean, std = std)
    
        time_loss_without_dummy = time_loss_without_dummy.item() / the_number_of_events
        time_loss_survival = time_loss_survival.item()
        events_loss = events_loss.item() / the_number_of_events
        fact = score.sum().item() / the_number_of_events
        
        return time_loss_without_dummy, time_loss_survival, fact, events_loss, mae, f1


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
    

    def log_print_format(input, procedure):
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
            format_dict['f1'] = pack_one_value_to_dict(input[5], '2.8f')
            return format_dict
        
        return (train_log_print_format(input) if procedure == 'Training' else test_log_print_format(input))

    format_dict_length = 6
    
    def choose_metric(evaluation_report_format_dict, test_report_format_dict):
        '''
        [relative loss on evaluation dataset, relative loss on test dataset, event loss on test dataset]
        '''
        return [evaluation_report_format_dict['absolute_NLL_loss'], 
                test_report_format_dict['absolute_NLL_loss']], \
               ['evaluation_absolute_loss', 'test_absolute_loss']
    
    metric_number = 2 # metric number is the length of the output of choose_metric