import torch, copy
from sklearn.metrics import f1_score
from einops import rearrange, repeat, reduce, pack
from scipy.stats import spearmanr

from src.TPP.model.basic_tpp_model import BasicModel, its_lower_bound, its_upper_bound
from src.TPP.model.ifib_c.submodel import IFIBC
from src.TPP.model.utils import *
from src.TPP.model.ifib_c.plot import *


class IFIBCModel(BasicModel):
    def __init__(self, d_history,
                 d_intensity,
                 dropout,
                 history_module_layers,
                 mlp_layers,
                 info_dict,
                 device,
                 removes_tail, tanh_parameter,
                 history_module = 'LSTM', survival_loss_during_training = False,
                 epsilon = 0.0, sample_rate = 32, mae_step = 32, mae_e_step = 32):
        super(IFIBCModel, self).__init__()
        self.device = device
        self.num_events = info_dict['num_events']
        self.start_time = info_dict['t_0']
        self.end_time = info_dict['T']
        self.epsilon = epsilon
        self.survival_loss_during_training = survival_loss_during_training
        self.sample_rate = sample_rate
        self.mae_step = mae_step
        self.mae_e_step = mae_e_step
        self.bisect_early_stop_threshold = 1e-5
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
            'mae_e_and_f1_by_time_event': self.get_mae_e_and_f1_by_time_event,
            'mae_e_and_f1': self.get_mae_e_and_f1,
            'graph': self.plot,
            'which_event_first': self.get_which_event_first,
            'samples_from_et': self.samples_from_et,
        }

        return task_mapper[task_name](*args, **kwargs)
    

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


    def train_procedure(self, input_time, input_events, mask, mean, var):
        time_history, time_next = self.divide_history_and_next(input_time)     # 2 * [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # 2 * [batch_size, seq_len]
        _, mask_next = self.divide_history_and_next(mask)                      # [batch_size, seq_len]

        time_next = repeat(time_next, 'b s -> b s ne', ne = self.num_events)   # [batch_size, seq_len, num_events]
        time_next.requires_grad = True

        '''
        \int_{t}^{+\inf}{p(m, \tau|\mathcal{H})d\tau}
        '''
        probability_integral_from_t_to_infinite = self.model(events_history, time_history, time_next, mean = mean, var = var)
                                                                               # [batch_size, seq_len, num_events]

        '''
        the value of probability distribution at t, or p(m, t|\mathcal{H})
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
        assert probability_for_each_event.shape == probability_integral_from_t_to_infinite.shape

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

        # Time loss: -log p(t) = \sum_{i = 1}^{N}{\lambda_{k}(t_i)} + \int_{t_0}^{t_N}{\sum_{k}\lambda_k^(\tau)d\tau}
        time_loss_without_dummy = self.nll_loss(probability = probability_for_each_event, \
                                                mask_next = mask_next_without_dummy, events_next = events_next_without_dummy)
        time_loss_survival = 0
        if self.survival_loss_during_training:
            # Survival probability: \int_{t_N}^{T}{\sum_{k}\lambda_k^(\tau)d\tau} = -\log(1 - P(t)) = -log(IFIB-C(t)).
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
    

    def evaluate_procedure(self, input_time, input_events, mask, mean, var):
        time_history, time_next = self.divide_history_and_next(input_time)     # 2 * [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # 2 * [batch_size, seq_len]
        _, mask_next = self.divide_history_and_next(mask)                      # [batch_size, seq_len]
        
        '''
        Remove the probability of the dummy event by mask.
        '''
        mask_next_without_dummy = self.remove_dummy_event_from_mask(mask_next) # [batch_size, seq_len
        events_next_without_dummy = events_next * mask_next_without_dummy      # [batch_size, seq_len]
        the_number_of_events = mask_next_without_dummy.sum().item()
        
        mae, pred_time = self.mean_absolute_error(events_history = events_history, time_history = time_history,\
                                                  time_next = time_next, mask_next = mask_next_without_dummy, mean = mean, var = var)
                                                                               # 2 * [batch_size, seq_len]
        mae = mae.sum().item() / the_number_of_events

        time_next = repeat(time_next, 'b s -> b s ne', ne = self.num_events)   # [batch_size, seq_len, num_events]
        pred_time = repeat(pred_time, 'b s -> b s ne', ne = self.num_events)   # [batch_size, seq_len, num_events]
        time_zero = torch.zeros_like(time_next, device = self.device)          # [batch_size, seq_len, num_events]


        time_next.requires_grad = True                                         # [batch_size, seq_len, num_events]
        pred_time.requires_grad = True                                         # [batch_size, seq_len, num_events]

        probability_integral_from_zero_to_infinite = self.model(events_history, time_history, time_zero, mean = mean, var = var)
                                                                               # [batch_size, seq_len, num_events]
        probability_integral_from_pred_time_to_infinite = self.model(events_history, time_history, pred_time, mean = mean, var = var)
                                                                               # [batch_size, seq_len, num_events]
        probability_integral_from_time_next_to_infinite = self.model(events_history, time_history, time_next, mean = mean, var = var)
                                                                               # [batch_size, seq_len, num_events]

        probability_for_each_event_at_pred_time = - torch.autograd.grad(
            outputs = probability_integral_from_pred_time_to_infinite,
            inputs = pred_time,
            grad_outputs = torch.ones_like(probability_integral_from_pred_time_to_infinite)
        )[0]                                                                   # [batch_size, seq_len, num_events]

        probability_for_each_event_at_time_next = - torch.autograd.grad(
            outputs = probability_integral_from_time_next_to_infinite,
            inputs = time_next,
            grad_outputs = torch.ones_like(probability_integral_from_time_next_to_infinite)
        )[0]                                                                   # [batch_size, seq_len, num_events]

        pred_time.requires_grad = False
        time_next.requires_grad = False

        f1_pred_at_pred_time, f1_pred_at_time_next = 0, 0
        '''
        macro-F1 value. Event predictions are made without time predictions.
        '''
        events_true = events_next_without_dummy[mask_next_without_dummy == 1]  # [batch_size, seq_len]
        events_pred_index = torch.argmax(probability_integral_from_zero_to_infinite, dim = -1)[mask_next_without_dummy == 1]
                                                                               # [batch_size, seq_len]
        events_pred_index, events_true = move_from_tensor_to_ndarray(events_pred_index, events_true)
        f1_pred_at_time_next = f1_score(y_true = events_true, y_pred = events_pred_index, average = 'macro')

        '''
        macro-F1 value. Event predictions are made with time predictions at pred_time.
        '''
        events_pred_index_at_pred_time = torch.argmax(probability_for_each_event_at_pred_time, dim = -1)[mask_next_without_dummy == 1]
                                                                               # [batch_size, seq_len]
        events_pred_index_at_pred_time = move_from_tensor_to_ndarray(events_pred_index_at_pred_time)
        f1_pred_at_pred_time = f1_score(y_true = events_true, y_pred = events_pred_index_at_pred_time, average = 'macro')

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

        # Time loss: -log p(t) = \sum_{i = 1}^{N}{\lambda_{k}(t_i)} + \int_{t_0}^{t_N}{\sum_{k}\lambda_k^(\tau)d\tau}
        time_loss_wihtout_dummy = self.nll_loss(probability = probability_for_each_event_at_time_next, mask_next = mask_next_without_dummy, events_next = events_next_without_dummy)
        # Survival probability: \int_{t_N}^{T}{\sum_{k}\lambda_k^(\tau)d\tau} = -\log(1 - P(t)) = -log(\sum_{m}{IFIB-C(m, t)}).
        dummy_event_index = mask_next.sum(dim = -1) - 1                        # [batch_size]
        probability_survival = probability_for_each_event_at_time_next.sum(dim = -1).gather(index = dummy_event_index.unsqueeze(dim = -1), dim = -1)
                                                                               # [batch_size, 1]
        time_loss_survival = -torch.log(probability_survival + self.epsilon).mean()

        return time_loss_wihtout_dummy, time_loss_survival, events_loss, f1_pred_at_time_next, mae, f1_pred_at_pred_time, the_number_of_events


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


    def mean_absolute_error_and_f1(self, events_history, time_history, events_next, time_next, mask_history, mask_next, mean, var):
        mae, pred_time = self.mean_absolute_error(events_history, time_history, time_next, mask_next, mean, var)
                                                                               # [batch_size, seq_len] * 2
        time_next_pred = repeat(pred_time, 'b s -> b s ne', ne = self.num_events)
                                                                               # [batch_size, seq_len, num_events]
        time_next_pred.requires_grad = True                                    # [batch_size, seq_len, num_events]

        probability_integral_from_pred_to_infinite = self.model(events_history, time_history, time_next_pred, mean = mean, var = var)
                                                                               # [batch_size, seq_len, num_events]
        probability_for_each_event = - torch.autograd.grad(
            outputs = probability_integral_from_pred_to_infinite,
            inputs = time_next_pred,
            grad_outputs = torch.ones_like(probability_integral_from_pred_to_infinite)
        )[0]                                                                   # [batch_size, seq_len, num_events]

        events_pred_index = torch.argmax(probability_for_each_event, dim = -1)[mask_next == 1]
        events_true = events_next[mask_next == 1]
        events_pred_index, events_true = move_from_tensor_to_ndarray(events_pred_index, events_true)
        f1 = f1_score(y_true = events_true, y_pred = events_pred_index, average = 'macro')
        
        return mae, f1


    def mean_absolute_error(self, events_history, time_history, time_next, mask_next, mean, var, return_mean = True, sample_rate = None, mae_step = None):
        # Preprocess
        sample_rate = sample_rate if sample_rate is not None else self.sample_rate
        mae_step = mae_step if mae_step is not None else self.mae_step
        sample_rate_list = step_split(sample_rate, mae_step)

        def evaluate(taus, probability_threshold, integral_from_zero_to_inf, ):
            taus = repeat(taus, '... -> ... ne', ne = self.num_events)         # [sample_rate, batch_size, seq_len, num_events]
            probability_integral_from_t_to_inf = self.model(events_history, time_history, taus, mean, var)
                                                                               # [sample_rate, batch_size, seq_len, num_events]
            # P_m(t) = \int_{0}^{t}{p(t|m, \mathcal{H})}
            probability_integral = integral_from_zero_to_inf - probability_integral_from_t_to_inf
                                                                               # [sample_rate, batch_size, seq_len, num_events]
            probability_integral = reduce(probability_integral, '... ne -> ...', 'sum')
                                                                               # [sample_rate, batch_size, seq_len]
            return probability_integral - probability_threshold

        tau_pred = []
        for sub_sample_rate in sample_rate_list:
            probability_threshold = torch.zeros((sub_sample_rate, *time_next.shape), device = self.device)
                                                                               # [sub_sample_rate, batch_size, seq_len]
            torch.nn.init.uniform_(probability_threshold, a = its_lower_bound, b = its_upper_bound)
                                                                               # [sub_sample_rate, batch_size, seq_len]

            time_next_zero = torch.zeros_like(probability_threshold)           # [sub_sample_rate, batch_size, seq_len]
            time_next_zero = repeat(time_next_zero, '... -> ... ne', ne = self.num_events)
                                                                               # [sub_sample_rate, batch_size, seq_len, num_events]

            integral_from_zero_to_inf = self.model(events_history, time_history, time_next_zero, mean = mean, var = var)
                                                                               # [sub_sample_rate, batch_size, seq_len, num_events]
            tau_pred.append(median_prediction(self.max_step, self.bisect_early_stop_threshold, \
                                              evaluate, probability_threshold, integral_from_zero_to_inf))
                                                                               # [sub_sample_rate, batch_size, seq_len]

        tau_pred = torch.cat(tau_pred, dim = 0)                                # [sample_rate, batch_size, seq_len]

        if return_mean:
            tau_pred = tau_pred.mean(dim = 0)                                  # [batch_size, seq_len]
            mae = torch.abs(tau_pred - time_next) * mask_next                  # [batch_size, seq_len]
        else:
            mae = torch.abs(tau_pred - time_next.unsqueeze(dim = 0)) * mask_next.unsqueeze(dim = 0)
                                                                               # [sample_rate, batch_size, seq_len]
            tau_pred = tau_pred * mask_next.unsqueeze(dim = 0)                 # [sample_rate, batch_size, seq_len]

        return mae, tau_pred


    def mean_absolute_error_e(self, events_history, events_next, time_history, time_next, mask_next, mean, var, return_mean = True, sample_rate = None, mae_e_step = None):
        '''
        Well...We will do something totally different by performing event-wise MAE.
        First, predict the event types by \int_{t_i}^{+\infty}{\lambda^*_i(t)\exp(-\int_{t_0}^{\tau}{\lambda^*_i(t)dt})d\tau}
        Next, given time predictions. (Expectation? or probability bigger than 0.5?)
        '''
        time_zero = torch.zeros_like(time_next)                                # [batch_size, seq_len]
        # preparing for multi-event training when needed
        time_zero = repeat(time_zero, 'b s -> b s ne', ne = self.num_events)   # [batch_size, seq_len, num_events]

        probability_integral_from_zero_to_infinite = \
            self.model(events_history, time_history, time_zero, mean = mean, var = var)
                                                                               # [batch_size, seq_len, num_events]

        probability_integral_sum = reduce(probability_integral_from_zero_to_infinite, 'b s ne -> b s', 'sum')
                                                                               # [batch_size, seq_len]
        predict_index = torch.argmax(probability_integral_from_zero_to_infinite, dim = -1)
                                                                               # [batch_size, seq_len]
        
        f1, top_k_acc = get_f1_and_top_k_acc_in_mae_e(events_next, self.num_events, probability_integral_from_zero_to_infinite)


        predict_index_one_hot = torch.nn.functional.one_hot(predict_index.long(), num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]
        events_next_one_hot = torch.nn.functional.one_hot(events_next.long(), num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]

        # step 2: get the time prediction for that kind of event
        tau_pred_all_event = self.prediction_with_all_event_types(events_history, time_history, probability_integral_from_zero_to_infinite, \
                                                                  mean, var, return_mean, sample_rate, mae_e_step)
                                                                               # [batch_size, seq_len, num_events] if return_mean = True else [sample_rate, batch_size, seq_len, num_events]
        
        if return_mean:
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


    def prediction_with_all_event_types(self, events_history, time_history, p_m, mean, var, return_mean, sample_rate, mae_e_step):
        '''
        The input should be the original minibatch
        MAE evaluation part, dwg and fullynn exclusive
        '''
        # Preprocess
        sample_rate = sample_rate if sample_rate is not None else self.sample_rate
        mae_e_step = mae_e_step if mae_e_step is not None else self.mae_e_step
        sample_rate_list = step_split(sample_rate, mae_e_step)

        def bisect_target(taus, probability_threshold):
            # \int_{tau}^{+\inf}{p(m, \tau|\mathcal{H})d\tau}
            probability_integral_from_t_to_infinite = self.model(events_history, time_history, taus, mean = mean, var = var)
                                                                               # [sample_rate, batch_size, seq_len, num_events]
            # \int_{0}^{tau}{p(m, \tau|\mathcal{H})d\tau}
            p_mt = p_m - probability_integral_from_t_to_infinite               # [sample_rate, batch_size, seq_len, num_events]
            p_t_m = p_mt / p_m                                                 # [sample_rate, batch_size, seq_len, num_events]
            p_gap = p_t_m - probability_threshold                              # [sample_rate, batch_size, seq_len, num_events]

            return p_gap

        # Preprocess
        tau_pred = []
        batch_size, seq_len = time_history.shape
        p_m = p_m.unsqueeze(dim = 0)                                           # [1, batch_size, seq_len, num_events]
    
        for sub_sample_rate in sample_rate_list:
            probability_threshold = torch.zeros((sub_sample_rate, batch_size, seq_len, self.num_events), device = self.device)
                                                                               # [sub_sample_rate, batch_size, seq_len, num_events]
            torch.nn.init.uniform_(probability_threshold, a = its_lower_bound, b = its_upper_bound)
                                                                               # [sub_sample_rate, batch_size, seq_len, num_events]
            tau_pred.append(median_prediction(self.max_step, self.bisect_early_stop_threshold, \
                                              bisect_target, probability_threshold))
                                                                               # [sub_sample_rate, batch_size, seq_len, num_events]

        tau_pred = torch.cat(tau_pred, dim = 0)                                # [sample_rate, batch_size, seq_len, num_events]
        if return_mean:
            tau_pred = tau_pred.mean(dim = 0)                                  # [batch_size, seq_len, num_events]

        return tau_pred


    def mean_absolute_error_e_and_f1_by_time_event(self, events_history, time_history, events_next, time_next, mask_history, mask_next, mean, var):
        sample_rate = 10000
        sample_step = 1000
        
        _, pred_time = self.mean_absolute_error(events_history, time_history, time_next, mask_next, \
                                                mean, var, return_mean = False, sample_rate = sample_rate, \
                                                mae_step = sample_step)
                                                                               # [sample_rate, batch_size, seq_len] * 2

        # Preprocess
        sample_rate_list = step_split(sample_rate, sample_step)
        probability_for_each_event = []

        time_next_pred = repeat(pred_time, '... b s -> ... b s ne', ne = self.num_events)
                                                                               # [sample_rate, batch_size, seq_len, num_events]
        time_next_pred.requires_grad = True                                    # [sample_rate, batch_size, seq_len, num_events]
        for idx, samples in enumerate(sample_rate_list):
            selected_time_next_pred = time_next_pred[idx * sample_step:idx * sample_step + samples, :, :]
            probability_integral_from_pred_to_infinite = self.model(events_history, time_history, \
                                                                    selected_time_next_pred, mean = mean, var = var)
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


    def sample_time_event(self, number_of_sampled_sequences, end_time, mean, var):
        '''
        This function will sample x sequences by the learned probability distribution following the time-event prediction procedure.
        Steps:
        1. Sample a time \(t_s\) from p^*(t) = \sum{n \in M}{p^*(m, t)} referring to existing history
        2. Judge the mark of this event by comparing \(\lambda^*(m, t_s)\).
        '''

        time_history_for_sampling = torch.zeros(number_of_sampled_sequences, 1, device = self.device)
                                                                               # [number_of_sampled_sequences, 1]
        events_history_for_sampling = torch.ones(number_of_sampled_sequences, 1, device = self.device, dtype = torch.int32) * self.num_events
                                                                               # [number_of_sampled_sequences, 1]
        tmp_sum_of_sampled_time = time_history_for_sampling.sum(dim = -1)      # [number_of_sampled_sequences]

        MAX_sampled_seq = 250
        seq_length = 1

        while seq_length < MAX_sampled_seq:
            sampled_time, sampled_events = \
                self.sample_one_event_from_model_time_event(number_of_sampled_sequences, events_history_for_sampling, time_history_for_sampling, mean, var)
                                                                               # [number_of_sampled_sequences, 1]
            # Ensure the sampled times and events are correct.
            assert sampled_time.shape == (number_of_sampled_sequences, 1)
            assert sampled_time.shape == (number_of_sampled_sequences, 1)

            tmp_events_history_for_sampling, _ = pack([events_history_for_sampling, sampled_events], 'nss *')
                                                                               # [number_of_sampled_sequences, history_length + 1]
            tmp_time_history_for_sampling, _ = pack([time_history_for_sampling, sampled_time], 'nss *')
                                                                               # [number_of_sampled_sequences, history_length + 1]
            tmp_sum_of_sampled_time = tmp_time_history_for_sampling.sum(dim = -1)
                                                                               # [number_of_sampled_sequences]
            seq_length += 1

            if tmp_sum_of_sampled_time.min() >= end_time:
                break
            else:
                events_history_for_sampling = tmp_events_history_for_sampling  # [number_of_sampled_sequences, new_length]
                time_history_for_sampling = tmp_time_history_for_sampling      # [number_of_sampled_sequences, new_length]

        sampled_mask = (time_history_for_sampling.cumsum(dim = -1) < end_time).int()
                                                                               # [number_of_sampled_sequences, sampled_sequences_length]

        return time_history_for_sampling, events_history_for_sampling, sampled_mask


    def sample_one_event_from_model_time_event(self, number_of_sampled_sequences, events_history_for_sampling, time_history_for_sampling, mean, var):
        def bisect_target_sample(taus, sample_input, integral_from_zero_to_inf):
            taus = repeat(taus, '... -> ... ne', ne = self.num_events)         # [number_of_sampled_sequences, 1, num_events]
            probability_integral_from_t_to_inf_for_sample = self.model.sample(events_history_for_sampling, time_history_for_sampling, taus, mean, var)
                                                                               # [number_of_sampled_sequences, 1, num_events]
            probability_integral_from_t_to_inf_for_sample = probability_integral_from_t_to_inf_for_sample.detach()
                                                                               # [number_of_sampled_sequences, 1, num_events]
            # P_m(t) = \int_{0}^{t}{p(t|m, \mathcal{H})}
            probability_integral = integral_from_zero_to_inf - probability_integral_from_t_to_inf_for_sample
                                                                               # [number_of_sampled_sequences, 1, num_events]
            probability_integral = reduce(probability_integral, '... ne -> ...', 'sum')
                                                                               # [number_of_sampled_sequences, 1]
            return probability_integral - sample_input


        sampled_threshold = torch.zeros((number_of_sampled_sequences, 1), device = self.device)
                                                                               # [number_of_sampled_sequences, 1]
        torch.nn.init.uniform_(sampled_threshold, a = its_lower_bound, b = its_upper_bound)
                                                                               # [number_of_sampled_sequences, 1]
        time_next_zero = torch.zeros(number_of_sampled_sequences, 1, device = self.device)
                                                                               # [number_of_sampled_sequences, 1]
        time_next_zero = repeat(time_next_zero, 'b s -> b s ne', ne = self.num_events)
                                                                               # [number_of_sampled_sequences, 1, num_events]
        integral_from_zero_to_inf = self.model.sample(events_history_for_sampling, time_history_for_sampling, time_next_zero, mean = mean, var = var)
                                                                               # [number_of_sampled_sequences, 1, num_events]
        tau_sampled = median_prediction(self.max_step, self.bisect_early_stop_threshold, \
                                        bisect_target_sample, sampled_threshold, integral_from_zero_to_inf)
                                                                               # [number_of_sampled_sequences, 1]
        repeated_tau_sampled = repeat(tau_sampled, 'b s -> b s ne', ne = self.num_events)
                                                                               # [number_of_sampled_sequences, 1, num_events]
        repeated_tau_sampled.requires_grad = True
        integral_from_sampled_time_to_inf = self.model(events_history_for_sampling, time_history_for_sampling, repeated_tau_sampled, mean = mean, var = var)
                                                                               # [number_of_sampled_sequences, 1, num_events]
        probability_for_each_event_at_pred_time = - torch.autograd.grad(
            outputs = integral_from_sampled_time_to_inf,
            inputs = repeated_tau_sampled,
            grad_outputs = torch.ones_like(integral_from_sampled_time_to_inf)
        )[0]                                                                   # [number_of_sampled_sequences, 1, num_events]
        repeated_tau_sampled.requires_grad = False

        distribution_of_marks = torch.distributions.categorical.Categorical(probability_for_each_event_at_pred_time)
        sampled_marks = distribution_of_marks.sample()                         # [number_of_sampled_sequences, 1]
        sampled_marks = sampled_marks.to(self.device)                          # [number_of_sampled_sequences, 1]

        return tau_sampled, sampled_marks


    def sample_event_time(self, number_of_sampled_sequences, end_time, mean, var):
        '''
        These two functions will sample a event sequence from the learned p^*(m, t) following the event-time prediction procedure.
        Steps:
        1. Sample the mark \(m_p\) from p^*(m) = \int_{t_l}^{+\infty}{p^*(m, \tau)d\tau}.
        2. Sample when a new \(m_p\) event would happen in the future time by \(p^*(t|m_p)\).
        '''
        time_history_for_sampling = torch.zeros((number_of_sampled_sequences, 1), device = self.device)
                                                                               # [number_of_sampled_sequences, 1]
        events_history_for_sampling = torch.ones((number_of_sampled_sequences, 1), device = self.device, dtype = torch.int32) * self.num_events
                                                                               # [number_of_sampled_sequences, 1]
        tmp_sum_of_sampled_time = time_history_for_sampling.sum(dim = -1)      # [number_of_sampled_sequences]

        MAX_sampled_seq = 250
        seq_length = 1

        while seq_length < MAX_sampled_seq:
            sampled_time, sampled_events = \
                self.sample_one_event_from_model_event_time(number_of_sampled_sequences, events_history_for_sampling, time_history_for_sampling, mean, var)
                                                                               # [number_of_sampled_sequences, 1]
            # Ensure the sampled times and events are correct.
            assert sampled_time.shape == (number_of_sampled_sequences, 1)
            assert sampled_time.shape == (number_of_sampled_sequences, 1)

            tmp_events_history_for_sampling, _ = pack([events_history_for_sampling, sampled_events], 'nss *')
                                                                               # [number_of_sampled_sequences, history_length + 1]
            tmp_time_history_for_sampling, _ = pack([time_history_for_sampling, sampled_time], 'nss *')
                                                                               # [number_of_sampled_sequences, history_length + 1]
            tmp_sum_of_sampled_time = tmp_time_history_for_sampling.sum(dim = -1)
                                                                               # [number_of_sampled_sequences]
            seq_length += 1

            if tmp_sum_of_sampled_time.min() >= end_time:
                break
            else:
                events_history_for_sampling = tmp_events_history_for_sampling  # [number_of_sampled_sequences, new_length]
                time_history_for_sampling = tmp_time_history_for_sampling      # [number_of_sampled_sequences, new_length]

        sampled_mask = (time_history_for_sampling.cumsum(dim = -1) < end_time).int()
                                                                               # [number_of_sampled_sequences, sampled_sequences_length]

        return time_history_for_sampling, events_history_for_sampling, sampled_mask
    

    def sample_one_event_from_model_event_time(self, number_of_sampled_sequences, events_history_for_sampling, \
                                               time_history_for_sampling, mean, var, mark_mask = None, output_p_m = False):
        '''
        events_history_for_sampling: [batch_size, seq_len](batch_size is number_of_sampled_sequences when sample_event_time() calls this function.)
        time_history_for_sampling: [batch_size, seq_len]
        '''
        def bisect_target_sample(taus, sample_input, integral_from_zero_to_inf):
            probability_integral_from_t_to_inf_for_sample = self.model.sample(events_history_for_sampling, time_history_for_sampling, taus, mean, var)
                                                                               # [...,batch_size, num_events]
            probability_integral_from_t_to_inf_for_sample = probability_integral_from_t_to_inf_for_sample.detach()
                                                                               # [..., number_of_sampled_sequences, batch_size, num_events]
            # P_m(t) = \int_{0}^{t}{p(t|m, \mathcal{H})}
            probability_integral = integral_from_zero_to_inf - probability_integral_from_t_to_inf_for_sample
                                                                               # [number_of_sampled_sequences, batch_size, num_events]
            probability_integral = probability_integral / integral_from_zero_to_inf
                                                                               # [number_of_sampled_sequences, batch_size, num_events]
            return probability_integral - sample_input


        batch_size, _ = events_history_for_sampling.shape
        time_next_zero = torch.zeros(number_of_sampled_sequences, batch_size, device = self.device)
                                                                               # [number_of_sampled_sequences, batch_size]
        time_next_zero = repeat(time_next_zero, 'nss b -> nss b ne', ne = self.num_events)
                                                                               # [number_of_sampled_sequences, batch_size, num_events]
        integral_from_zero_to_inf = self.model.sample(events_history_for_sampling, time_history_for_sampling, time_next_zero, mean = mean, var = var)
                                                                               # [number_of_sampled_sequences, batch_size, num_events]
        distribution_of_marks = torch.distributions.categorical.Categorical(integral_from_zero_to_inf)
        sampled_marks = distribution_of_marks.sample()                         # [number_of_sampled_sequences, batch_size]
        sampled_marks = sampled_marks.to(self.device)                          # [number_of_sampled_sequences, batch_size]

        sampled_threshold = torch.zeros((number_of_sampled_sequences, batch_size, self.num_events), device = self.device)
                                                                               # [number_of_sampled_sequences, batch_size, num_events]
        torch.nn.init.uniform_(sampled_threshold, a = its_lower_bound, b = its_upper_bound)
                                                                               # [number_of_sampled_sequences, batch_size, num_events]
        tau_sampled = median_prediction(self.max_step, self.bisect_early_stop_threshold, \
                                        bisect_target_sample, sampled_threshold, integral_from_zero_to_inf)
        
        if mark_mask is not None:
            '''
            Return all sampled timestamps of the selected marks if mark_mask is not None.
            '''
            einop = f'... -> {"() " * (len(tau_sampled.shape) - len(mark_mask.shape))} ...'
            tau_mask = rearrange(mark_mask, einop)                             # [number_of_sampled_sequences, batch_size, num_events]
                                                                               # [number_of_sampled_sequences, batch_size, num_events]
            tau_sampled = tau_sampled * tau_mask                               # [number_of_sampled_sequences, batch_size]
        else:
            tau_mask = torch.nn.functional.one_hot(sampled_marks, num_classes = self.num_events)
                                                                               # [number_of_sampled_sequences, batch_size, num_events]
            tau_sampled = (tau_sampled * tau_mask).sum(dim = -1)               # [number_of_sampled_sequences, batch_size]

        if output_p_m:
            return tau_sampled, sampled_marks, integral_from_zero_to_inf
        else:
            return tau_sampled, sampled_marks


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
        Caution: dataloader won't add the end dummy event during evaluation!

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

        return NotImplementedError('IFIB is intensity-free. Therefore, it can not provide the plot for the intensity function.')


    def integral(self, input_data, opt):
        '''
        Function prober, used by tpp_ploter to draw plots.

        Args:
        * input_data  type: list shape: [[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]
                      The original minibatch. Detailed information is available in extract_plot_data()
        * resolution  type: int shape: N/A
                      How many interpretive numbers we have between an event interval?
        '''
        return NotImplementedError('IFIB is intensity-free. Therefore, it can not provide the plot for the intensity integral.')


    def probability(self, input_data, opt):
        '''
        Function prober, used by tpp_ploter to draw plots.

        Args:
        * input_data  type: list shape: [[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]
                      The original minibatch. Detailed information is available in extract_plot_data()
        * resolution  type: int shape: N/A
                      How many interpretive numbers we have between an event interval?
        '''
        

        input_time, input_events, input_intensity, mask, mean, var = self.extract_plot_data(input_data)
        
        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # [batch_size, seq_len]
        _, mask_next = self.divide_history_and_next(mask)                      # [batch_size, seq_len]

        expand_probability, timestamp = \
            self.model.probability(events_history, time_history, time_next, opt.resolution, mean, var)
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
        

        input_time, input_events, input_intensity, mask, mean, var = self.extract_plot_data(input_data)

        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]

        mae, f1_1 = self.mean_absolute_error_and_f1(events_history, time_history, events_next, \
                                                    time_next, mask_history, mask_next, mean, var)
                                                                               # [batch_size, seq_len]
        data, timestamp = self.model.model_probe_function(events_history, time_history, time_next, opt.resolution, mean, var, mask_next)

        f1_2, top_k, probability_sum, _, tau_pred_all_event, maes_avg, maes \
            = self.mean_absolute_error_e(events_history, events_next, time_history, time_next, mask_next, mean, var, return_mean = False)

        '''
        We show how porobability distribution goes on two sampled sequences, one following the event-time routine, and the other following
        the time-event routine.
        '''
        time_history_for_sampling_event_time, events_history_for_sampling_event_time, sampled_mask_event_time \
            = self.sample_event_time(1, self.end_time - self.start_time, mean, var)
                                                                               # 3 * [number_of_sampled_sequences, length_of_sampled_sequences]

        sampled_time_history_event_time, sampled_time_next_event_time = self.divide_history_and_next(time_history_for_sampling_event_time)
                                                                               # 2 * [batch_size, seq_len]
        sampled_events_history_event_time, sampled_events_next_event_time = self.divide_history_and_next(events_history_for_sampling_event_time)
                                                                               # 2 * [batch_size, seq_len]
        _, sampled_mask_next_event_time = self.divide_history_and_next(sampled_mask_event_time)
                                                                               # 2 * [batch_size, seq_len]

        sampled_data_event_time, sampled_timestamp_event_time \
            = self.model.model_probe_function(sampled_events_history_event_time, sampled_time_history_event_time, \
                                              sampled_time_next_event_time, opt.resolution, mean, var, sampled_mask_next_event_time)


        time_history_for_sampling_time_event, events_history_for_sampling_time_event, sampled_mask_time_event \
            = self.sample_time_event(1, self.end_time - self.start_time, mean, var)
                                                                               # 3 * [number_of_sampled_sequences, length_of_sampled_sequences]

        sampled_time_history_time_event, sampled_time_next_time_event = self.divide_history_and_next(time_history_for_sampling_time_event)
                                                                               # 2 * [batch_size, seq_len]
        sampled_events_history_time_event, sampled_events_next_time_event = self.divide_history_and_next(events_history_for_sampling_time_event)
                                                                               # 2 * [batch_size, seq_len]
        _, sampled_mask_next_time_event = self.divide_history_and_next(sampled_mask_time_event)
                                                                               # 2 * [batch_size, seq_len]

        sampled_data_time_event, sampled_timestamp_time_event \
            = self.model.model_probe_function(sampled_events_history_time_event, sampled_time_history_time_event, \
                                              sampled_time_next_time_event, opt.resolution, mean, var, sampled_mask_next_time_event)

        '''
        Here, we show the relation between mark and time. Different mark should receive different time predictions.
        This part we do not sample a complete sequence.
        '''
        
        '''
        the_number_of_samples = 10000
        history_length = int(events_history.shape[1] * 0.4)
        mark_mask = torch.ones(self.num_events, device = self.device)
        events_history_for_sample = events_history[..., :history_length]       # [batch_size, seq_len]
        time_history_for_sample = time_history[..., :history_length]           # [batch_size, the_number_of_samples, seq_len]
        sampled_times, _, p_m = self.sample_one_event_from_model_event_time(the_number_of_samples, events_history_for_sample, \
                                                                       time_history_for_sample, mean, var, mark_mask = mark_mask, output_p_m = True)
        sampled_times_1, sampled_marks_1 = self.sample_one_event_from_model_time_event(the_number_of_samples, \
                                                                                       events_history_for_sample, \
                                                                                       time_history_for_sample, mean, var)

        import pickle as pkl
        import os
        f_sampled_time = open(os.path.join(opt.plot_store_dir_for_this_batch, 'sampled_time.pkl'), 'wb')
        pkl.dump({'sampled_times': move_from_tensor_to_ndarray(sampled_times), \
                  'p_m': move_from_tensor_to_ndarray(p_m)}, f_sampled_time)
        f_sampled_time.close()

        # which event will happen first.
        f_sampled_time = open(os.path.join(opt.plot_store_dir_for_this_batch, 'which_event_first.pkl'), 'wb')
        info = {'event_history': move_from_tensor_to_ndarray(events_history_for_sample),
                'time_history': move_from_tensor_to_ndarray(time_history_for_sample),
                'sampled_time': move_from_tensor_to_ndarray(sampled_times)}
        pkl.dump(info, f_sampled_time)
        f_sampled_time.close()

        # why time-event bad.
        f_event_time = open(os.path.join(opt.plot_store_dir_for_this_batch, 'time-event.pkl'), 'wb')
        info = {'event_history': move_from_tensor_to_ndarray(events_history_for_sample), 
                'time_history': move_from_tensor_to_ndarray(time_history_for_sample),
                'sampled_time': move_from_tensor_to_ndarray(sampled_times_1), 
                'sampled_mark': move_from_tensor_to_ndarray(sampled_marks_1)}
        pkl.dump(info, f_event_time)
        f_event_time.close()
        '''
    
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

        expand_probability, timestamp = \
            self.model.probability(events_history, time_history, time_next, opt.resolution, mean, var)
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
                                        timestamp = timestamp_per_seq, resolution = opt.resolution)
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

        f1_2, top_k, probability_sum, probability_integral_from_zero_to_infinite, tau_pred_all_event, maes_avg, maes \
            = self.mean_absolute_error_e(events_history, events_next, time_history, time_next, mask_next, mean, var)
        
        _, maes, probability_sum, = move_from_tensor_to_ndarray(*maes, probability_sum)

        return maes, f1_2, probability_sum, probability_integral_from_zero_to_infinite, events_next


    def get_mae_e_and_f1_by_time_event(self, input_data, opt):
        input_time, input_events, input_intensity, mask, mean, var = self.extract_plot_data(input_data)
        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]

        maes, f1_2, events_pred_index, events_next \
            = self.mean_absolute_error_e_and_f1_by_time_event(events_history, time_history, events_next, \
                                                              time_next, mask_history, mask_next, mean, var)
        
        maes, events_pred_index, events_next = move_from_tensor_to_ndarray(maes, events_pred_index, events_next)

        return maes, f1_2, events_pred_index, events_next


    def get_which_event_first(self, input_data, opt):
        '''
        Caution: This function only works when batch_size = 1.
        '''
        assert opt.evaluation_batch_size == 1
        
        '''
        Hyperparameters
        '''
        the_number_of_samples = 10000
        mark_mask = torch.ones(self.num_events, device = self.device)

        input_time, input_events, input_intensity, mask, mean, var = self.extract_plot_data(input_data)
        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]

        sampled_times = torch.zeros(the_number_of_samples, time_history.shape[-1], self.num_events, device = self.device)
                                                                               # [the_number_of_samples, seq_len, num_events]
        for end_index in range(1, time_history.shape[-1] + 1):
            sampled_times[:, end_index - 1:end_index, :], _ = \
                self.sample_one_event_from_model_event_time(the_number_of_samples, events_history[:, :end_index], \
                                                            time_history[:, :end_index], mean, var, mark_mask = mark_mask)
                                                                               # [the_number_of_samples, 1, num_events]
        
        sampled_times_mean = sampled_times.mean(dim = 0)                       # [seq_len, num_events]
        predicted_time, predicted_mark = sampled_times_mean.min(dim = -1)      # [seq_len] + [seq_len]
        predicted_time = predicted_time.unsqueeze(dim = 0)                     # [1, seq_len]
        predicted_mark = predicted_mark.unsqueeze(dim = 0)                     # [1, seq_len]

        maes = torch.abs(time_next - predicted_time)                           # [1, seq_len]
        events_next, predicted_mark = move_from_tensor_to_ndarray(events_next, predicted_mark)
        f1 = f1_score(y_true = events_next.squeeze(), y_pred = predicted_mark.squeeze(), average = 'macro')

        maes = move_from_tensor_to_ndarray(maes)

        return maes, f1
    

    def samples_from_et(self, input_data, opt):
        input_time, input_events, input_intensity, mask, mean, var = self.extract_plot_data(input_data)
        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # [batch_size, seq_len]

        the_number_of_samples = 1000
        samples = []
        p_ms = []
        mark_mask = torch.ones(self.num_events, device = self.device)
        for history_length in range(1, events_history.shape[-1]):
            events_history_for_sample = events_history[..., :history_length]   # [batch_size, seq_len]
            time_history_for_sample = time_history[..., :history_length]       # [batch_size, seq_len]
            sampled_times, _, p_m \
                = self.sample_one_event_from_model_event_time(the_number_of_samples, events_history_for_sample, \
                                                              time_history_for_sample, mean, var, mark_mask = mark_mask, output_p_m = True)
            samples.append(sampled_times)                                      # [the_number_of_samples, batch_size, seq_len]
            p_ms.append(p_m.mean(dim = 0))                                     # [batch_size, seq_len]

        samples = torch.stack(samples, dim = -2)                               # [the_number_of_samples, batch_size, seq_len, num_marks]
        p_ms = torch.stack(p_ms, dim = -2)                                     # [batch_size, seq_len, num_marks]

        return samples, p_ms


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
        [time_seq, event_seq, score, mask], (mean, var) = minibatch
        loss, time_loss_without_dummy, events_loss, the_number_of_events = model(         
                task_name = 'train', input_time = time_seq, input_events = event_seq, \
                mask = mask, mean = mean, var = var
        )
        
        loss.backward()
    
        time_loss_without_dummy = time_loss_without_dummy.item() / the_number_of_events
        events_loss = events_loss.item() / the_number_of_events
        fact = score.sum().item() / the_number_of_events
        
        return time_loss_without_dummy, fact, events_loss
    

    def evaluation_step(model, minibatch, device):
        ''' Epoch operation in evaluation phase '''
    
        model.eval()
        [time_seq, event_seq, score, mask], (mean, var) = minibatch
        time_loss_wihtout_dummy, time_loss_survival, events_loss, f1_pred_at_time_next, mae, f1_pred_at_pred_time, the_number_of_events \
        = model(
                task_name = 'evaluate', input_time = time_seq, input_events = event_seq, 
                mask = mask, mean = mean, var = var
        )
    
        time_loss_wihtout_dummy = time_loss_wihtout_dummy.item() / the_number_of_events
        time_loss_survival = time_loss_survival.item()
        events_loss = events_loss.item() / the_number_of_events
        fact = score.sum().item() / the_number_of_events
        
        return time_loss_wihtout_dummy, time_loss_survival, fact, events_loss, f1_pred_at_time_next, mae, f1_pred_at_pred_time


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
            return [input[0], input[1], input[0] - input[2], input[3], input[4], input[5], input[6]]
        
        return (train_postprocess(input) if procedure == 'Training' else test_postprocess(input))
    

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
            format_dict['f1_pred_at_time_next'] = input[4]
            format_dict['mae'] = input[5]
            format_dict['f1_pred_at_pred_time'] = input[6]
            format_dict['num_format'] = {'absolute_NLL_loss': ':6.5f', 'avg_survival_loss': ':6.5f', 
                                         'relative_NLL_loss': ':6.5f', 'events_loss': ':6.5f', 
                                         'f1_pred_at_time_next': ':2.8f', 'mae': ':2.8f', 
                                         'f1_pred_at_pred_time': ':2.8f'}
            return format_dict
        
        return (train_log_print_format(input) if procedure == 'Training' else test_log_print_format(input))

    format_dict_length = 7
    
    def choose_metric(evaluation_report_format_dict, test_report_format_dict):
        '''
        [relative loss on evaluation dataset, relative loss on test dataset, event loss on test dataset]
        '''
        return [evaluation_report_format_dict['absolute_NLL_loss'], 
                test_report_format_dict['absolute_NLL_loss']], \
               ['evaluation_absolute_loss', 'test_absolute_loss']
    
    metric_number = 2 # metric number is the length of the output of choose_metric