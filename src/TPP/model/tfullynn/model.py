import torch, copy
from sklearn.metrics import f1_score
from einops import rearrange, repeat, reduce
from scipy.stats import spearmanr

from src.TPP.model.basic_tpp_model import BasicModel, memory_ceiling, its_lower_bound, its_upper_bound
from src.TPP.model.tfullynn.submodel import TFullyNN
from src.TPP.utils import pack_one_value_to_dict
from src.TPP.model.utils import *
from src.TPP.model.tfullynn.plot import *


class TFullyNNModel(BasicModel):
    '''
    The original FullyNN model with dedicated marker prediction module.
    As the time distribution p^*(t) and p^*(m) are independent, only function time_event_prediction() works while
    event_time_prediction() is unimplemented.

    This FullyNN employs RNN modules as the history encoder. One should use TFullyNN if they want to evaluate FullyNN's performance
    against Transformer history encoders.
    '''
    def __init__(self, d_history,
                 d_intensity,
                 dropout,
                 mlp_layers,
                 info_dict,
                 device,
                 d_hidden, n_layers, \
                 n_head, d_qk, d_v,
                 sample_rate = 32, mae_step = 32, \
                 mae_e_step = 32, survival_loss_during_training = False):
        super(TFullyNNModel, self).__init__()
        self.device = device
        self.num_events = info_dict['num_events']
        self.start_time = info_dict['t_0']
        self.end_time = info_dict['T']
        self.sample_rate = sample_rate
        self.mae_step = mae_step
        self.mae_e_step = mae_e_step
        self.bisect_early_stop_threshold = 1e-5
        self.epsilon = 1e-20
        self.survival_loss_during_training = survival_loss_during_training
        self.max_step = 50

        self.model = TFullyNN(d_history, d_intensity, self.num_events, dropout,d_hidden, n_layers, \
                             n_head, d_qk, d_v, mlp_layers, device)


    def divide_history_and_next(self, input):
        '''
        Extract the history and prediction sequences from the original output

        Args:
        * input  type: torch.tensor shape: [batch_size, seq_len + 1]
                 The input tensor.
        
        Outputs:
        * input_history  type: torch.tensor shape: [batch_size, seq_len]
                         The history sequence extracted from the original input.
        * input_next     type: torch.tensor shape: [batch_size, seq_len]
                         The history sequence extracted from the original input.
        '''

        input_history, input_next = input[:, :-1].clone(), input[:, 1:].clone()
        return input_history, input_next


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
            'graph': self.plot
        }

        return task_mapper[task_name](*args, **kwargs)


    def train_procedure(self, input_time, input_events, mask, mean, var):
        '''
        The forwardpropagation function of the FullyNNModel, the wrapper of FullyNN with lots of useful
        utilities.
        
        Outputs:
        * time_loss             type: torch.tensor shape: [1]
                                The value of NLL loss: L = -log \frac{\partial \Lambda^*(m, t)}{\partial t} + \Lambda^*(m, t)
        * events_loss           type: torch.tensor shape: [1]
                                The value of the event loss: L = -log \frac{\lambda^*(m, t)}{\sum_{n \in M}{\lambda^*(n, t)}}
        * the_number_of_events  type: int shape: N/A
                                The number of legit predicted events.
        '''
        time_history, time_next = self.divide_history_and_next(input_time)     # 2 * [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # 2 * [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]

        '''
        preparing for multi-event training when needed
        '''
        time_next = repeat(time_next, 'b s -> b s ne', ne = self.num_events)   # [batch_size, seq_len, num_events]
        time_next.requires_grad = True
        integral_for_each_event = self.model(events_history, time_history, time_next, mask_history, mean = mean, var = var)
                                                                               # [batch_size, seq_len, num_events]
        '''
        Obtains intensity values.
        '''
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

        '''
        Calculate the event loss, macro-F1, and other possible metrics measuring event prediction accuracy.
        '''
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

        '''
        Calculate the NLL loss of p^*(m, t).
        L = -log \frac{\partial \Lambda^*(m, t)}{\partial t} + \Lambda^*(m, t)
        '''
        time_loss_without_dummy = self.nll_loss(intensity = intensity_for_each_event, events_next = events_next_without_dummy, \
                                                intensity_integral = integral_for_each_event, mask_next = mask_next_without_dummy)
        loss_survival = 0
        if self.survival_loss_during_training:
            # Survival probability: \int_{t_N}^{T}{\sum_{k}\lambda_k^(\tau)d\tau}
            dummy_event_index = mask_next.sum(dim = -1) - 1                    # [batch_size]
            integral_survival = integral_for_each_event.sum(dim = -1).gather(index = dummy_event_index.unsqueeze(dim = -1), dim = -1)
                                                                               # [batch_size, 1]
            loss_survival = integral_survival.sum()

        loss = time_loss_without_dummy + loss_survival

        return loss, time_loss_without_dummy, events_loss, the_number_of_events


    def evaluate_procedure(self, input_time, input_events, mask, mean, var):
        '''
        The forwardpropagation function of the FullyNNModel, the wrapper of FullyNN with lots of useful
        utilities, with evaluation enabled.

        Outputs:
        * time_loss             type: torch.tensor shape: [1]
                                The value of NLL loss: L = -log \frac{\partial \Lambda^*(m, t)}{\partial t} + \Lambda^*(m, t)
        * events_loss           type: torch.tensor shape: [1]
                                The value of the event loss: L = -log \frac{\lambda^*(m, t)}{\sum_{n \in M}{\lambda^*(n, t)}}
        * the_number_of_events  type: int shape: N/A
                                The number of legit predicted events.
        '''
        time_history, time_next = self.divide_history_and_next(input_time)     # 2 * [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # 2 * [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]

        mask_next_without_dummy = self.remove_dummy_event_from_mask(mask_next) # [batch_size, seq_len]
        events_next_without_dummy = (events_next * mask_next_without_dummy).long()
                                                                               # [batch_size, seq_len]
        the_number_of_events = mask_next_without_dummy.sum().item()

        mae, pred_time = self.mean_absolute_error(events_history = events_history, time_history = time_history,\
                                                  time_next = time_next, mask_history = mask_history, \
                                                  mask_next = mask_next_without_dummy, mean = mean, var = var)
                                                                               # 2 * [batch_size, seq_len]
        mae = mae.sum().item() / the_number_of_events

        pred_time = repeat(pred_time, 'b s -> b s ne', ne = self.num_events)   # [batch_size, seq_len, num_events]
        time_next = repeat(time_next, 'b s -> b s ne', ne = self.num_events)   # [batch_size, seq_len, num_events]

        '''
        preparing for multi-event training when needed

        Caution: We calculate the absolute and relative time loss at event_next, not pred_time.
        '''
        pred_time.requires_grad = True
        time_next.requires_grad = True
        integral_for_each_event_from_tl_to_pred_time = self.model(events_history, time_history, pred_time, mask_history, mean = mean, var = var)
                                                                               # [batch_size, seq_len, num_events]
        integral_for_each_event_from_tl_to_time_next = self.model(events_history, time_history, time_next, mask_history, mean = mean, var = var)
                                                                               # [batch_size, seq_len, num_events]

        '''
        Obtains intensity values.
        '''
        intensity_for_each_event_from_tl_to_pred_time = torch.autograd.grad(
            outputs = integral_for_each_event_from_tl_to_pred_time,
            inputs = pred_time,
            grad_outputs = torch.ones_like(integral_for_each_event_from_tl_to_pred_time),
        )[0]                                                                   # [batch_size, seq_len, num_events]
        intensity_for_each_event_from_tl_to_time_next = torch.autograd.grad(
            outputs = integral_for_each_event_from_tl_to_time_next,
            inputs = time_next,
            grad_outputs = torch.ones_like(integral_for_each_event_from_tl_to_time_next),
        )[0]                                                                   # [batch_size, seq_len, num_events]
        pred_time.requires_grad = False
        time_next.requires_grad = False
        check_tensor(intensity_for_each_event_from_tl_to_pred_time)            # [batch_size, seq_len, num_events]
        check_tensor(intensity_for_each_event_from_tl_to_time_next)            # [batch_size, seq_len, num_events]
        assert intensity_for_each_event_from_tl_to_pred_time.shape == integral_for_each_event_from_tl_to_pred_time.shape
        assert intensity_for_each_event_from_tl_to_time_next.shape == integral_for_each_event_from_tl_to_time_next.shape

        '''
        Calculate the event loss, macro-F1, and other possible metrics measuring event prediction accuracy.
        '''
        events_loss = torch.tensor(0., dtype = torch.float32)
        f1 = 0
        probability_for_each_event = torch.log(intensity_for_each_event_from_tl_to_pred_time + self.epsilon)
                                                                               # [batch_size, seq_len, num_events]
        events_probability = torch.nn.functional.softmax(probability_for_each_event, dim = -1)
                                                                               # [batch_size, seq_len, num_events]
        events_loss = torch.nn.functional.cross_entropy(rearrange(events_probability, 'b s ne -> b ne s'), \
                                                                  events_next_without_dummy, reduction = 'none')
                                                                               # [batch_size, seq_len]
        events_loss = events_loss * mask_next_without_dummy                    # [batch_size, seq_len]
        events_loss = events_loss.sum()

        events_pred_index = torch.argmax(events_probability, dim = -1)[mask_next_without_dummy == 1]
        events_true = events_next[mask_next_without_dummy == 1]
        events_pred_index, events_true = move_from_tensor_to_ndarray(events_pred_index, events_true)
        f1 = f1_score(y_true = events_true, y_pred = events_pred_index, average = 'macro')

        '''
        Calculate the NLL loss of p^*(m, t).
        L = -log \frac{\partial \Lambda^*(m, t)}{\partial t} + \Lambda^*(m, t)
        '''
        time_loss = self.nll_loss(intensity = intensity_for_each_event_from_tl_to_time_next, events_next = events_next_without_dummy, \
                                  intensity_integral = integral_for_each_event_from_tl_to_time_next, mask_next = mask_next_without_dummy)
        # Survival probability: \int_{t_N}^{T}{\sum_{k}\lambda_k^(\tau)d\tau}
        dummy_event_index = mask_next.sum(dim = -1) - 1                        # [batch_size]
        integral_survival = integral_for_each_event_from_tl_to_time_next.sum(dim = -1).gather(index = dummy_event_index.unsqueeze(dim = -1), dim = -1)
                                                                               # [batch_size, 1]
        loss_survival = integral_survival.mean()

        return time_loss, loss_survival, events_loss, mae, f1, the_number_of_events


    def nll_loss(self, intensity, intensity_integral, events_next, mask_next):
        '''
        This function calculates the NLL loss at each legit event in events_next.
    
        Args:
        * intensity           type: torch.tensor shape: [batch_size, seq_len, num_events]
                              intensity values at $ t_i $
        * intensity_integral  type: torch.tensor shape: [batch_size, seq_len, num_events]
                              intensity integral from $ t_{i - 1} $ to $ t_{i} $
        * events_next:        type: torch.tensor shape: [batch_size, seq_len]
                              The mark of the events that we need to predict.
        * mask_next:          type: torch.tensor shape: [batch_size, seq_len]
                              Needed mask to mask out unneeded loss values.
        
        Outputs:
        * loss                type: torch.tensor shape: [1]
                              the sum of NLL loss on all event.
        '''
        intensity_mask = torch.nn.functional.one_hot(events_next, num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]
        log_intensity = torch.log(intensity + self.epsilon) * intensity_mask
        log_intensity = reduce(log_intensity, '... ne -> ...', 'sum')          # [batch_size, seq_len]
        intensity_integral = reduce(intensity_integral, '... ne -> ...', 'sum')
                                                                               # [batch_size, seq_len]
        nll_p = -log_intensity + intensity_integral                            # [batch_size, seq_len]
        loss = nll_p * mask_next
        loss = torch.sum(loss)

        return loss


    def mean_absolute_error_and_f1(self, events_history, events_next, time_history, time_next, mask_history, mask_next, mean, var):
        '''

        '''
        mae, pred_time = self.mean_absolute_error(events_history = events_history, time_history = time_history,\
                                                  time_next = time_next, mask_history = mask_history, \
                                                  mask_next = mask_next, mean = mean, var = var)
                                                                               # 2 * [batch_size, seq_len]

        pred_time = repeat(pred_time, 'b s -> b s ne', ne = self.num_events)   # [batch_size, seq_len, num_events]
        '''
        preparing for multi-event training when needed
        '''
        pred_time.requires_grad = True
        integral_for_each_event = self.model(events_history, time_history, pred_time, mask_history = mask_history, mean = mean, var = var)
                                                                               # [batch_size, seq_len, num_events]
        '''
        Obtains intensity values.
        '''
        intensity_for_each_event = torch.autograd.grad(
            outputs = integral_for_each_event,
            inputs = pred_time,
            grad_outputs = torch.ones_like(integral_for_each_event),
        )[0]
        check_tensor(intensity_for_each_event)                                 # [batch_size, seq_len, num_events]
        assert intensity_for_each_event.shape == integral_for_each_event.shape
        pred_time.requires_grad = False

        '''
        Calculate the event loss, macro-F1, and other possible metrics measuring event prediction accuracy.
        '''
        probability_for_each_event = torch.log(intensity_for_each_event + self.epsilon)
                                                                               # [batch_size, seq_len, num_events]
        events_probability = torch.nn.functional.softmax(probability_for_each_event, dim = -1)
                                                                               # [batch_size, seq_len, num_events]
        events_pred_index = torch.argmax(events_probability, dim = -1)[mask_next == 1]
        events_true = events_next[mask_next == 1]
        events_pred_index, events_true = move_from_tensor_to_ndarray(events_pred_index, events_true)
        f1 = f1_score(y_true = events_true, y_pred = events_pred_index, average = 'macro')

        return mae, f1


    def mean_absolute_error(self, events_history, time_history, time_next, mask_history, mask_next, mean, var):
        '''
        MAE evaluation module.

        Args:
        * events_history  type: torch.tensor shape: [batch_size, seq_len]
                          Historical event sequences. Commonly, this sequence is a slice of 
                          the original event sequence from 0 to seq_len - 1(included). 
        * time_history    type: torch.tensor shape: [batch_size, seq_len]
                          Historical time sequences. Similar to events_history, we always generate
                          this sequence as a slice of the original time sequence from 0 to seq_len - 1(included).
        * time_next       type: torch.tensor shape: [batch_size, seq_len, num_events]
                          When the next event actually happens. 
        * mask_next       type: torch.tensor shape: [batch_size, seq_len]
                          Needed mask to mask out unneeded loss values.
        * mean            type: float shape: N/A
                          The mean of all $ t_i - t_{i - 1} $ in the entire dataset. Dataloader is responsible to provide
                          this value if needed.
        * var             type: float shape: N/A
                          The mean of all $ t_i - t_{i - 1} $ in the entire dataset. Dataloader is responsible to provide
                          this value if needed.
        Outputs:
        * mae             type: torch.tensor shape: [batch_size, seq_len]
                          MAE(Mean Absolute Error) between predicted time and ground truth.
        * tau_pred        type: torch.tensor shape: [batch_size, seq_len]
                          Time predicted by the sum of all intensity functions $ \lambda^*(m, t) $ over $ m $.
        '''
        # Preprocess
        sample_rate_list = step_split(self.sample_rate, self.mae_step)

        def bisect_target(taus, probability_threshold):
            taus = repeat(taus, '... -> ... ne', ne = self.num_events)         # [sample_rate, batch_size, seq_len, num_events]
            integral = self.model(events_history, time_history, taus, mask_history, mean, var)
                                                                               # [sample_rate, batch_size, seq_len, num_events]
            integral = integral.sum(dim = -1)                                  # [sample_rate, batch_size, seq_len]
            
            return integral + torch.log(1 - probability_threshold)
        
        tau_pred = []
        for sub_sample_rate in sample_rate_list:
            probability_threshold = torch.zeros((sub_sample_rate, *time_next.shape), device = self.device)
                                                                               # [sample_rate, batch_size, seq_len]
            torch.nn.init.uniform_(probability_threshold)                      # [sample_rate, batch_size, seq_len]
            tau_pred.append(median_prediction(self.max_step, self.bisect_early_stop_threshold, \
                                              bisect_target, probability_threshold))
                                                                               # [sample_rate, batch_size, seq_len]
        tau_pred = torch.cat(tau_pred, dim = 0)                                # [sample_rate, batch_size, seq_len]
        tau_pred = tau_pred.mean(dim = 0)                                      # [batch_size, seq_len]
        mae = torch.abs(tau_pred - time_next) * mask_next                      # [batch_size, seq_len]

        return mae, tau_pred


    def mean_absolute_error_e(self, events_history, events_next, time_history, time_next, mask_history, mask_next, mean, var):
        '''
        MAE-E evaluation module.

        Args:
        * events_history  type: torch.tensor shape: [batch_size, seq_len]
                          Historical event sequences. Commonly, this sequence is a slice of 
                          the original event sequence from 0 to seq_len - 1(included).
        * events_next     type: torch.tensor shape: [batch_size, seq_len]
                          The mark of the events that we need to predict.
        * time_history    type: torch.tensor shape: [batch_size, seq_len]
                          Historical time sequences. Similar to events_history, we always generate
                          this sequence as a slice of the original time sequence from 0 to seq_len - 1(included).
        * time_next       type: torch.tensor shape: [batch_size, seq_len, num_events]
                          When the next event actually happens. 
        * mask_next       type: torch.tensor shape: [batch_size, seq_len]
                          Needed mask to mask out unneeded loss values.
        * mean            type: float shape: N/A
                          The mean of all $ t_i - t_{i - 1} $ in the entire dataset. Dataloader is responsible to provide
                          this value if needed.
        * var             type: float shape: N/A
                          The mean of all $ t_i - t_{i - 1} $ in the entire dataset. Dataloader is responsible to provide
                          this value if needed.
        Outputs:
        * mae             type: torch.tensor shape: [batch_size, seq_len]
                          MAE(Mean Absolute Error) between predicted time and ground truth.
        * tau_pred        type: torch.tensor shape: [batch_size, seq_len]
                          Time predicted by the sum of all intensity functions $ \lambda^*(m, t) $ over $ m $.
        '''

        '''
        set a relatively large number as the infinity and decide resolution based on this large value and
        the memory_ceiling.
        '''
        inf_val, resolution_inf, resolution_between_events = decide_resolution_inf_and_resolution_between_events(events_next, memory_ceiling, self.num_events, mean, var)
        time_next_inf = torch.ones_like(time_history, device = self.device) * inf_val
                                                                               # [batch_size, seq_len]
        '''
        Debug: manually assign resolution here to investigate how the number of samples affects the sum of P^*(m) and MAE-E
        '''
        # resolution_inf = 2500

        '''
        Step 1: obtain p^*(m) = \int_{t_l}^{+infty}{p(m, t)\dt}
        '''
        expand_integral_to_inf, expand_intensity_to_inf, time_interval \
                = self.model.integral_intensity_time_next_2d(events_history, time_history, time_next_inf, mask_history, resolution_inf, mean, var)
                                                                               # [batch_size, seq_len, resolution, num_events]
        '''
        Step 2: provide event predictions
        '''
        expand_probability_per_event = expand_intensity_to_inf * torch.exp(-expand_integral_to_inf.sum(dim = -1, keepdim = True))
                                                                               # [batch_size, seq_len, resolution, num_events]
        p_m = approximate_integration(expand_probability_per_event, time_interval, dim = -2, only_integral = True)
                                                                               # [batch_size, seq_len, num_events]
        probability_integral_sum = reduce(p_m, 'b s ne -> b s', 'sum')         # [batch_size, seq_len]
        predict_index = torch.argmax(p_m, dim = -1)                            # [batch_size, seq_len]

        '''
        Step 3: calculate macro-F1 and top-K accuracy
        '''
        f1, top_k_acc = get_f1_and_top_k_acc_in_mae_e(events_next, p_m, mask_next, self.num_events)

        predict_index_one_hot_mask = torch.nn.functional.one_hot(predict_index.long(), num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]
        events_next_one_hot_mask = torch.nn.functional.one_hot(events_next.long(), num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]
        '''
        Step 4: get the time prediction for all, predicted, and real events.
        '''
        tau_pred_all_event = self.prediction_with_all_event_types(events_history, time_history, p_m, \
                                                                  resolution_between_events, mask_history, \
                                                                  mean, var, inf_val)
                                                                               # [batch_size, seq_len, num_events]
        mae_per_event_with_predict_index = torch.abs(((tau_pred_all_event * predict_index_one_hot_mask).sum(dim = -1)) - time_next) * mask_next
                                                                               # [batch_size, seq_len]
        mae_per_event_with_event_next = torch.abs(((tau_pred_all_event * events_next_one_hot_mask).sum(dim = -1)) - time_next) * mask_next
                                                                               # [batch_size, seq_len]

        mae_per_event_with_predict_index_avg = torch.sum(mae_per_event_with_predict_index, dim = -1) / mask_next.sum(dim = -1)
        mae_per_event_with_event_next_avg = torch.sum(mae_per_event_with_event_next, dim = -1) / mask_next.sum(dim = -1)

        return f1, top_k_acc, probability_integral_sum, tau_pred_all_event, \
               (mae_per_event_with_predict_index_avg, mae_per_event_with_event_next_avg), \
               (mae_per_event_with_predict_index, mae_per_event_with_event_next)


    def prediction_with_all_event_types(self, events_history, time_history, p_m, resolution, mask_history, mean, var, inf_val):
        '''
        The time prediction of every marker whose probability is not 0.

        Args:
        * events_history  type: torch.tensor shape: [batch_size, seq_len]
                          Historical event sequences. Commonly, this sequence is a slice of 
                          the original event sequence from 0 to seq_len - 1(included). 
        * time_history    type: torch.tensor shape: [batch_size, seq_len]
                          Historical time sequences. Similar to events_history, we always generate
                          this sequence as a slice of the original time sequence from 0 to seq_len - 1(included).
        * p_m             type: torch.tensor shape: [batch_size, seq_len]
                          the value of p(m) with given markers.
        * resolution      type: int shape: N/A
                          How many values do we need in each time interval [t_{i}, t_{i + 1}].
        * mask_next       type: torch.tensor shape: [batch_size, seq_len]
                          Needed mask to mask out unneeded loss values.
        * mean            type: float shape: N/A
                          The mean of all $ t_i - t_{i - 1} $ in the entire dataset. Dataloader is responsible to provide
                          this value if needed.
        * var             type: float shape: N/A
                          The mean of all $ t_i - t_{i - 1} $ in the entire dataset. Dataloader is responsible to provide
                          this value if needed.
        * max_val         type: float shape: N/A
                          The upper bound used in the bisect method.
        Outputs:
        * tau_pred        type: torch.tensor shape: [batch_size, seq_len]
                          Time predicted by the sum of all intensity functions $ \lambda^*(m, t) $ over $ m $.
        '''
        # Preprocess
        sample_rate_list = step_split(self.sample_rate, self.mae_e_step)

        def evaluate_all_event(taus):
            '''
            placeholder
            '''
            # Train k FullyNN models for k different event types.
            integral_all_events, intensity_all_events, time_interval \
                    = self.model.integral_intensity_time_next_3d(events_history, time_history, taus, mask_history, resolution, mean, var)
                                                                               # 2 * [sample_rate, batch_size, seq_len, resolution, num_events, num_events] + [sample_rate, batch_size, seq_len, resolution, num_events]
            event_mask = torch.diag(torch.ones(self.num_events, device = self.device))
                                                                               # [num_events, num_events]
            event_mask = rearrange(event_mask, f'ne ne1 -> {"() " * (len(intensity_all_events.shape) - 2)}ne ne1')
                                                                               # [sample_rate, batch_size, seq_len, resolution, num_events, num_events]
            intensity_all_events = reduce(intensity_all_events * event_mask, '... ne -> ...', 'sum')
                                                                               # [sample_rate, batch_size, seq_len, resolution, num_events]
            integral_all_events = reduce(integral_all_events, '... ne -> ...', 'sum')
                                                                               # [sample_rate, batch_size, seq_len, resolution, num_events]
            
            p_dist = intensity_all_events * torch.exp(-integral_all_events)    # [sample_rate, batch_size, seq_len, resolution, num_events]
            
            probability = approximate_integration(p_dist, time_interval, dim = -2, only_integral = True)
                                                                               # [sample_rate, batch_size, seq_len, num_events]
            return probability

        def bisect_target(taus, probability_threshold):
            p_mt = evaluate_all_event(taus)                                    # [sample_rate, batch_size, seq_len, num_events]
            p_t_m = p_mt / p_m                                                 # [sample_rate, batch_size, seq_len, num_events]
            p_gap = p_t_m - probability_threshold                              # [sample_rate, batch_size, seq_len, num_events]

            return p_gap

        tau_pred = []
        batch_size, seq_len = time_history.shape
        p_m = p_m.unsqueeze(dim = 0)                                           # [1, batch_size, seq_len, num_events]
        for sub_sample_rate in sample_rate_list:
            probability_threshold = torch.zeros((sub_sample_rate, batch_size, seq_len, self.num_events), device = self.device)
                                                                               # [sample_rate, batch_size, seq_len, num_events]
            torch.nn.init.uniform_(probability_threshold, a = its_lower_bound, b = its_upper_bound)
                                                                               # [sample_rate, batch_size, seq_len, num_events]
            tau_pred.append(median_prediction(self.max_step, self.bisect_early_stop_threshold, \
                                              bisect_target, probability_threshold, r_val = inf_val))
                                                                               # [sample_rate, batch_size, seq_len, num_events]
        tau_pred = torch.cat(tau_pred, dim = 0)                                # [sample_rate, batch_size, seq_len, num_events]
        tau_pred = tau_pred.mean(dim = 0)                                      # [batch_size, seq_len, num_events]

        return tau_pred


    '''
    Plot utilities
    '''
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
        input_time, input_events, input_intensity, mask, mean, var = self.extract_plot_data(input_data)
        
        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]

        expand_integral, expand_intensity, timestamp = \
            self.model.integral_intensity_time_next_2d(events_history, time_history, time_next, mask_history, opt.resolution, mean, var)
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
        input_time, input_events, input_intensity, mask, mean, var = self.extract_plot_data(input_data)
        
        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]

        expand_integral, expand_intensity, timestamp = \
            self.model.integral_intensity_time_next_2d(events_history, time_history, time_next, mask_history, opt.resolution, mean, var)
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
        input_time, input_events, input_intensity, mask, mean, var = self.extract_plot_data(input_data)
        
        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]

        expand_integral, expand_intensity, timestamp = \
            self.model.integral_intensity_time_next_2d(events_history, time_history, time_next, mask_history, opt.resolution, mean, var)
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
        input_time, input_events, input_intensity, mask, mean, var = self.extract_plot_data(input_data)

        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]

        mae, f1_1 = self.mean_absolute_error_and_f1(events_history, events_next, time_history, \
                                                    time_next, mask_history, mask_next, mean, var)
                                                                               # [batch_size, seq_len]
        
        f1_2, top_k, probability_sum, tau_pred_all_event, maes_avg, maes \
            = self.mean_absolute_error_e(events_history, events_next, time_history, time_next, mask_history, mask_next, mean, var)

        data, timestamp = self.model.model_probe_function(events_history, time_history, time_next, mask_history, mask_next, \
                                                          opt.resolution, mean, var)

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
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]

        expand_integral, expand_intensity, timestamp = \
            self.model.integral_intensity_time_next_2d(events_history, time_history, time_next, mask_history, opt.resolution, mean, var)
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

        mae, f1_1 = self.mean_absolute_error_and_f1(events_history, events_next, time_history, \
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
            = self.mean_absolute_error_e(events_history, events_next, time_history, time_next, mask_history, mask_next, mean, var)
        
        _, maes, probability_sum, = move_from_tensor_to_ndarray(*maes, probability_sum)

        return maes, f1_2, probability_sum


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
        time_loss, loss_survival, events_loss, mae, f1, the_number_of_events = model(
                task_name = 'evaluate', input_time = time_seq, input_events = event_seq, \
                mask = mask, mean = mean, var = var
        )
    
        time_loss = time_loss.item() / the_number_of_events
        loss_survival = loss_survival.item()
        events_loss = events_loss.item() / the_number_of_events
        fact = score.sum().item() / the_number_of_events
        
        return time_loss, loss_survival, fact, events_loss, mae, f1


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
            format_dict['f1_pred_at_pred_time'] = pack_one_value_to_dict(input[5], '2.8f')
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