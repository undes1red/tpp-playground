import torch, copy
import numpy as np
from sklearn.metrics import f1_score, top_k_accuracy_score, accuracy_score
from einops import rearrange, repeat, reduce, pack
from scipy.stats import spearmanr

from src.TPP.model.tifib_c.submodel import TIFIBC
from src.TPP.model.utils import *
from src.TPP.model.tifib_c.plot import *


class TIFIBCModel(BasicModule):
    def __init__(self, d_history,
                 d_intensity,
                 dropout,
                 mlp_layers,
                 nonlinear,
                 probability_threshold,
                 info_dict,
                 device,
                 d_hidden, n_layers,
                 n_head, d_qk, d_v,
                 epsilon = 0.0, pretrain = False, alpha = 0.5, beta = 0.1):
        super(TIFIBCModel, self).__init__()
        self.device = device
        self.probability_threshold = probability_threshold
        self.num_events = info_dict['num_events']
        self.start_time = info_dict['t_0']
        self.end_time = info_dict['T']
        self.epsilon = epsilon

        self.model = TIFIBC(d_history, d_intensity, self.num_events, dropout, d_hidden, n_layers, \
                            n_head, d_qk, d_v, mlp_layers, nonlinear, epsilon, pretrain, \
                            alpha, beta, device)


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
            'mae_e_and_f1': self.get_mae_e_and_f1,
            'graph': self.plot
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
        self.train()

        time_history, time_next = self.divide_history_and_next(input_time)     # 2 * [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # 2 * [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]

        time_next = repeat(time_next, 'b s -> b s ne', ne = self.num_events)   # [batch_size, seq_len, num_events]
        time_next.requires_grad = True

        '''
        \int_{t}^{+\inf}{p(m, \tau|\mathcal{H})d\tau}
        '''
        probability_integral_from_t_to_infinite = self.model(events_history, time_history, time_next, mask_history, 
                                                             mean = mean, var = var)
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
        mask_next_without_dummy = self.remove_dummy_event_from_mask(mask_next) # [batch_size, seq_len
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
        # Survival probability: \int_{t_N}^{T}{\sum_{k}\lambda_k^(\tau)d\tau} = -\log(1 - P(t)) = -log(IFIB-C(t)).
        dummy_event_index = mask_next.sum(dim = -1) - 1                        # [batch_size]
        probability_survival = probability_integral_from_t_to_infinite.sum(dim = -1).gather(index = dummy_event_index.unsqueeze(dim = -1), dim = -1)
                                                                               # [batch_size, 1]
        time_loss_survival = -torch.log(probability_survival + self.epsilon).sum()

        loss = time_loss_without_dummy + time_loss_survival

        # we need time_loss_without_dummy to compare our distribution against the ground truth.
        return loss, time_loss_without_dummy, events_loss, the_number_of_events


    def evaluate_procedure(self, input_time, input_events, mask, mean, var):
        self.eval()

        time_history, time_next = self.divide_history_and_next(input_time)     # 2 * [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # 2 * [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]
        
        '''
        Remove the probability of the dummy event by mask.
        '''
        mask_next_without_dummy = self.remove_dummy_event_from_mask(mask_next) # [batch_size, seq_len
        events_next_without_dummy = events_next * mask_next_without_dummy      # [batch_size, seq_len]
        the_number_of_events = mask_next_without_dummy.sum().item()
        
        mae, pred_time = self.mean_absolute_error(events_history = events_history, time_history = time_history,\
                                                  time_next = time_next, mask_history = mask_history, \
                                                  mask_next = mask_next_without_dummy, mean = mean, var = var)
                                                                               # 2 * [batch_size, seq_len]
        mae = mae.sum().item() / the_number_of_events

        time_next = repeat(time_next, 'b s -> b s ne', ne = self.num_events)   # [batch_size, seq_len, num_events]
        pred_time = repeat(pred_time, 'b s -> b s ne', ne = self.num_events)   # [batch_size, seq_len, num_events]
        time_zero = torch.zeros_like(time_next, device = self.device)          # [batch_size, seq_len, num_events]


        time_next.requires_grad = True                                         # [batch_size, seq_len, num_events]
        pred_time.requires_grad = True                                         # [batch_size, seq_len, num_events]

        probability_integral_from_zero_to_infinite = self.model(events_history, time_history, time_zero, mask_history, mean = mean, var = var)
                                                                               # [batch_size, seq_len, num_events]
        probability_integral_from_pred_time_to_infinite = self.model(events_history, time_history, pred_time, mask_history, mean = mean, var = var)
                                                                               # [batch_size, seq_len, num_events]
        probability_integral_from_time_next_to_infinite = self.model(events_history, time_history, time_next, mask_history, mean = mean, var = var)
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
        Event loss. Event predictions are made with time predictions  at time_next.
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
        time_next_pred = repeat(pred_time, 'b s -> b s ne', ne = self.num_events)
                                                                               # [batch_size, seq_len, num_events]
        time_next_pred.requires_grad = True                                    # [batch_size, seq_len, num_events]

        probability_integral_from_pred_to_infinite = self.model(events_history, time_history, time_next_pred, mask_history, mean = mean, var = var)
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


    def mean_absolute_error(self, events_history, time_history, time_next, mask_history, mask_next, mean, var):
        '''
        The input should be the original minibatch
        MAE evaluation part, dwg and fullynn exclusive
        '''
        def evaluate(integral_from_zero_to_inf, taus):
            taus = repeat(taus, 'b s -> b s ne', ne = self.num_events)         # [batch_size, seq_len, num_events]
            probability_integral_from_t_to_inf = self.model(events_history, time_history, taus, mask_history, mean, var)
                                                                               # [batch_size, seq_len, num_events]
            # P_m(t) = \int_{0}^{t}{p(t|m, \mathcal{H})}
            probability_integral = integral_from_zero_to_inf - probability_integral_from_t_to_inf
                                                                               # [batch_size, seq_len, num_events]
            probability_integral = reduce(probability_integral, '... ne -> ...', 'sum')
                                                                               # [batch_size, seq_len]
            return probability_integral

        def bisect_target(integral_from_zero_to_inf, taus):
            return evaluate(integral_from_zero_to_inf, taus) - self.probability_threshold
            
        def median_prediction(integral_from_zero_to_inf, l, r):
            for _ in range(50):
                c = (l + r)/2
                v = bisect_target(integral_from_zero_to_inf, c)
                l = torch.where(v < 0, c, l)
                r = torch.where(v >= 0, c, r)

            return (l + r)/2
        
        l = 0.0001*torch.ones_like(time_history, dtype = torch.float32)        # [batch_size, seq_len]
        r = 1e6*torch.ones_like(time_history, dtype = torch.float32)           # [batch_size, seq_len]
        time_next_zero = torch.zeros_like(time_next)                           # [batch_size, seq_len]
        time_next_zero = repeat(time_next_zero, 'b s -> b s ne', ne = self.num_events)
                                                                               # [batch_size, seq_len, num_events]
        integral_from_zero_to_inf = self.model(events_history, time_history, time_next_zero, mask_history, mean = mean, var = var)
                                                                               # [batch_size, seq_len, num_events]
        tau_pred = median_prediction(integral_from_zero_to_inf, l, r)          # [batch_size, seq_len]
        gap = (tau_pred - time_next) * mask_next                               # [batch_size, seq_len]
        gap = torch.abs(gap)                                                   # [batch_size, seq_len]

        return gap, tau_pred


    def mean_absolute_error_e(self, events_history, events_next, time_history, time_next, mask_history, mask_next, mean, var):
        '''
        Well...We will do something totally different by performing event-wise MAE.
        First, predict the event types by \int_{t_i}^{+\infty}{\lambda^*_i(t)\exp(-\int_{t_0}^{\tau}{\lambda^*_i(t)dt})d\tau}
        Next, given time predictions. (Expectation? or probability bigger than 0.5?)
        '''
        time_zero = torch.zeros_like(time_next)                                # [batch_size, seq_len]
        # preparing for multi-event training when needed
        time_zero = repeat(time_zero, 'b s -> b s ne', ne = self.num_events)   # [batch_size, seq_len, num_events]

        probability_integral_from_zero_to_infinite = \
            self.model(events_history, time_history, time_zero, mask_history, mean = mean, var = var)
                                                                               # [batch_size, seq_len, num_events]

        probability_integral_sum = reduce(probability_integral_from_zero_to_infinite, 'b s ne -> b s', 'sum')
                                                                               # [batch_size, seq_len]
        predict_index = torch.argmax(probability_integral_from_zero_to_infinite, dim = -1)
                                                                               # [batch_size, seq_len]

        f1 = []
        top_k_acc = []
        for (events_next_per_seq, probability_integral_per_seq) in zip(events_next, probability_integral_from_zero_to_infinite):
            f1.append(f1_score(y_true = events_next_per_seq.detach().cpu(),
                          y_pred = torch.argmax(probability_integral_per_seq, dim = -1).detach().cpu(), average = 'macro'))
            top_k_acc_single_event_seq = []
            if self.num_events > 2:
                for k in range(1, self.num_events):
                    top_k_acc_single_event_seq.append(
                        top_k_accuracy_score(y_true = events_next_per_seq.detach().cpu(),
                                             y_score = probability_integral_per_seq.detach().cpu(),
                                             k = k,
                                             labels = np.arange(self.num_events))
                    )
            else:
                top_k_acc_single_event_seq.append(
                    accuracy_score(
                        y_true = events_next_per_seq.detach().cpu(),
                        y_pred = torch.argmax(probability_integral_per_seq, dim = -1).detach().cpu()
                    )
                )
            top_k_acc.append(top_k_acc_single_event_seq)

        predict_index_one_hot = torch.nn.functional.one_hot(predict_index.long(), num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]
        events_next_one_hot = torch.nn.functional.one_hot(events_next.long(), num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]

        # step 2: get the time prediction for that kind of event
        tau_pred_all_event = self.prediction_with_all_event_types(events_history, time_history, mask_history, \
                                                                  probability_integral_from_zero_to_infinite, \
                                                                  mean, var)
                                                                               # [batch_size, seq_len, num_events]
        mae_per_event_pure_predict = torch.abs((tau_pred_all_event * predict_index_one_hot).sum(dim = -1) - time_next) * mask_next
                                                                               # [batch_size, seq_len, num_events]
        mae_per_event = torch.abs((tau_pred_all_event * events_next_one_hot).sum(dim = -1) - time_next) * mask_next
                                                                               # [batch_size, seq_len, num_events]

        mae_per_event_pure_predict_avg = torch.sum(mae_per_event_pure_predict, dim = -1) / mask_next.sum(dim = -1)
        mae_per_event_avg = torch.sum(mae_per_event, dim = -1) / mask_next.sum(dim = -1)

        return f1, top_k_acc, probability_integral_sum, tau_pred_all_event, (mae_per_event_pure_predict_avg, mae_per_event_avg), \
               (mae_per_event_pure_predict, mae_per_event)


    def prediction_with_all_event_types(self, events_history, time_history, mask_history, p_m, mean, var):
        '''
        The input should be the original minibatch
        MAE evaluation part, dwg and fullynn exclusive

        '''
        def evaluate_all_event(taus):
            # \int_{tau}^{+\inf}{p(m, \tau|\mathcal{H})d\tau}
            probability_integral_from_t_to_infinite = self.model(events_history, time_history, taus, mask_history, mean = mean, var = var)
                                                                               # [batch_size, seq_len, num_events]
            # \int_{0}^{tau}{p(m, \tau|\mathcal{H})d\tau}
            probability_from_zero_to_t = p_m - probability_integral_from_t_to_infinite
                                                                               # [batch_size, seq_len, num_events]
            return probability_from_zero_to_t

        def bisect_target(taus):
            p_mt = evaluate_all_event(taus)                                    # [batch_size, seq_len, num_events]
            p_t_m = p_mt / p_m                                                 # [batch_size, seq_len, num_events]
            p_gap = p_t_m - self.probability_threshold                         # [batch_size, seq_len, num_events]

            return p_gap
            
        def median_prediction(l, r):
            for _ in range(50):
                c = (l + r)/2
                v = bisect_target(c)
                l = torch.where(v < 0, c, l)
                r = torch.where(v >= 0, c, r)

            return (l + r)/2
        
        l = 0.0001*torch.ones((*time_history.shape, self.num_events), dtype = torch.float32, device = self.device)
                                                                               # [batch_size, seq_len, num_events]
        r = 1e6*torch.ones((*time_history.shape, self.num_events), dtype = torch.float32, device = self.device)
                                                                               # [batch_size, seq_len, num_events]
        tau_pred = median_prediction(l, r)                                     # [batch_size, seq_len, num_events]

        return tau_pred


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

        while True:
            sampled_time, sampled_events = \
                self.sample_one_events_from_model_time_event(number_of_sampled_sequences, events_history_for_sampling, time_history_for_sampling, mean, var)
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

            if tmp_sum_of_sampled_time.min() >= end_time:
                break
            else:
                events_history_for_sampling = tmp_events_history_for_sampling  # [number_of_sampled_sequences, new_length]
                time_history_for_sampling = tmp_time_history_for_sampling      # [number_of_sampled_sequences, new_length]

        sampled_mask = (time_history_for_sampling.cumsum(dim = -1) < end_time).int()
                                                                               # [number_of_sampled_sequences, sampled_sequences_length]

        return time_history_for_sampling, events_history_for_sampling, sampled_mask


    def sample_one_events_from_model_time_event(self, number_of_sampled_sequences, events_history_for_sampling, time_history_for_sampling, mean, var):
        def evaluate_sample(integral_from_zero_to_inf, taus):
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
            return probability_integral

        def bisect_target_sample(integral_from_zero_to_inf, taus, sample_input):
            return evaluate_sample(integral_from_zero_to_inf, taus) - sample_input
            
        def median_prediction_sample(integral_from_zero_to_inf, l, r):
            '''
            First, we randomly generate the probability_threshold from a uniform distribution.
            '''
            dist = torch.distributions.uniform.Uniform(torch.tensor(0.0), torch.tensor(1.0))
            sampled_threshold = dist.sample((number_of_sampled_sequences, 1))  # [number_of_sampled_sequences, 1]
            sampled_threshold = sampled_threshold.to(self.device)              # [number_of_sampled_sequences, 1]

            for _ in range(50):
                c = (l + r)/2
                v = bisect_target_sample(integral_from_zero_to_inf, c, sampled_threshold)
                l = torch.where(v < 0, c, l)
                r = torch.where(v >= 0, c, r)

            return (l + r)/2
        
        l = 0.0001*torch.ones((number_of_sampled_sequences, 1), dtype = torch.float32, device = self.device)
                                                                               # [number_of_sampled_sequences, 1]
        r = 1e6*torch.ones((number_of_sampled_sequences, 1), dtype = torch.float32, device = self.device)
                                                                               # [number_of_sampled_sequences, 1]
        time_next_zero = torch.zeros(number_of_sampled_sequences, 1, device = self.device)
                                                                               # [number_of_sampled_sequences, 1]
        time_next_zero = repeat(time_next_zero, 'b s -> b s ne', ne = self.num_events)
                                                                               # [number_of_sampled_sequences, 1, num_events]
        integral_from_zero_to_inf = self.model.sample(events_history_for_sampling, time_history_for_sampling, time_next_zero, mean = mean, var = var)
                                                                               # [number_of_sampled_sequences, 1, num_events]
        integral_from_zero_to_inf = integral_from_zero_to_inf.detach()         # [number_of_sampled_sequences, 1, num_events]
        tau_sampled = median_prediction_sample(integral_from_zero_to_inf, l, r)# [number_of_sampled_sequences, 1]
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

        distribution_of_marks = torch.distributions.categorical.Categorical(probability_for_each_event_at_pred_time)
        sampled_marks = distribution_of_marks.sample()                         # [number_of_sampled_sequences, 1]
        sampled_marks = sampled_marks.to(self.device)                          # [number_of_sampled_sequences, 1]
        repeated_tau_sampled.requires_grad = False

        return tau_sampled, sampled_marks


    def sample_event_time(self, number_of_sampled_sequences, end_time, mean, var):
        '''
        These two functions will sample a event sequence from the learned p^*(m, t) following the event-time prediction procedure.
        Steps:
        1. Sample the mark \(m_p\) from p^*(m) = \int_{t_l}^{+\infty}{p^*(m, \tau)d\tau}.
        2. Sample when a new \(m_p\) event would happen in the future time by \(p^*(t|m_p)\).
        '''
        time_history_for_sampling = torch.zeros(number_of_sampled_sequences, 1, device = self.device)
                                                                               # [number_of_sampled_sequences, 1]
        events_history_for_sampling = torch.ones(number_of_sampled_sequences, 1, device = self.device, dtype = torch.int32) * self.num_events
                                                                               # [number_of_sampled_sequences, 1]
        tmp_sum_of_sampled_time = time_history_for_sampling.sum(dim = -1)      # [number_of_sampled_sequences]

        while True:
            sampled_time, sampled_events = \
                self.sample_one_events_from_model_event_time(number_of_sampled_sequences, events_history_for_sampling, time_history_for_sampling, mean, var)
                                                                               # [number_of_sampled_sequences, 1]
            # Ensure the sampled times and events are correct.
            assert sampled_time.shape == (number_of_sampled_sequences, 1)
            assert sampled_time.shape == (number_of_sampled_sequences, 1)

            tmp_events_history_for_sampling, tmp_events_history_for_sampling_ps = pack([events_history_for_sampling, sampled_events], 'nss *')
                                                                               # [number_of_sampled_sequences, history_length + 1]
            tmp_time_history_for_sampling, tmp_time_history_for_sampling_ps = pack([time_history_for_sampling, sampled_time], 'nss *')
                                                                               # [number_of_sampled_sequences, history_length + 1]
            tmp_sum_of_sampled_time = tmp_time_history_for_sampling.sum(dim = -1)
                                                                               # [number_of_sampled_sequences]

            if tmp_sum_of_sampled_time.min() >= end_time:
                break
            else:
                events_history_for_sampling = tmp_events_history_for_sampling  # [number_of_sampled_sequences, new_length]
                time_history_for_sampling = tmp_time_history_for_sampling      # [number_of_sampled_sequences, new_length]

        sampled_mask = (time_history_for_sampling.cumsum(dim = -1) < end_time).int()
                                                                               # [number_of_sampled_sequences, sampled_sequences_length]

        return time_history_for_sampling, events_history_for_sampling, sampled_mask


    def sample_one_events_from_model_event_time(self, number_of_sampled_sequences, events_history_for_sampling, time_history_for_sampling, mean, var):
        def evaluate_sample(integral_from_zero_to_inf, taus):
            probability_integral_from_t_to_inf_for_sample = self.model.sample(events_history_for_sampling, time_history_for_sampling, taus, mean, var)
                                                                               # [number_of_sampled_sequences, 1, num_events]
            probability_integral_from_t_to_inf_for_sample = probability_integral_from_t_to_inf_for_sample.detach()
                                                                               # [number_of_sampled_sequences, 1, num_events]
            # P_m(t) = \int_{0}^{t}{p(t|m, \mathcal{H})}
            probability_integral = integral_from_zero_to_inf - probability_integral_from_t_to_inf_for_sample
                                                                               # [number_of_sampled_sequences, 1, num_events]
            probability_integral = probability_integral / integral_from_zero_to_inf
                                                                               # [number_of_sampled_sequences, 1, num_events]
            return probability_integral

        def bisect_target_sample(integral_from_zero_to_inf, taus, sample_input):
            return evaluate_sample(integral_from_zero_to_inf, taus) - sample_input
            
        def median_prediction_sample(integral_from_zero_to_inf, l, r):
            '''
            First, we randomly generate the probability_threshold from a uniform distribution.
            '''
            dist = torch.distributions.uniform.Uniform(torch.tensor(0.0), torch.tensor(1.0))
            sampled_threshold = dist.sample((number_of_sampled_sequences, 1, self.num_events))
                                                                               # [number_of_sampled_sequences, 1, num_events]
            sampled_threshold = sampled_threshold.to(self.device)              # [number_of_sampled_sequences, 1, num_events]

            for _ in range(50):
                c = (l + r)/2
                v = bisect_target_sample(integral_from_zero_to_inf, c, sampled_threshold)
                l = torch.where(v < 0, c, l)
                r = torch.where(v >= 0, c, r)

            return (l + r)/2
        
        l = 0.0001*torch.ones((number_of_sampled_sequences, 1, self.num_events), dtype = torch.float32, device = self.device)
                                                                               # [number_of_sampled_sequences, 1, num_events]
        r = 1e6*torch.ones((number_of_sampled_sequences, 1, self.num_events), dtype = torch.float32, device = self.device)
                                                                               # [number_of_sampled_sequences, 1, num_events]
        time_next_zero = torch.zeros(number_of_sampled_sequences, 1, device = self.device)
                                                                               # [number_of_sampled_sequences, 1]
        time_next_zero = repeat(time_next_zero, 'b s -> b s ne', ne = self.num_events)
                                                                               # [number_of_sampled_sequences, 1, num_events]
        integral_from_zero_to_inf = self.model.sample(events_history_for_sampling, time_history_for_sampling, time_next_zero, mean = mean, var = var)
                                                                               # [number_of_sampled_sequences, 1, num_events]
        integral_from_zero_to_inf = integral_from_zero_to_inf.detach()         # [number_of_sampled_sequences, 1, num_events]
        distribution_of_marks = torch.distributions.categorical.Categorical(integral_from_zero_to_inf)
        sampled_marks = distribution_of_marks.sample()                         # [number_of_sampled_sequences, 1]
        sampled_marks = sampled_marks.to(self.device)                          # [number_of_sampled_sequences, 1]

        tau_sampled = median_prediction_sample(integral_from_zero_to_inf, l, r)# [number_of_sampled_sequences, 1, num_events]
        tau_mask = torch.nn.functional.one_hot(sampled_marks, num_classes = self.num_events)
                                                                               # [number_of_sampled_sequences, 1, num_events]
        tau_sampled = (tau_sampled * tau_mask).sum(dim = -1)                   # [number_of_sampled_sequences, 1]

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
        self.model.eval()

        input_time, input_events, input_intensity, mask, mean, var = self.extract_plot_data(input_data)
        
        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]

        expand_probability, timestamp = \
            self.model.probability(events_history, time_history, time_next, mask_history, opt.resolution, mean, var)
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
        data, timestamp = self.model.model_probe_function(events_history, time_history, time_next, mask_history, mask_next, \
                                                          opt.resolution, mean, var)

        f1_2, top_k, probability_sum, tau_pred_all_event, maes_avg, maes \
            = self.mean_absolute_error_e(events_history, events_next, time_history, time_next, mask_history, mask_next, mean, var)

        '''
        We show how porobability distribution goes on two sampled sequences, one following the event-time routine, and the other following
        the time-event routine.
        '''
        time_history_for_sampling_event_time, events_history_for_sampling_event_time, sampled_mask_event_time \
            = self.sample_event_time(1, 70, mean, var)
                                                                               # 3 * [number_of_sampled_sequences, length_of_sampled_sequences]

        sampled_time_history_event_time, sampled_time_next_event_time = self.divide_history_and_next(time_history_for_sampling_event_time)
                                                                               # 2 * [batch_size, seq_len]
        sampled_events_history_event_time, sampled_events_next_event_time = self.divide_history_and_next(events_history_for_sampling_event_time)
                                                                               # 2 * [batch_size, seq_len]
        sampled_mask_history_event_time, sampled_mask_next_event_time = self.divide_history_and_next(sampled_mask_event_time)
                                                                               # 2 * [batch_size, seq_len]

        sampled_data_event_time, sampled_timestamp_event_time \
            = self.model.model_probe_function(sampled_events_history_event_time, sampled_time_history_event_time, \
                                              sampled_time_next_event_time, sampled_mask_history_event_time, \
                                              sampled_mask_next_event_time, opt.resolution, mean, var)


        time_history_for_sampling_time_event, events_history_for_sampling_time_event, sampled_mask_time_event \
            = self.sample_time_event(1, 70, mean, var)
                                                                               # 3 * [number_of_sampled_sequences, length_of_sampled_sequences]

        sampled_time_history_time_event, sampled_time_next_time_event = self.divide_history_and_next(time_history_for_sampling_time_event)
                                                                               # 2 * [batch_size, seq_len]
        sampled_events_history_time_event, sampled_events_next_time_event = self.divide_history_and_next(events_history_for_sampling_time_event)
                                                                               # 2 * [batch_size, seq_len]
        sampled_mask_history_time_event, sampled_mask_next_time_event = self.divide_history_and_next(sampled_mask_time_event)
                                                                               # 2 * [batch_size, seq_len]

        sampled_data_time_event, sampled_timestamp_time_event \
            = self.model.model_probe_function(sampled_events_history_time_event, sampled_time_history_time_event, \
                                              sampled_time_next_time_event, sampled_mask_history_time_event, \
                                              sampled_mask_next_time_event, opt.resolution, mean, var)

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
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]

        expand_probability, timestamp = \
            self.model.probability(events_history, time_history, time_next, mask_history, opt.resolution, mean, var)
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

        f1_2, top_k, probability_sum, tau_pred_all_event, maes_avg, maes \
            = self.mean_absolute_error_e(events_history, events_next, time_history, \
                                         time_next, mask_history, mask_next, mean, var)
        
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
        return [evaluation_report_format_dict['absolute_NLL_loss'] + evaluation_report_format_dict['avg_survival_loss'], 
                test_report_format_dict['absolute_NLL_loss'] + test_report_format_dict['avg_survival_loss']], \
               ['evaluation_absolute_loss', 'test_f1_absolute_loss']
    
    metric_number = 2 # metric number is the length of the output of choose_metric