import torch, copy
import numpy as np
from sklearn.metrics import f1_score, top_k_accuracy_score, accuracy_score
from einops import rearrange, repeat, reduce, pack
from scipy.stats import spearmanr

from src.TPP.model.llmtpp.submodel import LLMTPP
from src.TPP.model.basic_tpp_model import BasicModel, its_lower_bound, its_upper_bound
from src.TPP.model.utils import *
from src.TPP.model.llmtpp.plot import *


class LLMTPPModel(BasicModel):
    def __init__(self, info_dict, llm_class_name, full_llm_name, d_model, \
                 d_embedding, lm_layers, device, dropout, d_lm_embedding = 768, epsilon = 1e-20, lambda_t = 0.1, lambda_e = 1.0):
        super(LLMTPPModel, self).__init__()
        self.device = device
        self.num_events = info_dict['num_events']
        self.start_time = info_dict['t_0']
        self.end_time = info_dict['T']
        self.epsilon = epsilon
        self.lambda_t = lambda_t
        self.lambda_e = lambda_e

        self.model = LLMTPP(llm_class_name = llm_class_name, full_llm_name = full_llm_name, \
                            d_model = d_model, d_embedding = d_embedding, num_events = self.num_events, \
                            lm_layers = lm_layers, d_lm_embedding = d_lm_embedding, dropout = dropout, device = device)


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
            'graph': self.plot,
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


    def train_procedure(self, input_time, input_events, input_score, input_mask, mean, var):
        time_history, time_next = self.divide_history_and_next(input_time)     # 2 * [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # 2 * [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(input_mask)     # [batch_size, seq_len]
    
        pred_time, pred_event_prob = self.model('train', events_history = events_history, \
                                                time_history = time_history, mask_history = mask_history, \
                                                mean = mean, var = var)
                                                                               # [batch_size, seq_len] + [batch_size, seq_len, num_events]
        '''
        Remove the probability of the dummy event by mask.
        '''
        mask_next_without_dummy = self.remove_dummy_event_from_mask(mask_next) # [batch_size, seq_len]
        events_next_without_dummy = events_next * mask_next_without_dummy      # [batch_size, seq_len]
        the_number_of_events = mask_next_without_dummy.sum().item()

        '''
        cross entropy loss between p_{real} and p_{pred}.
        '''
        events_loss_without_dummy = torch.nn.functional.cross_entropy(rearrange(pred_event_prob, 'b s ne -> b ne s'), \
                                                                                events_next_without_dummy.long(), reduction = 'none')
                                                                               # [batch_size, seq_len]
        events_loss_without_dummy = events_loss_without_dummy * mask_next_without_dummy
                                                                               # [batch_size, seq_len]
        events_loss_without_dummy = events_loss_without_dummy.sum()

        # Time loss: -log p(t) = \sum_{i = 1}^{N}{\lambda_{k}(t_i)} + \int_{t_0}^{t_N}{\sum_{k}\lambda_k^(\tau)d\tau}
        time_loss_without_dummy = self.loss(pred_time = pred_time, time_next = time_next, mask_next = mask_next_without_dummy)
        loss = self.lambda_t * time_loss_without_dummy + self.lambda_e * events_loss_without_dummy

        # we need time_loss_without_dummy to compare our distribution against the ground truth.
        return loss, time_loss_without_dummy, events_loss_without_dummy, the_number_of_events


    def evaluate_procedure(self, input_time, input_events, input_score, input_mask, mean, var):
        time_history, time_next = self.divide_history_and_next(input_time)     # 2 * [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # 2 * [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(input_mask)     # [batch_size, seq_len]
        
        '''
        Remove the probability of the dummy event by mask.
        '''
        mask_next_without_dummy = self.remove_dummy_event_from_mask(mask_next) # [batch_size, seq_len
        events_next_without_dummy = events_next * mask_next_without_dummy      # [batch_size, seq_len]
        the_number_of_events = mask_next_without_dummy.sum().item()
        
        mae, pred_time, pred_event_prob \
            = self.mean_absolute_error(events_history = events_history, time_history = time_history, \
                                       time_next = time_next, mask_history = mask_history, \
                                       mask_next = mask_next_without_dummy, mean = mean, var = var)
                                                                               # 3 * [batch_size, seq_len]
        mae = mae.sum().item() / the_number_of_events
        time_loss_without_dummy = self.loss(pred_time = pred_time, time_next = time_next, mask_next = mask_next_without_dummy)

        events_loss_without_dummy = torch.nn.functional.cross_entropy(rearrange(pred_event_prob, 'b s ne -> b ne s'), \
                                                                                events_next_without_dummy.long(), reduction = 'none')
                                                                               # [batch_size, seq_len]
        events_loss_without_dummy = events_loss_without_dummy * mask_next_without_dummy
                                                                               # [batch_size, seq_len]
        events_loss_without_dummy = events_loss_without_dummy.sum()

        '''
        macro-F1 value for reference.
        '''
        pred_event = torch.argmax(pred_event_prob, dim = -1)                   # [batch_size, seq_len]
        events_pred_index_at_pred_time = pred_event[mask_next_without_dummy == 1]
        events_true = events_next[mask_next_without_dummy == 1]
        events_true, events_pred_index_at_pred_time = move_from_tensor_to_ndarray(events_true, events_pred_index_at_pred_time)
        f1_pred = f1_score(y_true = events_true, y_pred = events_pred_index_at_pred_time, average = 'macro')

        return time_loss_without_dummy + events_loss_without_dummy, time_loss_without_dummy, events_loss_without_dummy, f1_pred, mae, the_number_of_events


    def loss(self, pred_time, time_next, mask_next):
        '''
        The definition of loss.
    
        Args:
            probability:        [batch_size, seq_len, num_events]
            events_next:        [batch_size, seq_len]
            mask_next:          [batch_size, seq_len]
        '''
        gap = torch.abs(pred_time - time_next)                                 # [batch_size, seq_len]
        masked_gap = gap * mask_next                                           # [batch_size, seq_len]
        loss = torch.pow(masked_gap, 1)                                        # [batch_size, seq_len]
        loss = torch.sum(loss)

        return loss


    def mean_absolute_error_and_f1(self, events_history, time_history, events_next, time_next, mask_history, mask_next, mean, var):
        mae, pred_time = self.mean_absolute_error(events_history, time_history, time_next, mask_next, mean, var)
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


    def mean_absolute_error(self, events_history, time_history, time_next, mask_history, mask_next, mean, var):
        '''
        The input should be the original minibatch
        '''
        pred_time, pred_event_prob = self.model('evaluate', events_history = events_history, \
                                                time_history = time_history, mask_history = mask_history, \
                                                mean = mean, var = var)
                                                                               # [batch_size, seq_len] + [batch_size, seq_len, num_events]
        mae = torch.abs(pred_time - time_next) * mask_next                     # [batch_size, seq_len]

        return mae, pred_time, pred_event_prob


    def mean_absolute_error_e(self, events_history, events_next, time_history, time_next, mask_next, mean, var):
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
        tau_pred_all_event = self.prediction_with_all_event_types(events_history, time_history, \
                                                                  probability_integral_from_zero_to_infinite, mean, var)
                                                                               # [batch_size, seq_len, num_events]
        mae_per_event_pure_predict = torch.abs((tau_pred_all_event * predict_index_one_hot).sum(dim = -1) - time_next) * mask_next
                                                                               # [batch_size, seq_len, num_events]
        mae_per_event = torch.abs((tau_pred_all_event * events_next_one_hot).sum(dim = -1) - time_next) * mask_next
                                                                               # [batch_size, seq_len, num_events]

        mae_per_event_pure_predict_avg = torch.sum(mae_per_event_pure_predict, dim = -1) / mask_next.sum(dim = -1)
        mae_per_event_avg = torch.sum(mae_per_event, dim = -1) / mask_next.sum(dim = -1)

        return f1, top_k_acc, probability_integral_sum, tau_pred_all_event, (mae_per_event_pure_predict_avg, mae_per_event_avg), \
               (mae_per_event_pure_predict, mae_per_event)


    def prediction_with_all_event_types(self, events_history, time_history, p_m, mean, var):
        '''
        The input should be the original minibatch
        MAE evaluation part, dwg and fullynn exclusive
        '''

        # Preprocess
        batch_size, seq_len = time_history.shape
        dist = torch.distributions.uniform.Uniform(torch.tensor(its_lower_bound), torch.tensor(its_upper_bound))
        probability_threshold = dist.sample((self.sample_rate, batch_size, seq_len, self.num_events))
                                                                               # [sample_rate, batch_size, seq_len, num_events]
        probability_threshold = probability_threshold.to(self.device)
        p_m = p_m.unsqueeze(dim = 0)                                           # [1, batch_size, seq_len, num_events]

        def evaluate_all_event(taus):
            # \int_{tau}^{+\inf}{p(m, \tau|\mathcal{H})d\tau}
            probability_integral_from_t_to_infinite = self.model(events_history, time_history, taus, mean = mean, var = var)
                                                                               # [sample_rate, batch_size, seq_len, num_events]
            # \int_{0}^{tau}{p(m, \tau|\mathcal{H})d\tau}
            probability_from_zero_to_t = p_m - probability_integral_from_t_to_infinite
                                                                               # [sample_rate, batch_size, seq_len, num_events]
            return probability_from_zero_to_t

        def bisect_target(taus):
            p_mt = evaluate_all_event(taus)                                    # [sample_rate, batch_size, seq_len, num_events]
            p_t_m = p_mt / p_m                                                 # [sample_rate, batch_size, seq_len, num_events]
            p_gap = p_t_m - probability_threshold                              # [sample_rate, batch_size, seq_len, num_events]

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
        r = 1e6*torch.ones((self.sample_rate, batch_size, seq_len, self.num_events), dtype = torch.float32, device = self.device)
                                                                               # [sample_rate, batch_size, seq_len, num_events]
        tau_pred = median_prediction(l, r)                                     # [sample_rate, batch_size, seq_len, num_events]

        '''
        tau_pred_detached = tau_pred.detach()                                  # [sample_rate, batch_size, seq_len]
        tau_pred_detached.requires_grad = True
        probability_integral_from_t_to_inf = self.model(events_history, time_history, tau_pred_detached, mean, var)
                                                                               # [sample_rate, batch_size, seq_len, num_events]
        probability_for_each_event_at_pred_time = - torch.autograd.grad(
            outputs = probability_integral_from_t_to_inf,
            inputs = tau_pred_detached,
            grad_outputs = torch.ones_like(probability_integral_from_t_to_inf)
        )[0]                                                                   # [sample_rate, batch_size, seq_len, num_events]
        tau_pred_detached.requires_grad = False
        probability_for_each_event_at_pred_time = probability_for_each_event_at_pred_time
                                                                               # [sample_rate, batch_size, seq_len, num_events]
        tau_pred = (tau_pred * probability_for_each_event_at_pred_time).sum(dim = 0)
                                                                               # [batch_size, seq_len, num_events]
        '''
        tau_pred = tau_pred.mean(dim = 0)                                      # [batch_size, seq_len, num_events]

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

        MAX_sampled_seq = 2500
        seq_length = 1

        while seq_length < MAX_sampled_seq:
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
            seq_length += 1

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
            dist = torch.distributions.uniform.Uniform(torch.tensor(its_lower_bound), torch.tensor(its_upper_bound))
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

        MAX_sampled_seq = 2500
        seq_length = 1

        while seq_length < MAX_sampled_seq:
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
            seq_length += 1

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
            dist = torch.distributions.uniform.Uniform(torch.tensor(its_lower_bound), torch.tensor(its_upper_bound))
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

        f1_2, top_k, probability_sum, tau_pred_all_event, maes_avg, maes \
            = self.mean_absolute_error_e(events_history, events_next, time_history, time_next, mask_next, mean, var)

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
        sampled_mask_history_event_time, sampled_mask_next_event_time = self.divide_history_and_next(sampled_mask_event_time)
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
        sampled_mask_history_time_event, sampled_mask_next_time_event = self.divide_history_and_next(sampled_mask_time_event)
                                                                               # 2 * [batch_size, seq_len]

        sampled_data_time_event, sampled_timestamp_time_event \
            = self.model.model_probe_function(sampled_events_history_time_event, sampled_time_history_time_event, \
                                              sampled_time_next_time_event, opt.resolution, mean, var, sampled_mask_next_time_event)


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

        f1_2, top_k, probability_sum, tau_pred_all_event, maes_avg, maes \
            = self.mean_absolute_error_e(events_history, events_next, time_history, time_next, mask_next, mean, var)
        
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

        [input_time, input_events, input_score, input_mask], (mean, var) = minibatch
        loss, time_loss_without_dummy, events_loss, the_number_of_events = model(         
                task_name = 'train', input_time = input_time, input_events = input_events, \
                input_score = input_score, input_mask = input_mask, mean = mean, var = var
        )
        
        loss.backward()
    
        time_loss_without_dummy = time_loss_without_dummy.item() / the_number_of_events
        events_loss = events_loss.item() / the_number_of_events
        
        return time_loss_without_dummy, events_loss
    

    def evaluation_step(model, minibatch, device):
        ''' Epoch operation in evaluation phase '''
        model.eval()

        [input_time, input_events, input_score, input_mask], (mean, var) = minibatch
        loss, time_loss_without_dummy, events_loss, \
        f1_pred, mae, the_number_of_events = model(task_name = 'evaluate', input_time = input_time, input_events = input_events, \
                                                   input_score = input_score, input_mask = input_mask, mean = mean, var = var)
        
        loss = loss.item() / the_number_of_events
        time_loss_without_dummy = time_loss_without_dummy.item() / the_number_of_events
        events_loss = events_loss.item() / the_number_of_events
            
        return loss, time_loss_without_dummy, events_loss, f1_pred, mae


    def postprocess(input, procedure):
        def train_postprocess(input):
            '''
            Training process
            [absolute loss, relative loss, events loss]
            '''
            return [input[0], input[1]]
        
        def test_postprocess(input):
            '''
            Evaluation process
            [absolute loss, relative loss, events loss, mae value]
            '''
            return [input[0], input[1], input[2], input[3], input[4]]
        
        return (train_postprocess(input) if procedure == 'Training' else test_postprocess(input))
    

    def log_print_format(input, procedure):
        def train_log_print_format(input):
            format_dict = {}
            format_dict['time_loss'] = input[0]
            format_dict['events_loss'] = input[1]
            format_dict['num_format'] = {'time_loss': ':6.5f', 'events_loss': ':6.5f'}
            return format_dict

        def test_log_print_format(input):
            format_dict = {}
            format_dict['loss'] = input[0]
            format_dict['time_loss'] = input[1]
            format_dict['events_loss'] = input[2]
            format_dict['f1'] = input[3]
            format_dict['mae'] = input[4]
            format_dict['num_format'] = {'loss': ':6.5f', 'time_loss': ':6.5f', \
                                         'events_loss': ':6.5f', 'f1': ':6.5f', 'mae': ':6.5f'}
            return format_dict
        
        return (train_log_print_format(input) if procedure == 'Training' else test_log_print_format(input))

    format_dict_length = 5
    
    def choose_metric(evaluation_report_format_dict, test_report_format_dict):
        '''
        [relative loss on evaluation dataset, relative loss on test dataset, event loss on test dataset]
        '''
        # return [evaluation_report_format_dict['absolute_NLL_loss'] + evaluation_report_format_dict['avg_survival_loss'], 
        #         test_report_format_dict['absolute_NLL_loss'] + test_report_format_dict['avg_survival_loss']], \
        #        ['evaluation_absolute_loss', 'test_absolute_loss']
        return [evaluation_report_format_dict['loss'], 
                test_report_format_dict['loss']], \
               ['evaluation_loss', 'test_loss']
    
    metric_number = 2 # metric number is the length of the output of choose_metric