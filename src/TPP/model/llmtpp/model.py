import torch, copy
import numpy as np
from sklearn.metrics import f1_score, top_k_accuracy_score, accuracy_score
from einops import rearrange, repeat, reduce, pack
from scipy.stats import spearmanr

from src.toolbox.metrics import L1_distance_across_events
from src.toolbox.misc import pack_one_value_to_dict, compile_model

from src.TPP.model.basic_tpp_model import BasicModel
from src.TPP.model.llmtpp.submodel import LLMTPP
from src.TPP.model.utils import *
from src.TPP.model.llmtpp.plot import *


class LLMTPPModel(BasicModel):
    def __init__(self, opt, llm_class_name, full_llm_name, d_model, \
                 d_embedding, lm_layers, device, dropout, epsilon = 1e-20, lambda_t = 1.0, lambda_e = 1.0):
        super(LLMTPPModel, self).__init__()
        self.device = device
        self.num_events = opt.info_dict['num_events']
        self.start_time = opt.info_dict['t_0']
        self.end_time = opt.info_dict['T']
        self.epsilon = epsilon
        self.lambda_t = lambda_t
        self.lambda_e = lambda_e

        self.model = LLMTPP(llm_class_name = llm_class_name, full_llm_name = full_llm_name, \
                            num_events = self.num_events, d_model = d_model, d_embedding = d_embedding, \
                            lm_layers = lm_layers, dropout = dropout, device = device)
        
        self.model = compile_model(self.model, opt.compile)


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
            'mae_e_and_f1': self.get_mae_e_and_f1,
            'intensity': self.intensity,
            'integral': self.integral,
            'probability': self.probability,
            'debug': self.debug
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


    def train_procedure(self, input_time, input_events, input_score, input_mask, mean, std):
        time_history, time_next = self.divide_history_and_next(input_time)     # 2 * [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # 2 * [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(input_mask)     # [batch_size, seq_len]

        pred_time, pred_event_prob = self.model('train', events_history = events_history, time_history = time_history, \
                                                mask_history = mask_history, mean = mean, std = std)
                                                                               # [batch_size, seq_len, num_events] * 2
        '''
        Remove the probability of the dummy event by mask.
        '''
        mask_next_without_dummy = self.remove_dummy_event_from_mask(mask_next) # [batch_size, seq_len]
        events_next_without_dummy = events_next * mask_next_without_dummy      # [batch_size, seq_len]

        the_number_of_events = mask_next_without_dummy.sum().item()

        # cross entropy loss between p_{real} and p_{pred}.
        events_loss_without_dummy = torch.nn.functional.cross_entropy(rearrange(pred_event_prob, 'b s ne -> b ne s'), \
                                                                                events_next_without_dummy, reduction = 'none')
                                                                               # [batch_size, seq_len]
        events_loss_without_dummy = events_loss_without_dummy * mask_next_without_dummy
                                                                               # [batch_size, seq_len]
        events_loss_without_dummy = events_loss_without_dummy.sum()

        time_loss_without_dummy = self.loss(pred_time = pred_time, time_next = time_next, \
                                            event_next = events_next_without_dummy, mask_next = mask_next_without_dummy)
        time_loss_without_dummy = self.lambda_t * time_loss_without_dummy
        events_loss_without_dummy = self.lambda_e * events_loss_without_dummy
        loss = time_loss_without_dummy + events_loss_without_dummy

        # we need time_loss_without_dummy to compare our distribution against the ground truth.
        return loss, time_loss_without_dummy, events_loss_without_dummy, the_number_of_events


    def evaluate_procedure(self, input_time, input_events, input_score, input_mask, mean, std):
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
        
        mae, f1 = self.mean_absolute_error_and_f1(events_history = events_history, time_history = time_history, \
                                                  events_next = events_next, time_next = time_next, \
                                                  mask_history = mask_history, mask_next = mask_next_without_dummy, \
                                                  mean = mean, std = std)      # [batch_size, seq_len] + scalar
        mae = mae.sum().item() / the_number_of_events

        pred_time, pred_event_prob = self.model('evaluate', events_history = events_history, time_history = time_history, \
                                                mask_history = mask_history, mean = mean, std = std)
                                                                               # [batch_size, seq_len, num_events] * 2

        time_loss_without_dummy = self.loss(pred_time = pred_time, time_next = time_next, \
                                            event_next = events_next_without_dummy, mask_next = mask_next_without_dummy)

        events_loss_without_dummy = torch.nn.functional.cross_entropy(rearrange(pred_event_prob, 'b s ne -> b ne s'), \
                                                                                events_next_without_dummy, reduction = 'none')
                                                                               # [batch_size, seq_len]
        events_loss_without_dummy = events_loss_without_dummy * mask_next_without_dummy
                                                                               # [batch_size, seq_len]
        events_loss_without_dummy = events_loss_without_dummy.sum()
        time_loss_without_dummy = self.lambda_t * time_loss_without_dummy
        events_loss_without_dummy = self.lambda_e * events_loss_without_dummy
        loss = time_loss_without_dummy + events_loss_without_dummy

        return loss, time_loss_without_dummy, events_loss_without_dummy, f1, mae, the_number_of_events


    def loss(self, pred_time, time_next, event_next, mask_next):
        '''
        The definition of loss.
    
        Args:
            probability:        [batch_size, seq_len, num_events]
            events_next:        [batch_size, seq_len]
            mask_next:          [batch_size, seq_len]
        '''
        # pick the time.
        event_next_mask = torch.nn.functional.one_hot(event_next, num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]
        selected_pred_time = (pred_time * event_next_mask).sum(dim = -1)       # [batch_size, seq_len]
        gap = torch.abs(selected_pred_time - time_next)                        # [batch_size, seq_len]
        masked_gap = gap * mask_next                                           # [batch_size, seq_len]
        loss = torch.pow(masked_gap, 1)                                        # [batch_size, seq_len]
        loss = torch.sum(loss)

        return loss


    def mean_absolute_error_and_f1(self, events_history, time_history, events_next, time_next, mask_history, mask_next, mean, std):
        '''
        The input should be the original minibatch
        '''
        pred_time, pred_event_prob = self.model('evaluate', events_history = events_history, time_history = time_history, \
                                                mask_history = mask_history, mean = mean, std = std)
                                                                               # [batch_size, seq_len, num_events] * 2
        events_pred_index = torch.argmax(pred_event_prob, dim = -1)            # [batch_size, seq_len]
        events_pred_index_mask = torch.nn.functional.one_hot(events_pred_index, num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]
        pred_time = (pred_time * events_pred_index_mask).sum(dim = -1)         # [batch_size, seq_len]

        mae = torch.abs(pred_time - time_next) * mask_next                     # [batch_size, seq_len]

        events_true = events_next[mask_next == 1]
        events_pred_index = events_pred_index[mask_next == 1]
        events_pred_index, events_true = move_from_tensor_to_ndarray(events_pred_index, events_true)
        f1 = f1_score(y_true = events_true, y_pred = events_pred_index, average = 'macro')
         
        return mae, f1


    def mean_absolute_error_e(self, events_history, events_next, time_history, time_next, mask_history, mask_next, mean, std, return_mean = True):
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
        * std             type: float shape: N/A
                          The mean of all $ t_i - t_{i - 1} $ in the entire dataset. Dataloader is responsible to provide
                          this value if needed.
        Outputs:
        * mae             type: torch.tensor shape: [batch_size, seq_len]
                          MAE(Mean Absolute Error) between predicted time and ground truth.
        * tau_pred        type: torch.tensor shape: [batch_size, seq_len]
                          Time predicted by the sum of all intensity functions $ \\lambda^*(m, t) $ over $ m $.
        '''

        tau_pred_all_event, pred_event_prob = self.model('evaluate', events_history = events_history, time_history = time_history, \
                                                mask_history = mask_history, mean = mean, std = std)
                                                                               # [batch_size, seq_len, num_events] * 2
        predict_index = torch.argmax(pred_event_prob, dim = -1)                # [batch_size, seq_len]
        probability_integral_sum = torch.sum(pred_event_prob, dim = -1)        # [batch_size, seq_len]
        
        f1, top_k_acc = get_f1_and_top_k_acc_in_mae_e(events_next, pred_event_prob, mask_next, self.num_events)

        predict_index_one_hot_mask = torch.nn.functional.one_hot(predict_index, num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]
        events_next_one_hot_mask = torch.nn.functional.one_hot(events_next, num_classes = self.num_events)
                                                                               # [batch_size, seq_len, num_events]

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

        return f1, top_k_acc, probability_integral_sum, tau_pred_all_event, \
               (mae_per_event_with_predict_index_avg, mae_per_event_with_event_next_avg), \
               (mae_per_event_with_predict_index, mae_per_event_with_event_next)


    def sample_event_seq(self, number_of_sampled_sequences, end_time, mean, std):
        '''
        This function will sample x sequences by the learned probability distribution following the time-event prediction procedure.
        Steps:
        1. Sample a time \\(t_s\\) from p^*(t) = \\sum{n \\in M}{p^*(m, t)} referring to existing history
        2. Judge the mark of this event by comparing \\(\\lambda^*(m, t_s)\\).
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
                self.sample_one_patch_from_model(number_of_sampled_sequences, events_history_for_sampling, time_history_for_sampling, mean, std)
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


    def sample_one_patch_from_model(self, number_of_sampled_sequences, events_history_for_sampling, time_history_for_sampling, mean, std):
        def evaluate_sample(integral_from_zero_to_inf, taus):
            taus = repeat(taus, '... -> ... ne', ne = self.num_events)         # [number_of_sampled_sequences, 1, num_events]
            probability_integral_from_t_to_inf_for_sample = self.model.sample(events_history_for_sampling, time_history_for_sampling, taus, mean, std)
                                                                               # [number_of_sampled_sequences, 1, num_events]
            probability_integral_from_t_to_inf_for_sample = probability_integral_from_t_to_inf_for_sample.detach()
                                                                               # [number_of_sampled_sequences, 1, num_events]
            # P_m(t) = \\int_{0}^{t}{p(t|m, \\mathcal{H})}
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
        integral_from_zero_to_inf = self.model.sample(events_history_for_sampling, time_history_for_sampling, time_next_zero, mean = mean, std = std)
                                                                               # [number_of_sampled_sequences, 1, num_events]
        integral_from_zero_to_inf = integral_from_zero_to_inf.detach()         # [number_of_sampled_sequences, 1, num_events]
        tau_sampled = median_prediction_sample(integral_from_zero_to_inf, l, r)# [number_of_sampled_sequences, 1]
        repeated_tau_sampled = repeat(tau_sampled, 'b s -> b s ne', ne = self.num_events)
                                                                               # [number_of_sampled_sequences, 1, num_events]
        repeated_tau_sampled.requires_grad = True
        integral_from_sampled_time_to_inf = self.model(events_history_for_sampling, time_history_for_sampling, repeated_tau_sampled, mean = mean, std = std)
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


    def intensity(self, input_data, opt):
        '''
        Function prober, used by tpp_ploter to draw plots.

        Args:
        * input_data  type: list shape: [[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]
                      The original minibatch. Detailed information is available in extract_plot_data()
        * resolution  type: int shape: N/A
                      How many interpretive numbers we have between an event interval?
        '''

        return NotImplementedError('LLMTPP directly generates the next patch, so intensity function is unavailable.')


    def integral(self, input_data, opt):
        '''
        Function prober, used by tpp_ploter to draw plots.

        Args:
        * input_data  type: list shape: [[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]
                      The original minibatch. Detailed information is available in extract_plot_data()
        * resolution  type: int shape: N/A
                      How many interpretive numbers we have between an event interval?
        '''
        return NotImplementedError('LLMTPP directly generates the next patch, so intensity integral is unavailable.')


    def probability(self, input_data, opt):
        '''
        Function prober, used by tpp_ploter to draw plots.

        Args:
        * input_data  type: list shape: [[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]
                      The original minibatch. Detailed information is available in extract_plot_data()
        * resolution  type: int shape: N/A
                      How many interpretive numbers we have between an event interval?
        '''
        return NotImplementedError('LLMTPP directly generates the next patch, so probability distribution is unavailable.')


    def debug(self, input_data, opt):
        '''
        Args:
        time: [batch_size(always 1), seq_len + 1]
              The original dataset records. 
        resolution: int
              How many interpretive numbers we have between an event interval?
        '''
        
        '''
        input_time, input_events, input_intensity, mask, mean, std = self.extract_plot_data(input_data)

        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]

        mae, f1_1 = self.mean_absolute_error_and_f1(events_history, time_history, events_next, \
                                                    time_next, mask_history, mask_next, mean, std)
                                                                               # [batch_size, seq_len]
        data = {}
        plots = plot_debug(data, timestamp, opt)

        return plots
        '''

    '''
    Evaluation over the entire dataset.
    '''
    def get_spearman_and_l1(self, input_data, opt):
        return NotImplementedError('LLMTPP directly generates the next patch, so calculating spearman and L1 distance between learned and ground-truth distribution is impossible.')


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

        return mae, f1_1

    
    def get_mae_e_and_f1(self, input_data, opt):
        input_time, input_events, input_intensity, mask, mean, std = self.extract_plot_data(input_data)
        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]

        f1_2, top_k, probability_sum, tau_pred_all_event, maes_avg, maes \
            = self.mean_absolute_error_e(events_history, events_next, time_history, \
                                               time_next, mask_history, mask_next, mean, std)
                                                                               # [batch_size, seq_len]
        mae, probability_sum, events_next = move_from_tensor_to_ndarray(mae, probability_sum, events_next)

        return maes, f1_2, probability_sum, events_next


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
        [input_time, input_events, input_score, input_mask], (mean, std) = minibatch
        loss, time_loss_without_dummy, events_loss, the_number_of_events = model(         
                task_name = 'train', input_time = input_time, input_events = input_events, \
                input_score = input_score, input_mask = input_mask, \
                mean = mean, std = std
        )
        
        loss.backward()

        loss = loss.item() / the_number_of_events
        time_loss_without_dummy = time_loss_without_dummy.item() / the_number_of_events
        events_loss = events_loss.item() / the_number_of_events
        fact = input_score.sum().item() / the_number_of_events
        
        return loss, time_loss_without_dummy, events_loss, fact
    

    def evaluation_step(model, minibatch, device):
        ''' Epoch operation in evaluation phase '''
    
        model.eval()
        [input_time, input_events, input_score, input_mask], (mean, std) = minibatch
        loss, time_loss_without_dummy, events_loss, \
        f1_pred, mae, the_number_of_events = model(task_name = 'evaluate', input_time = input_time, input_events = input_events, \
                                                   input_score = input_score, input_mask = input_mask, mean = mean, std = std)
        
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
            return [input[0], input[1], input[2]]
        
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
            format_dict['loss'] = pack_one_value_to_dict(input[0])
            format_dict['time_loss'] = pack_one_value_to_dict(input[1])
            format_dict['events_loss'] = pack_one_value_to_dict(input[2])
            return format_dict

        def test_log_print_format(input):
            format_dict = {}
            format_dict['loss'] = pack_one_value_to_dict(input[0])
            format_dict['time_loss'] = pack_one_value_to_dict(input[1])
            format_dict['events_loss'] = pack_one_value_to_dict(input[2])
            format_dict['f1'] = pack_one_value_to_dict(input[3])
            format_dict['mae'] = pack_one_value_to_dict(input[4], '2.8f')
            return format_dict
        
        return (train_log_print_format(input) if procedure == 'Training' else test_log_print_format(input))

    format_dict_length = 5
    
    def choose_metric(evaluation_report_format_dict, test_report_format_dict):
        '''
        [relative loss on evaluation dataset, relative loss on test dataset, event loss on test dataset]
        '''
        return [evaluation_report_format_dict['loss'], 
                test_report_format_dict['loss']], \
               ['evaluation_loss', 'test_loss']
    
    metric_number = 2 # metric number is the length of the output of choose_metric