import torch, copy
import numpy as np
from sklearn.metrics import f1_score, top_k_accuracy_score, accuracy_score
from einops import rearrange, repeat, reduce, pack
from scipy.stats import spearmanr

from src.utils import pack_one_value_to_dict
from src.TPP.model.basic_tpp_model import BasicModel
from src.TPP.model.llmtpp_ad.submodel import LLMTPP
from src.TPP.model.utils import *
from src.TPP.model.llmtpp_ad.plot import *


class LLMTPPModel(BasicModel):
    def __init__(self, info_dict, llm_class_name, full_llm_name, patch_size, d_model, \
                 d_embedding, lm_layers, d_lm_embedding, device, dropout, epsilon = 1e-20, \
                lambda_t = 1.0, lambda_e = 1.0):
        super(LLMTPPModel, self).__init__()
        self.device = device
        self.patch_size = patch_size
        self.num_events = info_dict['num_events']
        self.start_time = info_dict['t_0']
        self.end_time = info_dict['T']
        self.epsilon = epsilon
        self.lambda_t = lambda_t
        self.lambda_e = lambda_e

        self.model = LLMTPP(llm_class_name = llm_class_name, full_llm_name = full_llm_name, \
                            patch_size = patch_size, d_model = d_model, \
                            d_embedding = d_embedding, num_events = self.num_events, \
                            lm_layers = lm_layers, dropout = dropout, d_lm_embedding = d_lm_embedding, \
                            device = device)


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
            'anomaly_detection': self.anomaly_detection,
            'graph': self.plot,
        }

        return task_mapper[task_name](*args, **kwargs)
    

    def remove_dummy_event_from_mask(self, mask, reference_tensor = None):
        '''
        Remove the probability of the dummy event by mask.
        '''
        if reference_tensor is None:
            reference_tensor = copy.deepcopy(mask)

        mask_without_dummy = torch.zeros_like(mask)
        for idx, (mask_per_seq, reference_mask_per_seq) in enumerate(zip(mask, reference_tensor)):
            dummy_index = reference_mask_per_seq.sum() - 1
            mask_without_dummy_per_seq = copy.deepcopy(mask_per_seq.detach())
            mask_without_dummy_per_seq[dummy_index] = 0
            mask_without_dummy[idx] = mask_without_dummy_per_seq
        
        return mask_without_dummy


    def pad_sequences(self, patch_len, *args):
        result = []
        for original_tensor in args:
            seq_len = original_tensor.shape[-1]
            target_pred_seq_length = self.patch_size * patch_len
            p1d = (0, target_pred_seq_length - seq_len)
            padded_mask_next = torch.nn.functional.pad(original_tensor, p1d, 'constant', 0)
                                                                                   # [batch_size, target_pred_seq_length]
            result.append(padded_mask_next)
        

        if len(result) == 1:
            return result[0]
        else:
            return result


    def train_procedure(self, input_time, input_mask, mean, var):
        time_history, time_next = self.divide_history_and_next(input_time)     # 2 * [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(input_mask)     # [batch_size, seq_len]
        batch_size, seq_len = time_next.shape

        pred_time = self.model('train', input_time = time_history, input_mask = mask_history, \
                               mean = mean, var = var)                         # [batch_size, patch_len, patch_size] + [batch_size, patch_len, patch_size, num_events]
        pred_time = rearrange(pred_time, 'b np lp -> b (np lp)')               # [batch_size, patch_len * patch_size]
        mask_next_without_dummy = self.remove_dummy_event_from_mask(mask_next) # [batch_size, seq_len]
        
        # Calculate the value of losses and other metrics.        
        mae = torch.abs(time_next - pred_time[..., :seq_len])                  # [batch_size, seq_len]
        time_loss_without_dummy = (mae * mask_next_without_dummy).sum()        # [batch_size, seq_len]
        the_number_of_events = mask_next_without_dummy.sum().item()

        loss = self.lambda_t * time_loss_without_dummy
        # we need time_loss_without_dummy to compare our distribution against the ground truth.
        return loss, time_loss_without_dummy, the_number_of_events

    
    @torch.no_grad()
    def evaluate_procedure(self, input_time, input_label, input_mask, mean, var):
        time_history, time_next = self.divide_history_and_next(input_time)     # 2 * [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(input_mask)     # [batch_size, seq_len]
        batch_size, seq_len = time_next.shape

        pred_time = self.model('evaluate', input_time = time_history, input_mask = mask_history, \
                               mean = mean, var = var)                         # [batch_size, patch_len, patch_size] + [batch_size, patch_len, patch_size, num_events]
        pred_time = rearrange(pred_time, 'b np lp -> b (np lp)')               # [batch_size, patch_len * patch_size]
        mask_next_without_dummy = self.remove_dummy_event_from_mask(mask_next) # [batch_size, seq_len]

        # Calculate the value of losses and other metrics.        
        mae = torch.abs(time_next - pred_time[..., :seq_len])                  # [batch_size, seq_len]
        time_loss_without_dummy = (mae * mask_next_without_dummy).sum()
        the_number_of_events = mask_next_without_dummy.sum().item()
        mean_mae = time_loss_without_dummy.item() / the_number_of_events

        loss = self.lambda_t * time_loss_without_dummy

        return loss, time_loss_without_dummy, mean_mae, the_number_of_events


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
        mae, pred_time, pred_event_prob = self.mean_absolute_error(events_history, time_history, time_next, \
                                                                   mask_history, mask_next, mean, var)
                                                                               # [batch_size, seq_len * patch_size, num_events] + [batch_size, seq_len * patch_size]
        target_length = pred_time.shape[-1]
        padded_mask_next = self.pad_sequences(int(target_length / self.patch_size), mask_next)

        events_pred_index = torch.argmax(pred_event_prob, dim = -1)[padded_mask_next == 1]
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
                                                                               # [batch_size, seq_len, patch_size, num_events] + [batch_size, seq_len, patch_size]
        patch_len = pred_time.shape[-2]
        pred_event_prob = rearrange(pred_event_prob, '... pl ps ne -> ... (pl ps) ne')
                                                                               # [batch_size, patch_len * patch_size, num_events]
        pred_time = rearrange(pred_time, '... p n -> ... (p n)')               # [batch_size, patch_len * patch_size]
        
        padded_mask_next, padded_time_next \
            = self.pad_sequences(patch_len, mask_next, time_next)      
        mae = torch.abs(pred_time - padded_time_next) * padded_mask_next       # [batch_size, seq_len]

        return mae, pred_time, pred_event_prob


    def sample_event_seq(self, number_of_sampled_sequences, end_time, mean, var):
        '''
        This function will sample x sequences by the learned probability distribution following the time-event prediction procedure.
        Steps:
        1. Sample a time \(t_s\) from p^*(t) = \\sum{n \\in M}{p^*(m, t)} referring to existing history
        2. Judge the mark of this event by comparing \(\\lambda^*(m, t_s)\).
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
                self.sample_one_patch_from_model(number_of_sampled_sequences, events_history_for_sampling, time_history_for_sampling, mean, var)
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


    def sample_one_patch_from_model(self, number_of_sampled_sequences, events_history_for_sampling, time_history_for_sampling, mean, var):
        def evaluate_sample(integral_from_zero_to_inf, taus):
            taus = repeat(taus, '... -> ... ne', ne = self.num_events)         # [number_of_sampled_sequences, 1, num_events]
            probability_integral_from_t_to_inf_for_sample = self.model.sample(events_history_for_sampling, time_history_for_sampling, taus, mean, var)
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


    def plot(self, minibatch, opt):
        plot_type_to_functions = {
            'intensity': self.intensity,
            'integral': self.integral,
            'probability': self.probability,
            'debug': self.debug
        }
    
        return plot_type_to_functions[opt.subtask_name](minibatch, opt)


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
        (input_time, input_labels, mask), (mean, var) = minibatch

        return input_time, input_labels, mask, mean, var


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
        return NotImplementedError('LLMTPP directly generates the next patch, so calculating spearman and L1 distance between learned and ground-truth distribution is impossible.')


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
        return NotImplementedError('LLMTPP directly generates the next patch, so searching for the time prediction given mark is impossible.')


    @torch.no_grad()
    def anomaly_detection(self, input_data, opt):
        input_time, input_label, input_mask, mean, var = self.extract_plot_data(input_data)

        time_history, time_next = self.divide_history_and_next(input_time)     # 2 * [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(input_mask)     # 2 * [batch_size, seq_len]
        _, label_next = self.divide_history_and_next(input_label)              # [batch_size, seq_len]
        batch_size, seq_len = time_next.shape

        pred_time = self.model('evaluate', input_time = time_history, input_mask = mask_history, \
                               mean = mean, var = var)                         # [batch_size, patch_len, patch_size] + [batch_size, patch_len, patch_size, num_events]
        pred_time = rearrange(pred_time, 'b np lp -> b (np lp)')               # [batch_size, patch_len * patch_size]
        mask_next_without_dummy = self.remove_dummy_event_from_mask(mask_next) # [batch_size, seq_len]

        # Calculate the value of losses and other metrics.        
        mae = torch.abs(time_next - pred_time[..., :seq_len])                  # [batch_size, seq_len]
        selected_mae = mae[mask_next_without_dummy == 1]
        selected_label = label_next[mask_next_without_dummy == 1]

        selected_mae, selected_label = move_from_tensor_to_ndarray(selected_mae, selected_label)

        return selected_mae, selected_label


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
        [input_time, input_label, input_mask], (mean, var) = minibatch
        loss, time_loss_without_dummy, the_number_of_events = model(         
                task_name = 'train', input_time = input_time, input_mask = input_mask, \
                mean = mean, var = var
        )
        
        loss.backward()

        loss = loss.item() / the_number_of_events
        time_loss_without_dummy = time_loss_without_dummy.item() / the_number_of_events
        
        return loss, time_loss_without_dummy
    

    def evaluation_step(model, minibatch, device):
        ''' Epoch operation in evaluation phase '''
    
        model.eval()
        [input_time, input_label, input_mask], (mean, var) = minibatch
        loss, time_loss_without_dummy, \
        mae, the_number_of_events = model(task_name = 'evaluate', \
                                          input_time = input_time, input_label = input_label, \
                                          input_mask = input_mask, mean = mean, var = var)
        
        loss = loss.item() / the_number_of_events
        time_loss_without_dummy = time_loss_without_dummy.item() / the_number_of_events
            
        return loss, time_loss_without_dummy, mae


    def postprocess(input, procedure):
        def train_postprocess(input):
            '''
            Training process
            [loss, time loss]
            '''
            return [input[0], input[1]]
        
        def test_postprocess(input):
            '''
            Evaluation process
            [loss, time loss, mae value]
            '''
            return [input[0], input[1], input[2]]
        
        return (train_postprocess(input) if procedure == 'Training' else test_postprocess(input))
    

    def log_print_format(input, procedure):
        def train_log_print_format(input):
            format_dict = {}
            format_dict['loss'] = pack_one_value_to_dict(input[0])
            format_dict['time_loss'] = pack_one_value_to_dict(input[1])
            return format_dict

        def test_log_print_format(input):
            format_dict = {}
            format_dict['loss'] = pack_one_value_to_dict(input[0])
            format_dict['time_loss'] = pack_one_value_to_dict(input[1])
            format_dict['mae'] = pack_one_value_to_dict(input[2])
            return format_dict
        
        return (train_log_print_format(input) if procedure == 'Training' else test_log_print_format(input))

    format_dict_length = 4
    
    def choose_metric(evaluation_report_format_dict, test_report_format_dict):
        '''
        [relative loss on evaluation dataset, relative loss on test dataset, event loss on test dataset]
        '''
        return [evaluation_report_format_dict['loss'], 
                test_report_format_dict['loss']], \
               ['evaluation_loss', 'test_loss']
    
    metric_number = 2 # metric number is the length of the output of choose_metric