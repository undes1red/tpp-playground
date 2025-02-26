import torch, copy
import torch.nn.functional as F
from einops import rearrange, repeat, reduce, pack
from sklearn.metrics import f1_score, roc_auc_score

from src.toolbox.misc import check_tensor, move_from_tensor_to_ndarray, pack_one_value_to_dict
from src.toolbox.integration import approximate_integration
from src.toolbox.metrics import L1_distance_between_two_funcs

from src.TPP.model.basic_tpp_model import memory_ceiling, BasicModel
from src.TPP.model.sahp.plot import *
from src.TPP.model.sahp.submodel import SAHP
from src.TPP.model.sahp.sample import sample_time
from src.TPP.model.utils import get_f1_and_top_k_acc_in_mae_e, decide_resolution_inf_and_resolution_between_events, pick_log_probability


class SAHPWrapper(BasicModel):
    def __init__(self, opt, device, d_input = 64, d_rnn = 64, d_hidden = 256, n_layers = 3,
                 n_head = 3, d_qk = 64, d_v = 64, dropout = 0.1, epsilon = 1e-20, sample_rate = 32,
                 mae_step = 4, mae_e_step = 4, integration_sample_rate = 100, survival_loss_during_training = True):
        super(SAHPWrapper, self).__init__()
        self.device = device
        self.compile_or_not = opt.compile
        self.num_events = opt.info_dict['num_events']
        self.start_time = opt.info_dict['t_0']
        self.end_time = opt.info_dict['T']
        self.integration_sample_rate = integration_sample_rate
        self.epsilon = epsilon
        self.survival_loss_during_training = survival_loss_during_training
        self.sample_rate = sample_rate
        self.mae_step = mae_step
        self.mae_e_step = mae_e_step
        self.bisect_early_stop_threshold = 1e-4
        self.max_step = 50

        self.model = SAHP(num_events = self.num_events, d_input = d_input, d_rnn = d_rnn, d_hidden = d_hidden, \
                          n_layers = n_layers, n_head = n_head, d_qk = d_qk, d_v = d_v, dropout = dropout, \
                          device = device, integration_sample_rate = integration_sample_rate)
    

    def divide_history_and_next(self, input):
        '''
        What divide_history_and_next should do?
        [a, b, c, d, e, pad, pad, pad]
        [1, 1, 1, 1, 1, 0,   0,   0]
                    |
                    |
                    |
                    \/
        [a, b, c, d, e, pad, pad], [b, c, d, e, pad, pad, pad]
        [1, 1, 1, 1, 1, 0,   0  ], [1, 1, 1, 1, 0,   0,   0  ]
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
            'which_event_occurs_first': self.get_which_event_first,
            'samples_from_et': self.samples_from_et,

            # Functions for the EHD task.
            'ehd_perplexity': self.ehd_perplexity,
            'ehd_event_emb': self.get_event_embedding,

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


    '''
    Functions for model training.
    '''
    def train_procedure(self, time, events, mask, mean, std):
        '''
        Check if events data is present.
        Now, we assume that no event data is available.
        Args:
        1. time: the sequence containing events' timestamps. shape: [batch_size, seq_len + 1]
        2. events: the sequence containing information about events. shape: [batch_size, seq_len + 1]
        3. mask: filter out the padding events in the event batches. shape: [batch_size, seq_len + 1]
        '''
        time_history, time_next = self.divide_history_and_next(time)           # [batch_size, seq_len] * 2
        events_history, events_next = self.divide_history_and_next(events)     # [batch_size, seq_len] * 2
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len] * 2

        integral_all_events, intensity_all_events = self.model(time_history, time_next, events_history, mask_history)
                                                                               # 2 * [batch_size, seq_len, num_events]

        mask_next_without_dummy = self.remove_dummy_event_from_mask(mask_next) # [batch_size, seq_len]
        event_next_without_dummy = (mask_next_without_dummy * events_next).long()
                                                                               # [batch_size, seq_len]
        the_number_of_events = mask_next_without_dummy.sum().item()

        # L = \\sum_{i}{\\lambda^_k*(t_i)} + \\int_{t_0}^{t_n}{\\sum_{k}{\\lambda^*_k(\\tau)}d\\tau}
        log_likeli_loss_without_dummy, marker_loss_without_dummy = self.loss_function(
             integral_all_events = integral_all_events, intensity_all_events = intensity_all_events, \
             events_next = event_next_without_dummy, mask_next = mask_next_without_dummy
        )

        loss_survival = 0
        if self.survival_loss_during_training:
            # survival_loss = \\int_{t_n}^{T}{\\sum_{k}{\\lambda^*_k(\\tau)}d\\tau}
            dummy_event_index = mask_next.sum(dim = -1) - 1                    # [batch_size]
            integral_survival = integral_all_events.sum(dim = -1).gather(index = dummy_event_index.unsqueeze(dim = -1), dim = -1)
                                                                               # [batch_size, 1]
            loss_survival = integral_survival.sum()

        loss = log_likeli_loss_without_dummy + loss_survival

        return loss, log_likeli_loss_without_dummy, marker_loss_without_dummy, the_number_of_events


    '''
    Functions for model evaluation
    '''
    @torch.no_grad()
    def evaluate_procedure(self, time, events, mask, mean, std):
        '''
        Check if events data is present.
        Now, we assume that no event data is available.
        Args:
        1. time: the sequence containing events' timestamps. shape: [batch_size, seq_len + 1]
        2. events: the sequence containing information about events. shape: [batch_size, seq_len + 1]
        3. mask: filter out the padding events in the event batches. shape: [batch_size, seq_len + 1]
        '''
        time_history, time_next = self.divide_history_and_next(time)           # [batch_size, seq_len] * 2
        events_history, events_next = self.divide_history_and_next(events)     # [batch_size, seq_len] * 2
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len] * 2

        mask_next_without_dummy = self.remove_dummy_event_from_mask(mask_next) # [batch_size, seq_len]
        event_next_without_dummy = (mask_next_without_dummy * events_next).long()
                                                                               # [batch_size, seq_len]
        the_number_of_events = mask_next_without_dummy.sum().item()

        integral_all_events_time_next, intensity_all_events_time_next = self.model(time_history, time_next, events_history, mask_history)
                                                                               # 2 * [batch_size, seq_len, num_events]
        mae, f1_pred_time = self.mean_absolute_error_and_f1(events_history = events_history, time_history = time_history, 
                                                            events_next = events_next, time_next = time_next, 
                                                            mask_history = mask_history, mask_next = mask_next_without_dummy, mean = mean, std = std)
        mae = mae.sum().item() / the_number_of_events
        # NLL loss and event loss at time_next
        # L = \\sum_{i}{\\lambda^_k*(t_i)} + \\int_{t_0}^{t_n}{\\sum_{k}{\\lambda^*_k(\\tau)}d\\tau}
        log_likeli_loss_time_next_without_dummy, marker_loss_time_next_without_dummy = self.loss_function(
             integral_all_events = integral_all_events_time_next, intensity_all_events = intensity_all_events_time_next, \
             events_next = event_next_without_dummy, mask_next = mask_next_without_dummy
        )
        # Survival probability: \\int_{t_N}^{T}{\\sum_{k}\\lambda_k^(\\tau)d\\tau}
        dummy_event_index = mask_next.sum(dim = -1) - 1                        # [batch_size]
        integral_survival = integral_all_events_time_next.sum(dim = -1).gather(index = dummy_event_index.unsqueeze(dim = -1), dim = -1)
                                                                               # [batch_size, 1]
        loss_survival = integral_survival.mean()

        return log_likeli_loss_time_next_without_dummy, loss_survival, marker_loss_time_next_without_dummy, \
               mae, f1_pred_time, the_number_of_events


    '''
    Loss functions
    '''
    def loss_function(self, integral_all_events, intensity_all_events, events_next, mask_next):
        """ Log-likelihood of sequence. """
        type_mask = F.one_hot(events_next, num_classes = self.num_events)      # [batch_size, seq_len, num_events]
        '''
        MTPP loss function
        '''
        selected_intensity = (intensity_all_events * type_mask).sum(dim = -1)  # [batch_size, seq_len]
        log_intensity = torch.log(selected_intensity + self.epsilon)           # [batch_size, seq_len]
        nll = -log_intensity + integral_all_events.sum(dim = -1)               # [batch_size, seq_len]
    
        mtpp_loss = torch.sum(nll * mask_next)

        '''
        Event loss function. Only for evaluation, do NOT use this loss as a part of the training loss.
        '''
        events_prediction_probability = torch.log(intensity_all_events + self.epsilon)
                                                                               # [batch_size, seq_len, num_events]
        events_prediction_probability = F.softmax(events_prediction_probability, dim = -1)
                                                                               # [batch_size, seq_len, num_events]
        reshaped_events_prediction_probability = rearrange(events_prediction_probability, 'b s ne -> b ne s')
                                                                               # [batch_size, num_events, seq_len]
        events_loss = F.cross_entropy(input = reshaped_events_prediction_probability, target = events_next, reduction = 'none')
                                                                               # [batch_size, seq_len]
        events_loss = (events_loss * mask_next).sum()

        return mtpp_loss, events_loss


    sample_time = sample_time
    

    @torch.no_grad()
    def mean_absolute_error_and_f1(self, events_history, time_history, events_next, time_next, mask_history, mask_next, mean, std):
        pred_time = self.sample_time(sampling_approach = 'its', task = 'tm',
                                time_history = time_history, events_history = events_history, mask_history = mask_history,
                                number_of_total_samples = self.sample_rate, step = self.mae_step, mean = mean, std = std)
                                                                               # [sample_rate, batch_size, seq_len]
        pred_time = pred_time.mean(dim = 0)                                    # [batch_size, seq_len]
        mae = torch.abs(pred_time - time_next) * mask_next                     # [batch_size, seq_len]

        _, intensity_all_events = self.model(time_history, pred_time, events_history, mask_history)
                                                                               # 2 * [batch_size, seq_len, num_events]
        predicted_events = torch.argmax(intensity_all_events, dim = -1)[mask_next == 1]
        events_true = events_next[mask_next == 1]
        predicted_events, events_true = move_from_tensor_to_ndarray(predicted_events, events_true)
        f1 = f1_score(y_pred = predicted_events, y_true = events_true, average = 'macro')

        return mae, f1


    @torch.no_grad()
    def mean_absolute_error_e(self, time_history, time_next, events_history, events_next, mask_history, mask_next, mean, std, return_mean = True):
        '''
        The precedure resembles the compute_integral_unbiased() but the output of small step MC takes would
        be recorded as part of the output.
        '''
        '''
        set a relatively large number as the infinity and decide resolution based on this large value and
        the memory_ceiling.
        '''
        inf_val, resolution_inf, resolution_between_events \
            = decide_resolution_inf_and_resolution_between_events(time_next, memory_ceiling, self.num_events, mean, std)
        time_next_inf = torch.ones_like(time_history, device = self.device) * inf_val

        expanded_integral_all_events_to_inf, expanded_intensity_all_events_to_inf, timestamp = \
            self.model.integral_intensity_time_next_2d(events_history, time_history, time_next_inf, mask_history, resolution_inf)
                                                                               # 2 * [batch_size, seq_len, resolution_inf, num_events]

        expanded_integral_sum_over_events_to_inf = expanded_integral_all_events_to_inf.sum(dim = -1, keepdim = True)
                                                                               # [batch_size, seq_len, resolution_inf, 1]
        expanded_probability_inf = expanded_intensity_all_events_to_inf * torch.exp(-expanded_integral_sum_over_events_to_inf)
                                                                               # [batch_size, seq_len, resolution_inf, num_events]
        probability_integral_to_inf = approximate_integration(expanded_probability_inf, timestamp, dim = -2, only_integral = True)
                                                                               # [batch_size, seq_len, num_events]
        probability_integral_sum = probability_integral_to_inf.sum(dim = -1)   # [batch_size, seq_len]
        predicted_events = torch.argmax(probability_integral_to_inf, dim = -1) # [batch_size, seq_len]

        f1, top_k_acc = get_f1_and_top_k_acc_in_mae_e(events_next, probability_integral_to_inf, mask_next, self.num_events)

        tau_pred_all_event = self.sample_time(sampling_approach = 'its', task = 'mt', 
                                         events_history = events_history, time_history = time_history, mask_history = mask_history,
                                         p_m = probability_integral_to_inf, resolution = resolution_between_events, number_of_total_samples = self.sample_rate, step = self.mae_e_step, inf_val = inf_val,
                                         mean = mean, std = std)               # [sample_rate, batch_size, seq_len, num_events]

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
    

    @torch.no_grad()
    def figure_intensity(self, input_data, opt):
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
            'input_intensity': input_intensity
            }
        plots = generate_intensity_figure(data, timestamp, opt)
        
        return plots


    @torch.no_grad()
    def figure_integral(self, input_data, opt):
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
            'input_intensity': input_intensity
            }
        plots = generate_integral_figure(data, timestamp, opt)
        return plots


    @torch.no_grad()
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
            'input_intensity': input_intensity
            }
        
        generate_probability_figure(data, timestamp, opt)


    @torch.no_grad()
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
        data, timestamp = self.model.model_probe_function(events_history, time_history, time_next, \
                                                          mask_history, mask_next, opt.resolution)
        f1_2, top_k, probability_sum, _, tau_pred_all_event, maes_avg, maes \
            = self.mean_absolute_error_e(time_history, time_next, events_history, events_next, mask_history, mask_next, mean, std, return_mean = False)

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

        generate_debug_figure(data, timestamp, opt)


    '''
    Evaluation over the entire dataset.
    '''
    @torch.no_grad()
    def get_spearman_and_l1(self, input_data, opt):
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
    

    @torch.no_grad()
    def get_mae_and_f1(self, input_data, opt):
        input_time, input_events, input_intensity, mask, mean, std = self.extract_plot_data(input_data)
        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]

        mae, f1_1 = self.mean_absolute_error_and_f1(events_history, time_history, events_next, \
                                                    time_next, mask_history, mask_next, mean, std)
                                                                               # [batch_size, seq_len]
        mae, events_next = move_from_tensor_to_ndarray(mae, events_next)

        return mae, f1_1, events_next


    @torch.no_grad()
    def get_mae_e_and_f1(self, input_data, opt):
        input_time, input_events, input_intensity, mask, mean, std = self.extract_plot_data(input_data)
        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(mask)           # [batch_size, seq_len]

        f1_2, top_k, probability_sum, probability_integral_to_inf, tau_pred_all_event, maes_avg, maes \
            = self.mean_absolute_error_e(time_history, time_next, events_history, \
                                         events_next, mask_history, mask_next, mean, std)
        
        _, maes, probability_sum, probability_integral_to_inf, events_next = move_from_tensor_to_ndarray(*maes, probability_sum, probability_integral_to_inf, events_next)

        return maes, f1_2, probability_sum, probability_integral_to_inf, events_next


    @torch.no_grad()
    def get_which_event_first(self, input_data, opt):
        '''
        Hyperparameters
        '''
        the_number_of_samples = 10000
        substep = 500

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
                                              p_m = probability_integral_to_inf, resolution = resolution_between_events, number_of_total_samples = the_number_of_samples, step = substep, inf_val = inf_val, 
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
        input_time, input_events, input_intensity, mask, mean, std = self.extract_plot_data(input_data)
        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # [batch_size, seq_len]
        the_number_of_samples = 3000
        substep = 500

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
                                              p_m = probability_integral_to_inf, resolution = resolution_between_events, number_of_total_samples = the_number_of_samples, step = substep, inf_val = inf_val, 
                                              mean = mean, std = std)          # [sample_rate, batch_size, seq_len, num_events]

        return tau_pred_all_event, probability_integral_to_inf


    def get_event_embedding(self, input_events):
        return self.model.get_event_embedding(input_events)                     # [batch_size, seq_len, d_history]


    def ehd_perplexity(self, padded_filtered_time, padded_filtered_events, padded_filtered_event_embeddings, padded_filtered_masks, seq_len_x, mean, std):
        padded_filtered_time_history, padded_filtered_time_next = self.divide_history_and_next(padded_filtered_time)
                                                                               # 2 * [batch_size, filtered_seq_len - 1]
        padded_filtered_events_history, padded_filtered_events_next = self.divide_history_and_next(padded_filtered_events)
                                                                               # 2 * [batch_size, filtered_seq_len- 1]
        padded_filtered_events_embeddings_history, padded_filtered_events_embeddings_next \
            = self.divide_history_and_next(padded_filtered_event_embeddings)   # 2 * [batch_size, filtered_seq_len- 1, d_history]
        padded_filtered_mask_history, padded_filtered_mask_next = self.divide_history_and_next(padded_filtered_masks)
                                                                               # [batch_size, filtered_seq_len - 1]
        the_number_of_events_per_sequence = padded_filtered_mask_next.sum(dim = -1)
                                                                               # [batch_size]
        # \\int_{t}^{+\\inf}{p(m, \\tau|\\mathcal{H})d\\tau}
        padded_filtered_intensity_integral_from_t_o_to_t, \
            padded_filtered_intensity_at_t = self.model(padded_filtered_time_history, padded_filtered_time_next, \
                                                        padded_filtered_events_embeddings_history, padded_filtered_mask_history, \
                                                        custom_events_history = True)
                                                                               # [batch_size, filtered_seq_len - 1, num_events]
        padded_filtered_mask_next_without_dummy = self.remove_dummy_event_from_mask(padded_filtered_mask_next)
                                                                               # [batch_size, filtered_seq_len - 1]
        padded_filtered_events_next_without_dummy = padded_filtered_events_next * padded_filtered_mask_next_without_dummy
                                                                               # [batch_size, filtered_seq_len - 1]
        event_mask = torch.nn.functional.one_hot(padded_filtered_events_next_without_dummy, num_classes = self.num_events)
                                                                               # [batch_size, filtered_seq_len - 1, num_events]
        padded_filtered_intensity_at_t = (padded_filtered_intensity_at_t * event_mask).sum(dim = -1)
                                                                               # [batch_size, filtered_seq_len - 1]
        log_probability = torch.log(padded_filtered_intensity_at_t + self.epsilon) - padded_filtered_intensity_integral_from_t_o_to_t.sum(dim = -1)
                                                                               # [batch_size, filtered_seq_len - 1]
        log_probability_x = pick_log_probability(log_probability, the_number_of_events_per_sequence, seq_len_x)
                                                                               # [batch_size, seq_len_x]
        # -\\frac{1}{N} \\log p(\\mathbf{x}_o|\\mathcal{H})
        log_perplexity = -log_probability_x.mean(dim = -1)                     # [batch_size]

        return log_perplexity


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


    '''
    Static methods
    '''
    def train_step(model, minibatch, device):
        ''' Epoch operation in training phase'''
        model.train()

        time, events, score, mask = minibatch[0]                                 # 3 * [batch_size, seq_len + 1, 1] & [batch_size, seq_len, 1]
        mean, std = minibatch[1]

        loss, time_loss_without_dummy, events_loss, the_number_of_events = model('train', time, events, mask, mean, std)

        loss.backward()
    
        time_loss_without_dummy = time_loss_without_dummy.item() / the_number_of_events
        events_loss = events_loss.item() / the_number_of_events
        fact = score.sum().item() / the_number_of_events
        
        return time_loss_without_dummy, fact, events_loss
    

    def evaluation_step(model, minibatch, device):
        ''' Epoch operation in evaluation phase '''
        model.eval()

        time, events, score, mask = minibatch[0]                                # 3 * [batch_size, seq_len + 1, 1] & [batch_size, seq_len, 1]
        mean, std = minibatch[1]

        time_loss, loss_survival, events_loss, mae, f1, the_number_of_events = model('evaluate', time, events, mask, mean, std)

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
    smaller_is_better = [True, True]