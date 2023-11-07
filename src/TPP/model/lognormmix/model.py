import torch, copy
from sklearn.metrics import f1_score
from einops import rearrange, reduce, repeat

from src.TPP.model.lognormmix.log_norm_mix import LogNormMix
from src.TPP.model.utils import BasicModule, move_from_tensor_to_ndarray
from src.TPP.model.lognormmix.plot import *


class LogNormMixWrapper(BasicModule):
    def __init__(self, info_dict: dict, device, context_size: int = 32, mark_embedding_size: int = 32, \
                 num_mix_components: int = 16, rnn_type: str = "LSTM", probability_threshold = 0.5, \
                 survival_loss_during_training = False):
        super(LogNormMixWrapper, self).__init__()
        self.device = device
        self.num_events = info_dict['num_events']
        self.probability_threshold = probability_threshold
        self.survival_loss_during_training = survival_loss_during_training
        self.sample_rate = 32


        self.model = LogNormMix(
            self.num_events + 1,
            self.device,
            context_size,
            mark_embedding_size,
            num_mix_components,
            rnn_type,
        )
    

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


    def divide_history_and_next(self, input):
        history, next = input[:, :-1].clone(), input[:, 1:].clone()
        return history, next                                                   # [batch_size, seq_len, 1] or [batch_size, seq_len]


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


    def train_procedure(self, input_events, input_time, input_mask, mean, var):
        '''
        The shape of minibatch
        [
            [
                event_tensor,
                time_tensor,
                mask_tensor
            ],
            score,
            [
                mean,
                var
            ](if self.input_norm_data is True)
        ]
        '''

        the_number_of_events = input_mask.sum().item()
        log_prob, log_surv_last, log_p_event = self.model.log_prob(input_events, input_time, input_mask, mean, var)
                                                                               # [batch_size, seq_len + 1]
                                                                               # [batch_size, seq_len + 1]
        log_prob = log_prob * input_mask                                       # [batch_size, seq_len + 1]
        log_p_event = log_p_event * input_mask                                 # [batch_size, seq_len + 1]
        
        time_loss = self.loss_f(log_prob)
        event_loss = self.loss_f(log_p_event)
        surv_last_loss = 0
        if self.survival_loss_during_training:
            surv_last_loss = self.loss_f(log_surv_last)

        return time_loss + surv_last_loss, time_loss, event_loss, the_number_of_events


    def evaluate_procedure(self, input_events, input_time, input_mask, mean, var):
        '''
        The shape of minibatch
        [
            [
                event_tensor,
                time_tensor,
                mask_tensor
            ],
            score,
            [
                mean,
                var
            ](if self.input_norm_data is True)
        ]
        '''

        the_number_of_events = input_mask.sum().item()
        log_prob, log_surv_last, log_p_event = self.model.log_prob(input_events, input_time, input_mask, mean, var)
                                                                               # [batch_size, seq_len + 1]
        log_prob = log_prob * input_mask                                       # [batch_size, seq_len + 1]
        log_p_event = log_p_event * input_mask                                 # [batch_size, seq_len + 1]
        
        time_loss = self.loss_f(log_prob)
        event_loss = self.loss_f(log_p_event)
        surv_last_loss = self.loss_f(log_surv_last)

        mae, pred_time = self.mean_absolute_error(input_events, input_time, input_mask, mean, var)
                                                                               # [batch_size, seq_len + 1]
        mae = (mae * input_mask).sum().item() / the_number_of_events

        predicted_events_at_time_next = self.model.event_prober(input_events, input_time, input_mask, mean, var)
                                                                               # [batch_size, seq_len + 1]
        predicted_events_at_pred_time = self.model.event_prober(input_events, pred_time, input_mask, mean, var)
                                                                               # [batch_size, seq_len + 1]
        predicted_events_at_time_next = predicted_events_at_time_next[input_mask == 1]
        predicted_events_at_pred_time = predicted_events_at_pred_time[input_mask == 1]
        input_events = input_events[input_mask == 1]
        predicted_events_at_time_next, predicted_events_at_pred_time, input_events \
            = move_from_tensor_to_ndarray(predicted_events_at_time_next, predicted_events_at_pred_time, input_events)
        
        f1_pred_time = f1_score(y_pred = predicted_events_at_pred_time, y_true = input_events, average = 'macro')

        return time_loss, surv_last_loss, event_loss, mae, f1_pred_time, the_number_of_events


    def loss_f(self, loglik):
        '''
        The definition of loss.
        '''
        return (-loglik).sum()


    def mean_absolute_error(self, input_events, input_time, input_mask, mean, var):
        '''
        The input should be the original minibatch.
        MAE evaluation part for intensity-free model.
        '''
        dist = torch.distributions.uniform.Uniform(torch.tensor(0.0), torch.tensor(1.0))
        probability_threshold = dist.sample((self.sample_rate, *input_time.shape))
                                                                               # [sample_rate, batch_size, seq_len]
        probability_threshold = probability_threshold.to(self.device)

        def evaluate(taus):
            probability, _ = self.model.log_cdf(input_events, input_time, input_mask, taus, mean, var)
                                                                               # [sample_rate, batch_size, seq_len + 1]
            return probability

        def bisect_target(taus):
            return evaluate(taus) - probability_threshold
        
        def median_prediction(l, r):
            for _ in range(30):
                c = (l + r)/2
                v = bisect_target(c)
                l = torch.where(v < 0, c, l)
                r = torch.where(v >= 0, c, r)

            return (l + r)/2

        l = 0.0001*torch.ones_like(probability_threshold, dtype = torch.float32)
                                                                               # [sample_rate, batch_size, seq_len + 1]
        r = 1e6*torch.ones_like(probability_threshold, dtype = torch.float32)  # [sample_rate, batch_size, seq_len + 1]
        tau_pred = median_prediction(l, r)                                     # [sample_rate, batch_size, seq_len + 1]

        '''
        log_p, _, _ = self.model.log_prob(input_events, tau_pred, input_mask, mean, var)
                                                                               # [sample_rate, batch_size, seq_len + 1]
        p = torch.exp(log_p)                                                   # [sample_rate, batch_size, seq_len + 1]
        tau_pred = (tau_pred * p).sum(dim = 0)                                 # [batch_size, seq_len + 1]
        '''
        
        tau_pred = tau_pred.mean(dim = 0)                                      # [batch_size, seq_len]
        mae = torch.abs(tau_pred - input_time) * input_mask                    # [batch_size, seq_len]
        
        return mae, tau_pred
    

    def plot(self, minibatch, opt):
        plot_type_to_functions = {
            'intensity': self.intensity,
            'integral': self.integral,
            'probability': self.probability,
            'debug': self.debug
        }
    
        return plot_type_to_functions[opt.plot_type](minibatch, opt)


    def mean_absolute_error_and_f1(self, input_events, input_time, input_mask, mean, var):
        # Obtain dedicated MAE and predicted time.
        gap, pred_time = self.mean_absolute_error(input_events, input_time, input_mask, mean, var)
                                                                               # [batch_size, seq_len + 1]
        predicted_events  = self.model.event_prober(input_events, input_time, input_mask, mean, var)
                                                                               # [batch_size, seq_len + 1]
        
        gap = gap[input_mask == 1]                                             # [batch_size * seq_len]
        predicted_events = predicted_events[input_mask == 1]                   # [batch_size * seq_len]
        input_events = input_events[input_mask == 1]                           # [batch_size * seq_len]
        predicted_events, input_events = move_from_tensor_to_ndarray(predicted_events, input_events)

        batch_size = pred_time.shape[0]
        gap = rearrange(gap, '(b s) -> b s', b = batch_size)                   # [batch_size, seq_len]

        f1 = f1_score(y_pred = predicted_events, y_true = input_events, average = 'macro')

        return gap, f1


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
        (input_events, input_time, padded_score, input_mask, input_intensity), mean_and_var  = minibatch
        mean, var = 0, 1
        if mean_and_var is not None:
            mean, var = mean_and_var

        return input_time, input_events, input_mask, input_intensity, mean, var


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
        return NotImplementedError('LogNormMix is intensity-free. Therefore, it can not provide the plot for the intensity integral.')


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

        input_time, input_events, input_mask, input_intensity, mean, var = self.extract_plot_data(input_data)

        batch_size, _ = input_time.shape
        input_time_for_generating_reference = torch.cat((torch.zeros(batch_size, 1, device = self.device), input_time[:, :-1]), dim = -1)
        input_events_for_generating_reference = torch.cat((torch.ones(batch_size, 1, device = self.device, dtype = torch.int) * self.num_events, input_events[:, :-1]), dim = -1)
        input_mask_for_generating_reference = torch.cat((torch.ones(batch_size, 1, device = self.device, dtype = torch.int), input_mask[:, :-1]), dim = -1)

        time_history, time_next = self.divide_history_and_next(input_time_for_generating_reference)
                                                                               # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events_for_generating_reference)
                                                                               # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(input_mask_for_generating_reference)
                                                                               # [batch_size, seq_len]

        expand_probability, timestamp = \
            self.model.probability_prober(input_events, input_time, input_mask, opt.resolution, mean, var)
                                                                               # [batch_size, seq_len, resolution] * 2

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

        input_time, input_events, input_mask, input_intensity, mean, var = self.extract_plot_data(input_data)

        time_history, time_next = self.divide_history_and_next(input_time)     # [batch_size, seq_len]
        events_history, events_next = self.divide_history_and_next(input_events)
                                                                               # [batch_size, seq_len]
        mask_history, mask_next = self.divide_history_and_next(input_mask)     # [batch_size, seq_len]

        mae, f1_1 = self.mean_absolute_error_and_f1(input_events, input_time, input_mask, mean, var)
                                                                               # [batch_size, seq_len]
        
        _, timestamp = \
            self.model.probability_prober(input_events, input_time, input_mask, opt.resolution, mean, var)
                                                                               # [batch_size, seq_len, resolution] * 2
        data = {}
        '''
        Append additional info into the data dict.
        '''
        data['events_next'] = events_next
        data['time_next'] = time_next
        data['mask_next'] = mask_next
        data['f1_after_time_pred'] = f1_1
        data['mae_before_event'] = mae

        plots = plot_debug(data, timestamp, opt)

        return plots


    '''
    Evaluation over the entire dataset.
    '''
    def get_spearman_and_l1(self, input_data, opt):
        input_time, input_events, input_mask, input_intensity, mean, var = self.extract_plot_data(input_data)
                                                                               # [batch_size, seq_len + 1] * 4 + float + float
        expand_probability, timestamp = \
            self.model.probability_prober(input_events, input_time, input_mask, opt.resolution, mean, var)
                                                                               # [batch_size, seq_len, resolution] * 2
        true_probability = expand_true_probability(input_time[:, :-1], input_intensity, opt)
                                                                               # [batch_size, seq_len, resolution] or batch_size * None
        
        expand_probability, true_probability, timestamp = move_from_tensor_to_ndarray(expand_probability, true_probability, timestamp)
        zipped_data = zip(expand_probability, true_probability, timestamp, input_mask)

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

        batch_size = input_mask.shape[0]
        spearman /= batch_size
        l1 /= batch_size

        return spearman, l1
    

    def get_mae_and_f1(self, input_data, opt):
        input_time, input_events, input_mask, input_intensity, mean, var = self.extract_plot_data(input_data)

        mae, f1_1 = self.mean_absolute_error_and_f1(input_events, input_time, input_mask, mean, var)
                                                                               # [batch_size, seq_len]
        mae = move_from_tensor_to_ndarray(mae)

        return mae, f1_1

    
    def get_mae_e_and_f1(self, input_data, opt):

        return NotImplementedError('Event-time evaluation is unavailable because LognormMix is a TPP model.')


    def train_step(model, minibatch, device):
        ''' Epoch operation in training phase'''
    
        def extract_minibatch(minibatch):
            (input_events, input_time, _, input_mask), mean_and_var = minibatch
            mean, var = 0, 1
            if mean_and_var is not None:
                mean, var = mean_and_var
            return {'input_events': input_events, 'input_time': input_time, 'input_mask': input_mask, 'mean': mean, 'var': var}

        model.train()
        time_loss, time_loss_without_dummy, events_loss, the_number_of_events\
              = model(task_name = 'train', **extract_minibatch(minibatch))

        loss = time_loss + events_loss
        loss.backward()
    
        time_loss_without_dummy = time_loss_without_dummy.item() / the_number_of_events
        events_loss = events_loss.item() / the_number_of_events
        fact = minibatch[0][2].sum().item() / the_number_of_events
    
        return time_loss_without_dummy, events_loss, fact
    

    def evaluation_step(model, minibatch, device):
        ''' Epoch operation in evaluation phase '''
    
        def extract_minibatch(minibatch):
            (input_events, input_time, _, input_mask), mean_and_var = minibatch
            mean, var = 0, 1
            if mean_and_var is not None:
                mean, var = mean_and_var
            return {'input_events': input_events, 'input_time': input_time, 'input_mask': input_mask, 'mean': mean, 'var': var}

        model.eval()
        time_loss, surv_last_loss, event_loss, mae, f1_pred_time, the_number_of_events \
            = model(task_name = 'evaluate', **extract_minibatch(minibatch))

        time_loss = time_loss.item() / the_number_of_events
        surv_last_loss = surv_last_loss.item() / the_number_of_events
        event_loss = event_loss.item() / the_number_of_events
        fact = minibatch[0][2].sum().item() / the_number_of_events
    
        return time_loss, surv_last_loss, fact, event_loss, mae, f1_pred_time,


    def postprocess(input, procedure):
        def train_postprocess(input):
            '''
            Training process
            [absolute loss, relative loss, events loss]
            '''
            return [input[0], input[0] - input[2], input[1]]
        
        def test_postprocess(input):
            '''
            Evaluation process
            '''
            return [input[0], input[1], input[0] - input[2], input[3], input[4], input[5]]
        
        return train_postprocess(input) if procedure == 'Training' else test_postprocess(input)


    format_dict_length = 6


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
            format_dict['mae'] = input[4]
            format_dict['f1_pred_at_pred_time'] = input[5]
            format_dict['num_format'] = {'absolute_NLL_loss': ':6.5f', 'avg_survival_loss': ':6.5f', \
                                         'relative_NLL_loss': ':6.5f', 'events_loss': ':6.5f',
                                         'mae': ':2.8f', 'f1_pred_at_pred_time': ':6.5f'}
            return format_dict
        
        return (train_log_print_format(input) if procedure == 'Training' else test_log_print_format(input))


    def choose_metric(evaluation_report_format_dict, test_report_format_dict):
        '''
        [relative loss on evaluation dataset, relative loss on test dataset, event loss on test dataset]
        '''
        # return [evaluation_report_format_dict['absolute_NLL_loss'] + evaluation_report_format_dict['avg_survival_loss'], 
        #         test_report_format_dict['absolute_NLL_loss'] + test_report_format_dict['avg_survival_loss']], \
        #        ['evaluation_absolute_loss', 'test_absolute_loss']
        return [evaluation_report_format_dict['absolute_NLL_loss'], 
                test_report_format_dict['absolute_NLL_loss']], \
               ['evaluation_absolute_loss', 'test_absolute_loss']

    metric_number = 2 # metric number is the length of the output of choose_metric
