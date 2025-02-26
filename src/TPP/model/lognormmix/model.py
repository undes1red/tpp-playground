import torch, copy
from sklearn.metrics import f1_score
from einops import rearrange, reduce, repeat

from src.toolbox.misc import move_from_tensor_to_ndarray, conditional_decorator, pack_one_value_to_dict

from src.TPP.model.lognormmix.log_norm_mix import LogNormMix
from src.TPP.model.lognormmix.plot import *
from src.TPP.model.basic_tpp_model import BasicModel
from src.TPP.model.lognormmix.sample import sample_time


class LogNormMixWrapper(BasicModel):
    def __init__(self, opt, device, context_size: int = 32, mark_embedding_size: int = 32, \
                 num_mix_components: int = 16, rnn_type: str = "LSTM", \
                 survival_loss_during_training = True):
        super(LogNormMixWrapper, self).__init__()
        self.device = device
        self.compile_or_not = opt.compile
        self.num_events = opt.info_dict['num_events']
        self.survival_loss_during_training = survival_loss_during_training
        self.sample_rate = 32
        self.max_step = 50
        self.bisect_early_stop_threshold = 1e-4

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

            # Figure Drawing.
            'intensity': self.figure_intensity,
            'integral': self.figure_integral,
            'probability': self.figure_probability,
            'debug': self.figure_debug
        }

        return task_mapper[task_name](*args, **kwargs)


    def divide_history_and_next(self, input):
        history, next = input[:, :-1].clone(), input[:, 1:].clone()
        return history, next                                                   # [batch_size, seq_len, 1] or [batch_size, seq_len]


    def train_procedure(self, input_events, input_time, input_mask, mean, std):
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
                std
            ](if self.input_norm_data is True)
        ]
        '''
        the_number_of_events = input_mask.sum().item()
        log_prob, log_surv_last, log_p_event = self.model.log_prob(input_events, input_time, input_mask, mean, std)
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


    @torch.inference_mode()
    def evaluate_procedure(self, input_events, input_time, input_mask, mean, std):
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
                std
            ](if self.input_norm_data is True)
        ]
        '''
        the_number_of_events = input_mask.sum().item()
        log_prob, log_surv_last, log_p_event = self.model.log_prob(input_events, input_time, input_mask, mean, std)
                                                                               # [batch_size, seq_len + 1]
        log_prob = log_prob * input_mask                                       # [batch_size, seq_len + 1]
        log_p_event = log_p_event * input_mask                                 # [batch_size, seq_len + 1]
        
        time_loss = self.loss_f(log_prob)
        event_loss = self.loss_f(log_p_event)
        surv_last_loss = self.loss_f(log_surv_last)

        mae, f1 = self.mean_absolute_error_and_f1(input_events, input_time, input_mask, mean, std)
                                                                               # [batch_size, seq_len + 1]
        mae = mae.sum().item() / the_number_of_events
        
        return time_loss, surv_last_loss, event_loss, mae, f1, the_number_of_events


    def loss_f(self, loglik):
        '''
        The definition of loss.
        '''
        return (-loglik).sum()
    

    def sample_time(self, *args, **kwargs):
        return conditional_decorator(torch.compile, self.compile_or_not, sample_time)(self, *args, **kwargs)


    @torch.inference_mode()
    def mean_absolute_error_and_f1(self, input_events, input_time, input_mask, mean, std):
        # Obtain dedicated MAE and predicted time.
        tau_pred = self.sample_time('its', 'tm', input_events, input_time, input_mask, mean, std)
                                                                               # [sample_rate, batch_size, seq_len + 1]
        tau_pred = tau_pred.mean(dim = 0)                                      # [batch_size, seq_len]
        mae = torch.abs(tau_pred - input_time) * input_mask                    # [batch_size, seq_len]

        predicted_events  = self.model.event_prober(input_events, input_time, input_mask, mean, std)
                                                                               # [batch_size, seq_len + 1]
        predicted_events = predicted_events[input_mask == 1]                   # [batch_size * seq_len]
        input_events = input_events[input_mask == 1]                           # [batch_size * seq_len]
        predicted_events, input_events = move_from_tensor_to_ndarray(predicted_events, input_events)
        f1 = f1_score(y_pred = predicted_events, y_true = input_events, average = 'macro')

        return mae, f1


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
        (input_events, input_time, padded_score, input_mask, input_intensity), mean_and_std  = minibatch
        mean, std = 0, 1
        if mean_and_std is not None:
            mean, std = mean_and_std

        return input_time, input_events, input_mask, input_intensity, mean, std


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
        return NotImplementedError('LogNormMix is intensity-free. Therefore, it can not provide the plot for the intensity integral.')


    @torch.inference_mode()
    def figure_probability(self, input_data, opt):
        '''
        Function prober, used by tpp_ploter to draw plots.

        Args:
        * input_data  type: list shape: [[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]
                      The original minibatch. Detailed information is available in extract_plot_data()
        * resolution  type: int shape: N/A
                      How many interpretive numbers we have between an event interval?
        '''
        input_time, input_events, input_mask, input_intensity, mean, std = self.extract_plot_data(input_data)

        batch_size, _ = input_time.shape
        input_time_for_generating_reference = torch.cat((torch.zeros(batch_size, 1, device = self.device), input_time[:, :-1]), dim = -1)
        input_events_for_generating_reference = torch.cat((torch.ones(batch_size, 1, device = self.device, dtype = torch.int) * self.num_events, input_events[:, :-1]), dim = -1)
        input_mask_for_generating_reference = torch.cat((torch.ones(batch_size, 1, device = self.device, dtype = torch.int), input_mask[:, :-1]), dim = -1)

        _, time_next = self.divide_history_and_next(input_time_for_generating_reference)
                                                                               # [batch_size, seq_len]
        _, events_next = self.divide_history_and_next(input_events_for_generating_reference)
                                                                               # [batch_size, seq_len]
        _, mask_next = self.divide_history_and_next(input_mask_for_generating_reference)
                                                                               # [batch_size, seq_len]
        expand_probability, timestamp = \
            self.model.probability_prober(input_events, input_time, input_mask, opt.resolution, mean, std)
                                                                               # [batch_size, seq_len, resolution] * 2
        data = {
            'time_next': time_next,
            'events_next': events_next,
            'mask_next': mask_next,
            'expand_probability': expand_probability,
            'input_intensity': input_intensity
            }
        plots = generate_probability_figure(data, timestamp, opt)
        return plots


    @torch.inference_mode()
    def figure_debug(self, input_data, opt):
        '''
        Args:
        time: [batch_size(always 1), seq_len + 1]
              The original dataset records. 
        resolution: int
              How many interpretive numbers we have between an event interval?
        '''
        input_time, input_events, input_mask, input_intensity, mean, std = self.extract_plot_data(input_data)

        _, mask_next = self.divide_history_and_next(input_mask)                # [batch_size, seq_len]

        mae, _ = self.mean_absolute_error_and_f1(input_events, input_time, input_mask, mean, std)
                                                                               # [batch_size, seq_len]
        data = {}
        '''
        Append additional info into the data dict.
        '''
        data['mask_next'] = mask_next
        data['mae_before_event'] = mae

        plots = generate_debug_figure(data, None, opt)

        return plots


    '''
    Evaluation over the entire dataset.
    '''
    @torch.inference_mode()
    def get_spearman_and_l1(self, input_data, opt):
        input_time, input_events, input_mask, input_intensity, mean, std = self.extract_plot_data(input_data)
                                                                               # [batch_size, seq_len + 1] * 4 + float + float
        expand_probability, timestamp = \
            self.model.probability_prober(input_events, input_time, input_mask, opt.resolution, mean, std)
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

            l1_per_seq = L1_distance_between_two_funcs(x = true_probability_per_seq[:seq_len, :], y = expand_probability_per_seq[:seq_len, :], \
                                                       timestamp = timestamp_per_seq)
            spearman += spearman_per_seq
            l1 += l1_per_seq

        batch_size = input_mask.shape[0]
        spearman /= batch_size
        l1 /= batch_size

        return spearman, l1
    

    @torch.inference_mode()
    def get_mae_and_f1(self, input_data, opt):
        input_time, input_events, input_mask, input_intensity, mean, std = self.extract_plot_data(input_data)

        mae, f1_1 = self.mean_absolute_error_and_f1(input_events, input_time, input_mask, mean, std)
                                                                               # [batch_size, seq_len]
        mae = move_from_tensor_to_ndarray(mae)

        return mae, f1_1

    
    @torch.inference_mode()
    def get_mae_e_and_f1(self, input_data, opt):
        raise NotImplemented("get_mae_e_and_f1() not implemented for vanilla RMTPP because it is a TPP model.")


    @torch.inference_mode()
    def get_which_event_first(self, input_data, opt):
        return NotImplemented('get_which_event_first() not implemented for vanilla RMTPP because it is a TPP model.')


    @torch.inference_mode()
    def samples_from_et(self, input_data, opt):
        return NotImplemented('samples_from_et() not implemented for vanilla RMTPP because it is a TPP model.')


    def train_step(model, minibatch, device):
        ''' Epoch operation in training phase'''
        def extract_minibatch(minibatch):
            (input_events, input_time, _, input_mask), mean_and_std = minibatch
            mean, std = 0, 1
            if mean_and_std is not None:
                mean, std = mean_and_std
            return {'input_events': input_events, 'input_time': input_time, 'input_mask': input_mask, 'mean': mean, 'std': std}

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
            (input_events, input_time, _, input_mask), mean_and_std = minibatch
            mean, std = 0, 1
            if mean_and_std is not None:
                mean, std = mean_and_std
            return {'input_events': input_events, 'input_time': input_time, 'input_mask': input_mask, 'mean': mean, 'std': std}

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


    def choose_metric(evaluation_report_format_dict, test_report_format_dict):
        '''
        [relative loss on evaluation dataset, relative loss on test dataset, event loss on test dataset]
        '''
        return [evaluation_report_format_dict['absolute_NLL_loss'], 
                test_report_format_dict['absolute_NLL_loss']], \
               ['evaluation_absolute_loss', 'test_absolute_loss']


    metric_number = 2 # metric number is the length of the output of choose_metric
