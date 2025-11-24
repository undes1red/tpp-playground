import torch, copy
from sklearn.metrics import f1_score
from einops import rearrange, reduce, repeat

from src.toolbox.misc import move_from_tensor_to_ndarray, pack_one_value_to_dict, argument_check

from src.TPP.model.lognormmix.log_norm_mix import LogNormMix
from src.TPP.model.lognormmix.plot import *
from src.TPP.model.basic_tpp_model import BasicModel
from src.TPP.model.lognormmix.sample import sample_time


class LogNormMixWrapper(BasicModel):
    '''
    The original LogNormmix by Shchur et al. @ ICLR 2020.
    '''
    def __init__(self, opt, device, context_size: int = 32, mark_embedding_size: int = 32, \
                 num_mix_components: int = 16, rnn_type: str = "LSTM", \
                 sample_rate = 32, mae_step = 16, mae_e_step = 16, \
                 survival_loss_during_training = True):
        '''
        This function creates a LogNormMix model.
        
        ### Args
            * ```int``` context_size
              The dimension of the history embedding.
            * ```int``` mark_embedding_size
              The dimension of the mark embedding.
            * ```int``` num_mix_components
              How many log-norm distribution are they in a LogNormMix?
            * ```str``` rnn_type
              The structure of the RNN module. Defualt: LSTM.
            * ```namespace``` opt
              Model arguments.
            * ```torch.device``` device
              Running models on GPU or CPU?
            * ```int``` sample_rate
              This tells how many time samples from the time distribution are needed for one time prediction.
            * ```int``` mae_step
              This parameter controls how many samples are generated in one shot when sampling from p(t).
            * ```int``` mae_e_step
              This parameter controls how many samples are generated in one shot when sampling from all p(t|m)s at the same time.
              mae_step and mae_e_step are useful when you cannot get sample_rate time samples from time distributions because of insufficient GPU memory.
            * ```bool``` survival_loss_during_training
              When true, the training loss includes the integral between the last observed event to the end time T. Most of time this argument should be true.
        '''
        super(LogNormMixWrapper, self).__init__()
        self.device = device
        self.compile_or_not = opt.compile
        self.num_events = opt.info_dict['num_events']
        self.survival_loss_during_training = survival_loss_during_training
        self.sample_rate = sample_rate
        self.mae_step = mae_step
        self.mae_e_step = mae_e_step
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
        The entrance of the MarkedLogNormMix.
        
        ### Args
            * ```str``` task_name
              The name of the executed task.
        '''
        task_mapper = {
            'train': self.train_procedure,
            'evaluate': self.evaluate_procedure,
            'spearman_and_l1': self.get_spearman_and_l1,
            'mae_and_f1': self.get_mae_and_f1,
            'mae_e_and_f1': self.get_mae_e_and_f1,
            'which_mark_occurs_first': self.get_which_event_first,
            'samples_from_et': self.samples_from_et,

            # Figure Drawing.
            'intensity': self.figure_intensity,
            'integral': self.figure_integral,
            'probability': self.figure_probability,
            'debug': self.figure_debug
        }

        return task_mapper[task_name](*args, **kwargs)


    def divide_history_and_next(self, input):
        '''
        Extract the history and prediction sequences from the input sequence.
        
        ### Args
            * ```torch.tensor``` input
              shape: [batch_size, seq_len + 1]
              The input sequence.
        
        ### Outputs
            * ```torch.tensor``` input_history
              shape: [batch_size, seq_len]
              The history sequence extracted from the original input.
            * ```torch.tensor``` input_next
              shape: [batch_size, seq_len]
              The history sequence extracted from the original input.
        '''
        history, next = input[:, :-1].clone(), input[:, 1:].clone()
        return history, next                                                   # [batch_size, seq_len, 1] or [batch_size, seq_len]


    def train_procedure(self, input_events, input_time, input_mask, mean, std):
        '''
        LogNormMix's forwardpropagation function for training.
        
        ### Args
            * ```torch.tensor``` input_time
              shape: ```[batch_size, seq_len + 1]```
              Time sequence for training.
            * ```torch.tensor``` input_events
              shape: ```[batch_size, seq_len + 1]```
              Event sequence for training.
            * ```torch.tensor``` input_mask
              shape: ```[batch_size,, seq_len + 1]```
              Mask sequence. Events whose corresponding mask is 0 are dummy events.
            * ```float``` mean
            * ```float``` std
              Used for input time scaling.

        ### Outputs
            * ```torch.tensor``` loss
              shape: ```[1]```
              The sum of NLL loss L = -log \\frac{\\partial \\Lambda^*(m, t)}{\\partial t} + \\Lambda^*(m, t) at each happened event (the dummy event at end time T included).
            * ```torch.tensor``` time_loss
              shape: ```[1]```
              The sum of NLL loss L = -log \\frac{\\partial \\Lambda^*(m, t)}{\\partial t} + \\Lambda^*(m, t) at each happened event (the dummy event at end time T excluded).
            * ```int``` the_number_of_events
              The number of legit events.
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
        LogNormMix's forwardpropagation function for evaluation.
        
        ### Args
            * ```torch.tensor``` input_time
              shape: ```[batch_size, seq_len + 1]```
              Time sequencalculatesce for training.
            * ```torch.tensor``` input_events
              shape: ```[batch_size, seq_len + 1]```
              Event sequence for training.
            * ```torch.tensor``` input_mask
              shape: ```[batch_size,, seq_len + 1]```
              Mask sequence. Events whose corresponding mask is 0 are dummy events.
            * ```float``` mean
            * ```float``` std
              Used for input time scaling.

        ### Outputs
            * ```torch.tensor``` time_loss
              shape: ```[1]```
              The sum of NLL loss L = -log \\frac{\\partial \\Lambda^*(m, t)}{\\partial t} + \\Lambda^*(m, t) at each happened event.
            * ```torch.tensor``` surv_last_loss
              shape: ```[1]```
              The sum of the integration \\Lambda^*(m, t) from the last observed event to the end time T.
            * ```float``` mae
              The average error between predicted time and real time.
            * ```float``` f1
              The prediction accuracy of predicted marks.
            * ```int``` the_number_of_events
              The number of legit events.
        '''
        the_number_of_events = input_mask.sum().item()
        log_prob, log_surv_last, log_p_event = self.model.log_prob(input_events, input_time, input_mask, mean, std)
                                                                               # [batch_size, seq_len + 1]
        log_prob = log_prob * input_mask                                       # [batch_size, seq_len + 1]
        log_p_event = log_p_event * input_mask                                 # [batch_size, seq_len + 1]
        
        time_loss = self.loss_f(log_prob)
        event_loss = self.loss_f(log_p_event)
        surv_last_loss = self.loss_f(log_surv_last)

        mae, f1, _ = self.mean_absolute_error_and_f1(input_events, input_time, input_mask, mean, std)
                                                                               # [batch_size, seq_len + 1]
        mae = mae.sum().item() / the_number_of_events
        
        return time_loss, surv_last_loss, event_loss, mae, f1, the_number_of_events


    def loss_f(self, loglik):
        '''
        This function computes the NLL loss at each event in events_next.
        Please note that this function does not care if the corresponding event is real or dummy.
        They should be handled before calling this loss funtion.
    
        ### Args
            * ```torch.tensor``` loglik
              shape: ```[batch_size, seq_len, num_events]```
              values of the p(m = m_k, t) at t_i.
        
        ### Outputs
            * ```torch.tensor``` loss
              shape: ```[1]```
              The sum of NLL loss on all event.
        '''
        return (-loglik).sum()
    

    sample_time = sample_time


    @torch.inference_mode()
    def mean_absolute_error_and_f1(self, input_events, input_time, input_mask, mean, std):
        '''
        Called by evaluate_procedure(), debug() and get_mae_and_f1(), this function computed the MAE and macro-F1 of one minibatch.

        ### Args
            * ```torch.tensor``` input_events
              shape: ```[batch_size, seq_len + 1]```
            * ```torch.tensor``` input_time
              shape: ```[batch_size, seq_len + 1]```
            * ```torch.tensor``` input_mask
              shape: ```[batch_size, seq_len + 1]```
            * ```float``` mean
            * ```float``` std
              Used for input time scaling.

        ### Outputs
            * ```torch.tensor``` mae
              shape: ```[batch_size, seq_len]```
              Mean Absolute Error(MAE) between predicted times \\(t_p\\) and ground truths \\(t_i\\). MAE = |t_p - t_i|.
            * ```float``` f1
              macro-F1 value between events predicted at \\(t_p\\) and the ground truths.
        '''
        tau_pred = self.sample_time('its', 'tm', input_events, input_time, input_mask, mean, std)
                                                                               # [sample_rate, batch_size, seq_len + 1]
        tau_pred = tau_pred.mean(dim = 0)                                      # [batch_size, seq_len + 1]
        mae = torch.abs(tau_pred - input_time) * input_mask                    # [batch_size, seq_len + 1]

        predicted_events, dist = self.model.event_prober(input_events, input_time, input_mask, mean, std)
                                                                               # [batch_size, seq_len + 1]
        predicted_events = predicted_events[input_mask == 1]                   # [batch_size * seq_len]
        input_events = input_events[input_mask == 1]                           # [batch_size * seq_len]
        predicted_events, input_events = move_from_tensor_to_ndarray(predicted_events, input_events)
        f1 = f1_score(y_pred = predicted_events, y_true = input_events, average = 'macro')

        return mae, f1, dist


    def extract_plot_data(self, minibatch):
        '''
        This function extracts input_time, input_events, input_intensity, mask, mean, and std from the minibatch.

        ### Args
            * ```list``` minibatch
              shape: ```[[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]```
              data structure: [[input_time, input_events, score, mask], (mean, std)]
              data type: [```torch.tensor```, ```torch.tensor```, ```torch.tensor```, ```torch.tensor```, (```float```, ```float```)]
              The input minibatch.
              
        ### Outputs
            * ```torch.tensor``` input_time
              shape: ```[batch_size, seq_len + 1]```
              Raw event timestamp sequence.
            * ```torch.tensor``` input_events
              shape: ```[batch_size, seq_len + 1]```
              Raw event marks sequence.
            * ```torch.tensor``` input_mask
              shape: ```[batch_size, seq_len + 1]```
              Raw mask sequence.
            * ```torch.tensor``` input_intensity
              shape: ```[batch_size, seq_len + 1]```
              The true intensity function if available. It will be zero tensors if the true intensity if unavailable.
            * ```int``` mean
              The mean of all $ t_i - t_{i - 1} $ in the entire dataset. Dataloader is responsible to provide this value if needed.
            * ```int``` std
              The mean of all $ t_i - t_{i - 1} $ in the entire dataset. Dataloader is responsible to provide this value if needed.
        '''
        (input_events, input_time, padded_score, input_mask, input_intensity), mean_and_std  = minibatch
        mean, std = 0, 1
        if mean_and_std is not None:
            mean, std = mean_and_std

        return input_time, input_events, input_mask, input_intensity, mean, std


    def figure_intensity(self, input_data, opt):
        '''
        Function prober, used by evaluator to draw plots of the intensity function.
        Not implemented for MarkedLogNormMix since this model does not provide the intensity function.

        ### Args
            * ```list``` input_data
              shape: ```[[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]```
              data structure: [[input_time, input_events, score, mask], (mean, std)]
              data type: [```torch.tensor```, ```torch.tensor```, ```torch.tensor```, ```torch.tensor```, (```float```, ```float```)]
              The input minibatch.
            * ```namespace``` opt
              plot and model configs
        '''
        return NotImplementedError('LogNormMix is intensity-free. Therefore, it can not provide the plot for the intensity function.')


    def figure_integral(self, input_data, opt):
        '''
        Function prober, used by evaluator to draw plots of integral of the intensity function.
        Not implemented for MarkedLogNormMix since this model does not provide the intensity function.

        ### Args
            * ```list``` input_data
              shape: ```[[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]```
              data structure: [[input_time, input_events, score, mask], (mean, std)]
              data type: [```torch.tensor```, ```torch.tensor```, ```torch.tensor```, ```torch.tensor```, (```float```, ```float```)]
              The input minibatch.
            * ```namespace``` opt
              plot and model configs
        '''
        return NotImplementedError('LogNormMix is intensity-free. Therefore, it can not provide the plot for the intensity integral.')


    @torch.inference_mode()
    def figure_probability(self, input_data, opt):
        '''
        Function prober, used by evaluator to draw plots of the probability distribution.
        
        You should declare the following arguments in your config file:
        1. ```int``` resolution: The number of interpolated points in a time interval between two adjoint events for integration estimation.
                                 The number of interpolated points counts the start and end point of the interval.
        
        ### Args
            * ```list``` input_data
              shape: ```[[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]```
              data structure: [[input_time, input_events, score, mask], (mean, std)]
              data type: [```torch.tensor```, ```torch.tensor```, ```torch.tensor```, ```torch.tensor```, (```float```, ```float```)]
              The input minibatch.
            * ```namespace``` opt
              plot and model configs
        '''
        argument_check(opt, **{'resolution': int})
        
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
            'input_intensity': input_intensity,
            'timestamp': timestamp
        }
        plots = generate_probability_figure(data, opt)
        return plots


    @torch.inference_mode()
    def figure_debug(self, input_data, opt):
        '''
        Function prober, used by evaluator to draw plots for deeper insight of the learned distribution and other metrics.
        
        You should declare the following arguments in your config file:
        1. ```int``` resolution: The number of interpolated points in a time interval between two adjoint events for integration estimation.
                                 The number of interpolated points counts the start and end point of the interval.
        2. ```int``` sample_rate: The number of interpolated points in a time interval between two adjoint events for integration estimation.
                                  The number of interpolated points counts the start and end point of the interval.
        3. ```int``` mae_step: This parameter controls how many samples are generated in one shot when sampling from p(t).
        4. ```int``` mae_e_step: This parameter controls how many samples are generated in one shot when sampling from all p(t|m)s at the same time.
        
        ### Args
            * ```list``` input_data
              shape: ```[[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]```
              data structure: [[input_time, input_events, score, mask], (mean, std)]
              data type: [```torch.tensor```, ```torch.tensor```, ```torch.tensor```, ```torch.tensor```, (```float```, ```float```)]
              The input minibatch.
            * ```namespace``` opt
              plot and model configs
        '''
        argument_check(opt, **{'resolution': int, 'sample_rate': int, 'mae_step': int, 'mae_e_step': int})
        
        input_time, input_events, input_mask, input_intensity, mean, std = self.extract_plot_data(input_data)

        _, mask_next = self.divide_history_and_next(input_mask)                # [batch_size, seq_len]

        mae, _, _ = self.mean_absolute_error_and_f1(input_events, input_time, input_mask, mean, std)
                                                                               # [batch_size, seq_len + 1]
        mae = mae[..., :-1]                                                    # [batch_size, seq_len]
        
        data = {}
        
        # Append additional info into the data dict.
        data = {
            'mask_next': mask_next,
            'mae': mae
        }

        generate_debug_figure(data, opt)


    # Evaluation over the entire dataset.
    @torch.inference_mode()
    def get_spearman_and_l1(self, input_data, opt):
        '''
        Used by evaluator to calculate the average gap between the predicted and real distribution using L1 distance and spearman coefficient.
        
        You should declare the following arguments in your config file:
        1. ```int``` resolution: The number of interpolated points in a time interval between two adjoint events for integration estimation.
                                 The number of interpolated points counts the start and end point of the interval.
        
        ### Args
            * ```list``` input_data
              shape: ```[[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]```
              data structure: [[input_time, input_events, score, mask], (mean, std)]
              data type: [```torch.tensor```, ```torch.tensor```, ```torch.tensor```, ```torch.tensor```, (```float```, ```float```)]
              The input minibatch.
            * ```namespace``` opt
              plot and model configs
        
        ### Outputs:
            * ```float``` spearman
              The spearman coefficient between the predicted and real distribution.
            * ```float``` l1
              The l1 distance between the predicted and real distribution.
        '''
        argument_check(opt, **{'resolution', int})
        
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
        '''
        Used by evaluator to evaluate the performance of predicted time from p(t) and mark from p(m|t).
        
        You should declare the following arguments in your config file:
        1. ```int``` sample_rate: how many time samples from the time distribution are needed.
        2. ```int``` mae_step: This parameter controls how many samples are generated in one shot when sampling from p(t).
        
        ### Args
            * ```list``` input_data
              shape: ```[[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]```
              data structure: [[input_time, input_events, score, mask], (mean, std)]
              data type: [```torch.tensor```, ```torch.tensor```, ```torch.tensor```, ```torch.tensor```, (```float```, ```float```)]
              The input minibatch.
            * ```namespace``` opt
              plot and model configs
        
        ### Outputs:
            * ```np.ndarray``` mae
              shape: ```[batch_size, seq_len]```
              The MAE value, which is the time gap between each predicted and real event.
            * ```float``` f1_1
              The f1 value shows the accuracy of the predicted marks.
            * ```torch.tensor``` dist
              shape: ```[batch_size, seq_len, num_events]```
              The mark distribution at when the real event happens.
            * ```np.ndarray``` events_next
              shape: ```[batch_size, seq_len]```
              Real marks of observed events.
        '''
        argument_check(opt, **{'sample_rate': int, 'mae_step': int})
        
        input_time, input_events, input_mask, input_intensity, mean, std = self.extract_plot_data(input_data)

        mae, f1_1, dist = self.mean_absolute_error_and_f1(input_events, input_time, input_mask, mean, std)
                                                                               # [batch_size, seq_len]
        mae, dist, input_events = move_from_tensor_to_ndarray(mae, dist, input_events)

        return mae[..., :-1], f1_1, dist, input_events[..., :-1]

    
    @torch.inference_mode()
    def get_mae_e_and_f1(self, input_data, opt):
        raise NotImplemented("get_mae_e_and_f1() not implemented for LogNormMix.")


    @torch.inference_mode()
    def get_which_event_first(self, input_data, opt):
        return NotImplemented('get_which_event_first() not implemented for LogNormMix.')


    @torch.inference_mode()
    def samples_from_et(self, input_data, opt):
        return NotImplemented('samples_from_et() not implemented for LogNormMix.')


    def train_step(model, minibatch, device):
        '''
        This function unpacks the minibatch, calls the train_procedure() to calculate the loss, and do the backpropagation.

        ### Args
            * ```torch.nn.Module``` model
              The MTPP model that we train.
            * ```list``` minibatch
              shape: ```[[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]```
              data structure: [[input_time, input_events, score, mask], (mean, std)]
              data type: [```torch.tensor```, ```torch.tensor```, ```torch.tensor```, ```torch.tensor```, (```float```, ```float```)]
              The input minibatch.
            * ```torch.device``` device
              where we train the model.

        ### Outputs:
            * ```float``` time_loss_without_dummy
              The average NLL loss without dummy events, specifically the start and the end event.
            * ```float``` events_loss
              The average cross-entropy loss of the event prediction distribution. The value is only for performance measure porpose.
              The training loss does not and should not include this value.
            * ```float``` fact
              The average NLL loss with the real distribution. This value only makes sense for synthetic datasets.
        '''
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
        '''
        This function unpacks the minibatch, calls the evaluation_procedure() to calculate the metrics.

        ### Args
            * ```torch.nn.Module``` model
              The MTPP model that we train.
            * ```list``` minibatch
              shape: ```[[batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], [batch_size, seq_len + 1], (int, int)]```
              data structure: [[input_time, input_events, score, mask], (mean, std)]
              data type: [```torch.tensor```, ```torch.tensor```, ```torch.tensor```, ```torch.tensor```, (```float```, ```float```)]
              The input minibatch.
            * ```torch.device``` device
              where we train the model.

        ### Outputs:
            * ```float``` time_loss
              The average NLL loss without dummy events, specifically the start and the end event.
            * ```float``` surv_last_loss
              The average NLL loss of the end event, which is the integral of the intensity function from the last occurred event to the end time.
            * ```float``` fact
              The average NLL loss with the real distribution. This value only makes sense for synthetic datasets.
            * ```float``` events_loss
              The average cross-entropy loss of the event prediction distribution. The value is only for performance measure porpose.
            * ```float``` mae
              The average error between predicted time and real time.
            * ```float``` f1
              The prediction accuracy of predicted marks.
        '''
        def extract_minibatch(minibatch):
            (input_events, input_time, _, input_mask), mean_and_std = minibatch
            mean, std = 0, 1
            if mean_and_std is not None:
                mean, std = mean_and_std
            return {'input_events': input_events, 'input_time': input_time, 'input_mask': input_mask, 'mean': mean, 'std': std}

        model.eval()

        time_loss, surv_last_loss, event_loss, mae, f1, the_number_of_events \
            = model(task_name = 'evaluate', **extract_minibatch(minibatch))

        time_loss = time_loss.item() / the_number_of_events
        surv_last_loss = surv_last_loss.item() / the_number_of_events
        event_loss = event_loss.item() / the_number_of_events
        fact = minibatch[0][2].sum().item() / the_number_of_events
    
        return time_loss, surv_last_loss, fact, event_loss, mae, f1


    def postprocess(input, procedure):
        '''
        This function makes some modifications to the output of training_step() and evaluation_step().

        ### Args
            * ```list``` input
              The output of either training_step() or evaluation_step().
            * ```str``` procedure
              This string tells the function which function the input comes from.

        ### Outputs:
            * ```list```
              The postprocessed outputs.
        '''
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


    def log_print_format(input, procedure):
        '''
        This function packs the procedure input into a dict that can be handled by trainer and evaluator for logging.

        ### Args
            * ```list``` input
              The output of either training_step() or evaluation_step().
            * ```str``` procedure
              This string tells the function which function the input comes from.

        ### Outputs:
            * ```dict``` format_dict
              format: {..., <variable name>: {'data': <value>, 'num_format': <num_format>, 'suffix': <suffix>}, ...}
              example: {..., 'memory': {'data': 12.123456, 'num_format': ':2.4f', 'suffix': 'GiB'}, ...}
              The formated results.
        '''
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


    '''
    The maximum length of the format_dict in different procedures.
    '''
    format_dict_length = 6


    def choose_metric(evaluation_report_format_dict, test_report_format_dict):
        '''
        This function helps the trainer to pick the best checkpoint based on several metrics.

        ### Args
            * ```dict``` evaluation_report_format_dict
            * ```dict``` test_report_format_dict
              The formated output of training_step() and evaluation_step().

        ### Outputs:
            * ```list```
              The picked metrics used for model select.
            * ```list```
              The name of these metrics.
        '''
        return [evaluation_report_format_dict['absolute_NLL_loss'], 
                test_report_format_dict['absolute_NLL_loss']], \
               ['evaluation_absolute_loss', 'test_absolute_loss']

    '''
    metric number is the length of the output of choose_metric
    '''
    metric_number = 2 # metric number is the length of the output of choose_metric
