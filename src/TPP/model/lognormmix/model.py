from src.TPP.model.lognormmix.log_norm_mix import LogNormMix
from src.TPP.model.utils import BasicModule
from sklearn.metrics import f1_score

import torch

class LogNormMixWrapper(BasicModule):
    def __init__(self, num_events: int, device, context_size: int = 32, mark_embedding_size: int = 32, \
                 num_mix_components: int = 16, rnn_type: str = "LSTM", probability_threshold = 0.5):
        super(LogNormMixWrapper, self).__init__()
        self.device = device
        self.probability_threshold = probability_threshold

        self.model = LogNormMix(
            num_events + 1,
            self.device,
            context_size,
            mark_embedding_size,
            num_mix_components,
            rnn_type,
        )
    

    def forward(self, input_events, input_time, input_mask, mean, var, evaluate):
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
        return self.evaluate_procedure(input_events, input_time, input_mask, mean, var) if evaluate \
            else self.train_procedure(input_events, input_time, input_mask, mean, var)
    

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
        log_prob, log_p_event = self.model.log_prob(input_events, input_time, mean, var)
                                                                               # [batch_size, seq_len + 1]
        log_prob = log_prob * input_mask                                       # [batch_size, seq_len + 1]
        log_p_event = log_p_event * input_mask                                 # [batch_size, seq_len + 1]
        
        time_loss = self.loss_f(log_prob)
        event_loss = self.loss_f(log_p_event)

        return time_loss, event_loss, the_number_of_events


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
        log_prob, log_p_event = self.model.log_prob(input_events, input_time, mean, var)
                                                                               # [batch_size, seq_len + 1]
        log_prob = log_prob * input_mask                                       # [batch_size, seq_len + 1]
        log_p_event = log_p_event * input_mask                                 # [batch_size, seq_len + 1]
        
        time_loss = self.loss_f(log_prob)
        event_loss = self.loss_f(log_p_event)

        mae, pred_time = self.mean_absolute_error(input_events, input_time, input_mask, mean, var)

        return time_loss, event_loss, mae, the_number_of_events


    def loss_f(self, loglik):
        '''
        The definition of loss.
        '''
        return (-loglik).sum()


    def evaluate(self, minibatch, taus):
        probability, _ = self.model.log_cdf(minibatch, taus)
        return probability


    def mean_absolute_error_and_f1(self, input_events, input_time, mask, mean, var):

        # Obtain dedicated MAE and predicted time.
        gap, pred_time = self.mean_absolute_error(minibatch)
                                                                               # [batch_size, seq_len + 1]
        predicted_events  = self.model.event_prober(input_events, input_time, mask, [mean, var])
                                                                               # [batch_size, seq_len + 1]
        gap = gap[:, :-1]
        predicted_events = predicted_events[:, :-1]
        input_events = input_events[:, :-1]

        f1 = f1_score(y_pred = predicted_events.squeeze().detach().cpu().numpy(), \
                      y_true = input_events.squeeze().detach().cpu().numpy(), average = 'macro')

        return gap, f1


    def mean_absolute_error(self, minibatch):
        '''
        The input should be the original minibatch.
        MAE evaluation part for intensity-free model.
        '''
        def bisect_target(minibatch, taus):
            return self.evaluate(minibatch, taus) - 1 / self.mae_threshold
        
        def median_prediction(minibatch, l, r):
            for _ in range(30):
                c = (l + r)/2
                v = bisect_target(minibatch, c)
                l = torch.where(v < 0, c, l)
                r = torch.where(v >= 0, c, r)

            return (l + r)/2
        
        sequences, _, _ = minibatch
        event, time_interval, mask = sequences
        number_of_events = mask.sum()

        l = 0.0001*torch.ones_like(event, dtype = torch.float32)               # [batch_size, seq_len]
        r = 1e6*torch.ones_like(event, dtype = torch.float32)                  # [batch_size, seq_len]
        tau_pred = median_prediction(minibatch, l, r)
        gap = (tau_pred - time_interval) * mask                                # [batch_size, seq_len]
        gap = torch.abs(gap)                                                   # [batch_size, seq_len]

        return gap, tau_pred
    

    def function_prober(self, data, resolution):
        self.model.eval()
        return self.model.log_prob_prober(data, resolution)                    # [batch_size, seq_len * resolution]


    def train_step(model, minibatch, device):
        ''' Epoch operation in training phase'''
    
        def extract_minibatch(minibatch):
            (input_events, input_time, input_mask), padded_score, (mean, var) = minibatch
            return {'input_events': input_events, 'input_time': input_time, 'input_mask': input_mask, 'mean': mean, 'var': var}

        model.train()
        time_loss, event_loss, the_number_of_events = model(**extract_minibatch(minibatch), evaluate = False)

        loss = time_loss + event_loss
        loss.backward()
    
        loss = loss.item() / the_number_of_events
        time_loss = time_loss.item() / the_number_of_events
        event_loss = event_loss.item() / the_number_of_events
        fact = minibatch[1].sum().item() / the_number_of_events
    
        return loss, time_loss, event_loss, fact
    

    def evaluation_step(model, minibatch, device):
        ''' Epoch operation in evaluation phase '''
    
        def extract_minibatch(minibatch):
            (input_events, input_time, input_mask), padded_score, (mean, var) = minibatch
            return {'input_events': input_events, 'input_time': input_time, 'input_mask': input_mask, 'mean': mean, 'var': var}

        model.eval()
        time_loss, event_loss, mae, the_number_of_events = model(**extract_minibatch(minibatch), evaluate = True)
        loss = time_loss + event_loss

        loss = loss.item() / the_number_of_events
        time_loss = time_loss.item() / the_number_of_events
        event_loss = event_loss.item() / the_number_of_events
        fact = minibatch[1].sum().item() / the_number_of_events
    
        return loss, time_loss, event_loss, fact, mae


    def postprocess(input, procedure):
        def train(input):
            return [input[0], input[0] - input[-1], input[1], input[2]]

        def evaluate(input):
            return [input[0], input[0] - input[-2], input[1], input[2], input[3]]
        
        return train(input) if procedure == 'Training' else evaluate(input)


    def log_print_format(input, procedure):
        def train(input):
            format_dict = {}
            format_dict['absolute_loss'] = input[0]
            format_dict['relative_loss'] = input[1]
            format_dict['time_loss'] = input[2]
            format_dict['event_loss'] = input[3]
            format_dict['num_format'] = {'absolute_loss': ':8.5f', 'relative_loss': ':8.5f',\
                                         'time_loss': ':8.5f', 'event_loss': ':8.5f'}
            return format_dict
        
        def evaluate(input):
            format_dict = {}
            format_dict['absolute_loss'] = input[0]
            format_dict['relative_loss'] = input[1]
            format_dict['time_loss'] = input[2]
            format_dict['event_loss'] = input[3]
            format_dict['MAE'] = input[4]
            format_dict['num_format'] = {'absolute_loss': ':8.5f', 'relative_loss': ':8.5f', 'time_loss': ':8.5f', \
                                         'event_loss': ':8.5f', 'MAE': ':2.8f'}
            return format_dict
        
        return train(input) if procedure == 'Training' else evaluate(input)


    format_dict_length = 5
    

    def choose_metric(evaluation_report_format_dict, test_report_format_dict):
        '''
        [relative loss on evaluation dataset, relative loss on test dataset]
        '''
        return [evaluation_report_format_dict['loss_time_next'], test_report_format_dict['loss_time_next']], \
               ['evaluation_absolute_loss', 'test_absolute_loss']
    

    metric_number = 1 # metric number is the length of the output of choose_metric
