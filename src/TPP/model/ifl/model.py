from .log_norm_mix import LogNormMix
from ..utils import BasicModule
from sklearn.metrics import f1_score

import torch

class IFL(BasicModule):
    def __init__(self, num_events: int, device, mean_log_inter_time: float = 0.0, std_log_inter_time: float = 1.0, 
                       context_size: int = 32, mark_embedding_size: int = 32, num_mix_components: int = 16, rnn_type: str = "LSTM",
                       mae_threshold = 2):
        super(IFL, self).__init__()
        self.device = device
        self.mae_threshold = mae_threshold

        self.model = LogNormMix(
            num_events + 1,
            self.device,
            mean_log_inter_time,
            std_log_inter_time,
            context_size,
            mark_embedding_size,
            num_mix_components,
            rnn_type,
        )
    
    def forward(self, minibatch, evaluate = False):
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
        mae = 0
        if evaluate:
            mae = self.mean_absolute_error(minibatch)

        return self.model.log_prob(minibatch), mae

    def evaluate(self, minibatch, taus):
        probability, _ = self.model.log_cdf(minibatch, taus)
        return probability

    def mean_absolute_error_and_f1(self, input_events, input_time, mask, mean, var):
        # where should be padded scores is now an empty list. 
        minibatch = [[input_events, input_time, mask], [], [mean, var]]

        # Obtain dedicated MAE and predicted time.
        gap, pred_time = self.mean_absolute_error(minibatch, sum = False, output_pred = True)
                                                                               # [batch_size, seq_len + 1]
        predicted_events  = self.model.event_prober(input_events, input_time, mask, [mean, var])
                                                                               # [batch_size, seq_len + 1]
        gap = gap[:, :-1]
        predicted_events = predicted_events[:, :-1]
        input_events = input_events[:, :-1]

        f1 = f1_score(y_pred = predicted_events.squeeze().detach().cpu().numpy(), \
                      y_true = input_events.squeeze().detach().cpu().numpy(), average = 'macro')

        return gap, f1

    def mean_absolute_error(self, minibatch, sum = True, output_pred = False):
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

        if sum:
            gap_mean = torch.sum(gap) / mask.sum()
            if output_pred:
                return gap_mean.item(), tau_pred
            else:
                gap_mean = torch.sum(gap) / mask.sum()
                return gap_mean.item()
        else:
            if output_pred:
                return gap, tau_pred
            else:
                return gap
    
    def function_prober(self, data, resolution):
        self.model.eval()
        return self.model.log_prob_prober(data, resolution)                    # [batch_size, seq_len * resolution]

    def train_step(model, minibatch, device):
        ''' Epoch operation in training phase'''
    
        model.train()
            
        (log_likelihood, the_number_of_events), mae = model(minibatch)
    
        loss = loss_f(log_likelihood)
        loss.backward()
    
        loss = loss.item() / the_number_of_events
        fact = minibatch[1].sum() / the_number_of_events
    
        return loss, fact
    
    def evaluation_step(model, minibatch, device):
        ''' Epoch operation in evaluation phase '''
    
        model.eval()
        (log_likelihood, the_number_of_events), mae = model(minibatch, evaluate = True)
    
        loss = loss_f(log_likelihood)
    
        loss = loss.item() / the_number_of_events
        fact = minibatch[1].sum() / the_number_of_events
    
        return loss, fact, mae

    def postprocess(input, procedure):
        def train(input):
            return [input[0], input[0] - input[1]]

        def evaluate(input):
            return [input[0], input[0] - input[1], input[2]]
        
        return train(input) if procedure == 'Training' else evaluate(input)

    def log_print_format(input, procedure):
        def train(input):
            format_dict = {}
            format_dict['absolute_loss'] = input[0]
            format_dict['relative_loss'] = input[1]
            format_dict['num_format'] = {'absolute_loss': ':8.5f', 'relative_loss': ':8.5f'}
            return format_dict
        
        def evaluate(input):
            format_dict = {}
            format_dict['absolute_loss'] = input[0]
            format_dict['relative_loss'] = input[1]
            format_dict['MAE'] = input[2]
            format_dict['num_format'] = {'absolute_loss': ':8.5f', 'relative_loss': ':8.5f', 'MAE': ':2.8f'}
            return format_dict
        
        return train(input) if procedure == 'Training' else evaluate(input)

    format_dict_length = 3
    
    logfile_format = {'step': '', 'absolute loss': ':8.5f', 'relative loss': ':8.5f'}

    def logfile_print_format(input):
        format_dict = {}
        format_dict['absolute loss'] = input[0]
        format_dict['relative loss'] = input[1]
        return format_dict
    
    def choose_metric(evaluation_report, test_report):
        '''
        [relative loss on evaluation dataset, relative loss on test dataset]
        '''
        return [test_report[1],]
    
    metric_number = 1 # metric number is the length of the output of choose_metric

def loss_f(loglik):
    '''
    The definition of loss.
    '''
    return (-loglik).sum()
