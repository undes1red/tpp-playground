import torch


from src.fakenews.model.heard.heard import HEARD
from src.fakenews.model.utils import BasicModule


class HEARDWrapper(BasicModule):
    '''
    The new house for the BEARD model. We would directly fit the code here, instead of implementing
    the model by ourselves.
    '''
    def __init__(self, num_events, device, batch_size, RD_d_hidden, HC_d_hidden, RD_d_input, RD_num_layer, 
                 loss_weight_HC, loss_weight_early_pred, less_weight_shift_count, fcn_dropout, \
                 lstm_dropout):
        super(HEARDWrapper, self).__init__()

        self.model = HEARD(num_events = num_events, batch_size = batch_size, RD_d_hidden = RD_d_hidden, \
                           HC_d_hidden = HC_d_hidden, RD_d_input = RD_d_input, RD_num_layer = RD_num_layer, \
                           loss_weight_HC = loss_weight_HC, loss_weight_early_pred = loss_weight_early_pred, \
                           less_weight_shift_count = less_weight_shift_count, fcn_dropout = fcn_dropout, \
                           lstm_dropout = lstm_dropout, device = device)


    def forward(self, minibatch, evaluate):
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
        return self.evaluate_procedure(minibatch) if evaluate \
            else self.train_procedure(minibatch)


    def train_procedure(self, minibatch):
        self.train()

        label_seqs, post_indexes, real_lens = minibatch[0],minibatch[3],minibatch[10]

        outputs = self.model(minibatch, if_dp = True)
        loss,stop_points,stop_preds,_,_ = self.model.compute_log_likelihood(minibatch, if_dp = True)

        return loss, stop_points, stop_preds


    def evaluate_procedure(self, minibatch):
        self.eval()

        label_seqs, post_indexes, real_lens = minibatch[0],minibatch[3],minibatch[10]

        outputs = self.model(minibatch, if_dp = False)
        loss,stop_points,stop_preds,_,_ = self.model.compute_log_likelihood(minibatch, if_dp = False)

        return loss, stop_points, stop_preds


    def train_step(model, minibatch, device):
        loss, stop_points, stop_preds = model(minibatch, evaluate = False)

        loss.backward()

        loss_val = loss.item()

        return loss_val


    def evaluation_step(model, minibatch, device):
        pass


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
            return [input[0], input[0] - input[1], input[2], input[3], input[4]]
        
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
            format_dict['absolute_loss'] = input[0]
            format_dict['relative_loss'] = input[1]
            format_dict['events_loss'] = input[2]
            format_dict['mae'] = input[3]
            format_dict['f1_value'] = input[4]
            format_dict['num_format'] = {'absolute_loss': ':6.5f', 'relative_loss': ':6.5f',
                                         'events_loss': ':6.5f', 'mae': ':2.8f', 'f1_value': ':2.8f'}
            return format_dict
        
        return (train_log_print_format(input) if procedure == 'Training' else test_log_print_format(input))

    format_dict_length = 5
    
    logfile_format = {'step': '', 'absolute loss': ':6.5f', 'relative loss': ':6.5f', 'events loss': ':6.5f', 'mae': ':2.8f', 'f1_value': ':2.8f'}


    # The largest length of the format_dict
    format_dict_length = 0

    logfile_format = {'step': ''}


    def logfile_print_format(input):
        '''
        See the annotations above.
        '''
        pass


    def choose_metric(evaluation_report, test_report):
        pass