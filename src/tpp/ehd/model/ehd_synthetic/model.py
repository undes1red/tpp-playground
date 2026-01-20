import torch, copy, os, importlib, math
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat, reduce, pack

from src.ehd.model.ehd_synthetic.submodel import EHD_backend
from src.ehd.model.basic_ehd_model import BasicModule, check_tensor, move_from_tensor_to_ndarray
from src.ehd.model.ehd_synthetic.plot import * 
from src.ehd.model.ehd_synthetic.nes.nes import NES
from src.ehd.model.ehd_synthetic.syn_function_repo import syn


from src.toolbox.misc import get_logger
logger = get_logger(__name__)


class EHD(BasicModule):
    '''
    The EHD module.
    This module takes data and trained MTPP model, such as FullyNN, FENN, IFIB-C, etc.
    '''
    def __init__(self, opt, training, d_hidden, mlp_layers, lambda_l_c, lambda_l_p, 
                 epsilon, device, additional_model, samples_for_l_p = 32):
        super(EHD, self).__init__()
        self.device = device
        self.opt = opt
        self.training = training
        # The probability gap is the ratio between p(x, H_{o, t_l} - H) and p(x, H_{o, t_l}).
        self.lambda_l_c = lambda_l_c
        self.lambda_l_p = lambda_l_p
        self.samples_for_l_p = samples_for_l_p

        '''
        Preparing the EHD model-agnostic part.
        '''
        self.epsilon = epsilon
        self.num_events = opt.info_dict['num_events']

        self.model = EHD_backend(d_hidden = d_hidden, mlp_layers = mlp_layers, device = device)

        self.nes_module = NES(num_of_samples_mask = self.samples_for_l_p, epsilon = self.epsilon, tau = 0.1, device = self.device)
        
        self.synthetic_model = syn(additional_model['function'], **additional_model['kwargs'])


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
            'graph': self.plot
        }

        return task_mapper[task_name](*args, **kwargs)


    def train_procedure(self, input_x, input_y, label):
        (L_c, L_g), _, _, (padded_selected_filtered_input_x, padded_selected_filtered_input_y) \
            = self.nes_module(self.model, (input_x, input_y), \
                              discrete_inputs = (label, ), \
                              continuous_inputs = (input_x, input_y), pick_essential_events = True)
                                                                               # [samples_for_l_p * batch_size, *]
        pad_selected_mask = (padded_selected_filtered_input_x != -100).int()                     # [samples_for_l_p * batch_size, *]
        gap = self.synthetic_model(x = padded_selected_filtered_input_x, y = padded_selected_filtered_input_y)

        L_p = (gap * pad_selected_mask).sum() / max(pad_selected_mask.sum(), 1)
        Loss = self.lambda_l_c * L_c + self.lambda_l_p * L_p

        return Loss, L_c, L_p, L_g


    def evaluate_procedure(self, input_x, input_y, label, percentage = True):
        '''
        Since we removed all sequence shorter than seq_len_x + seq_len_h.
        We do not need to worry about the input_mask anymore.
        '''
        _, _, (padded_filtered_labels,), (padded_filtered_input_x, padded_filtered_input_y) \
            = self.nes_module(self.model, (input_x, input_y), \
                              discrete_inputs = (label, ), \
                              continuous_inputs = (input_x, input_y), evaluate = True)
                                                                               # [samples_for_l_p * batch_size, *]
        (L_c, L_g), _, (padded_selected_filtered_labels,), (padded_selected_filtered_input_x, padded_selected_filtered_input_y) \
            = self.nes_module(self.model, (input_x, input_y), \
                              discrete_inputs = (label, ), \
                              continuous_inputs = (input_x, input_y), \
                              evaluate = True, pick_essential_events = True)   # [samples_for_l_p * batch_size, *]
        
        pad_selected_mask = (padded_selected_filtered_input_x != -100).int()   # [samples_for_l_p * batch_size, *]

        gap = self.synthetic_model(x = padded_selected_filtered_input_x, y = padded_selected_filtered_input_y)

        L_p = (gap * pad_selected_mask).sum() / (max(pad_selected_mask.sum(), 1))
        Loss = self.lambda_l_c * L_c + self.lambda_l_p * L_p

        average_label = (padded_selected_filtered_labels * pad_selected_mask).sum() / (max(pad_selected_mask.sum(), 1))

        return Loss, L_c, L_p, L_g, average_label


    def filter(self, input_time, input_events, events_embeddings, input_mask, filter_mask, evaluate = False, output_removed_events = False):
        '''
        Now, filter() should provide \mathcal{H}_{s,o,t_l} and \mathcal{H}_{r,o,t_l} when evaluate = True.
        filter still only provide \mathcal{H}_{r,o,t_l} when evaluate = False.
        '''
        '''
        Please be careful: the mean and var should come from the training dataset!
        '''
        assert filter_mask is not None, "You want to filter the existing history following the filter mask, but filter mask is unavailable!"
        assert torch.is_tensor(filter_mask), "The filter mask has to be a pytorch tensor!"
        if not evaluate:
            assert filter_mask.requires_grad, "The filter mask must be differentiable!"
        samples_for_l_p, batch_size = filter_mask.shape[0], filter_mask.shape[1]

        '''
        Dealing with time.
        We select the time whose history[:, :, 0] == 1(meaning this event will remain).
        '''
        filter_mask_for_nominated = filter_mask[..., 1 if output_removed_events else 0]
                                                                               # [samples_for_l_p, batch_size, seq_len]

        '''
        Why this works?
        We generate the history_mark with Gumbel-softmax trick with zero temperature.
        That enforce the possible values of history_mark is either 1 or 0, although the data type is float.
        We use discrete_history_mask_for_nominated for data selection after we multiply history_mask_for_nominated
        with the input sequence data to introduce the gradient of mask to the selected data sequence.
        Caveat: We convert the float tensor history_mask_for_nominated to LongTensor because we ensure this tensor only contains
        0 and 1. DO NOT do this if your float tensor contains non-integers!
        '''
        discrete_filter_mask_for_nominated = filter_mask[..., 1 if output_removed_events else 0].detach().int()
                                                                               # [samples_for_l_p, batch_size, seq_len]
        the_number_of_remained_event = discrete_filter_mask_for_nominated.sum(dim = -1)
                                                                               # [samples_for_l_p, batch_size]
                
        repeated_input_time = repeat(input_time, '... -> n ...', n = samples_for_l_p)
                                                                               # [samples_for_l_p, batch_size, seq_len]
        repeated_input_events = repeat(input_events, '... -> n ...', n = samples_for_l_p)
                                                                               # [samples_for_l_p, batch_size, seq_len]
        repeated_events_embeddings = repeat(events_embeddings, '... -> n ...', n = samples_for_l_p)
                                                                               # [samples_for_l_p, batch_size, seq_len, d_history]
        repeated_input_mask = repeat(input_mask, '... -> n ...', n = samples_for_l_p)
                                                                               # [samples_for_l_p, batch_size, seq_len]
        
        repeated_cumsum_time = repeated_input_time.cumsum(dim = -1)            # [samples_for_l_p, batch_size, seq_len]
        
        # select the remained events from the original input.
        selected_time = repeated_cumsum_time * filter_mask_for_nominated       # [samples_for_l_p, batch_size, seq_len]
        selected_time = selected_time[discrete_filter_mask_for_nominated == 1] # [...]
        selected_input_events = repeated_input_events[discrete_filter_mask_for_nominated == 1]
                                                                               # [...]
        selected_events_embeddings = repeated_events_embeddings * filter_mask_for_nominated.unsqueeze(dim = -1)
                                                                               # [samples_for_l_p, batch_size, seq_len, d_history]
        selected_events_embeddings = selected_events_embeddings[discrete_filter_mask_for_nominated == 1]
                                                                               # [..., d_history]
        selected_input_mask = repeated_input_mask[discrete_filter_mask_for_nominated == 1]
                                                                               # [...]
        
        data_start_index = 0
        all_reshaped_time, all_reshaped_input_events, all_reshaped_events_embeddings, all_reshaped_input_mask \
            = [], [], [], []
        for the_number_of_remained_event_per_batch in the_number_of_remained_event:
            '''
            Padding the selected timestamps.
            '''
            reshaped_time, reshaped_input_events, reshaped_events_embeddings, reshaped_input_mask \
                = [], [], [], []
            for the_number_of_remained_event_per_batch_per_seq in the_number_of_remained_event_per_batch:
                reshaped_time.append(selected_time[data_start_index:data_start_index + the_number_of_remained_event_per_batch_per_seq])
                reshaped_input_events.append(selected_input_events[data_start_index:data_start_index + the_number_of_remained_event_per_batch_per_seq])
                reshaped_events_embeddings.append(selected_events_embeddings[data_start_index:data_start_index + the_number_of_remained_event_per_batch_per_seq, :])
                reshaped_input_mask.append(selected_input_mask[data_start_index:data_start_index + the_number_of_remained_event_per_batch_per_seq])

                data_start_index += the_number_of_remained_event_per_batch_per_seq
                        
            padded_reshaped_time = torch.nn.utils.rnn.pad_sequence(reshaped_time, batch_first = True)
                                                                               # [batch_size, padded_seq_len]
            padded_input_events = torch.nn.utils.rnn.pad_sequence(reshaped_input_events, batch_first = True)
                                                                               # [batch_size, padded_seq_len]
            padded_events_embeddings = torch.nn.utils.rnn.pad_sequence(reshaped_events_embeddings, batch_first = True)
                                                                               # [batch_size, padded_seq_len, d_history]
            padded_input_mask = torch.nn.utils.rnn.pad_sequence(reshaped_input_mask, batch_first = True)
                                                                               # [batch_size, padded_seq_len]
            
            padded_reshaped_time = padded_reshaped_time.diff(dim = -1, prepend = torch.zeros(batch_size, 1, device = self.device))
                                                                               # [batch_size, padded_seq_len]
            all_reshaped_time.append(padded_reshaped_time)
            all_reshaped_input_events.append(padded_input_events)
            all_reshaped_events_embeddings.append(padded_events_embeddings)
            all_reshaped_input_mask.append(padded_input_mask)

            del reshaped_time, reshaped_input_events, reshaped_events_embeddings, reshaped_input_mask
            del padded_reshaped_time, padded_input_events, padded_events_embeddings, padded_input_mask
        
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return all_reshaped_time, all_reshaped_input_events, all_reshaped_events_embeddings, all_reshaped_input_mask


    def plot(self, minibatch, opt):
        plot_type_to_functions = {
            'removed_events': self.removed_events
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
        input_x, input_y, label = minibatch
        return input_x, input_y, label


    def removed_events(self, input_data, opt):
        '''
        Extract data from the input minibatch.
        '''
        input_x, input_y, label = self.extract_plot_data(input_data)

        (L_c, L_g), event_mask, \
        (padded_filtered_labels,), \
        (padded_filtered_input_x, padded_filtered_input_y) \
            = self.nes_module(self.model, (input_x, input_y), \
                              discrete_inputs = (label,), \
                              continuous_inputs = (input_x, input_y), evaluate = True)

        (L_c_selected, L_g_selected), \
        _, \
        (padded_selected_filtered_labels,), \
        (padded_selected_filtered_input_x, padded_selected_filtered_input_y) \
            = self.nes_module(self.model, (input_x, input_y), \
                              discrete_inputs = (label,), \
                              continuous_inputs = (input_x, input_y), evaluate = True, pick_essential_events = True)
        
        steps = 1000
        expand_x = torch.linspace(start = 0, end = 1, steps = steps, device = self.device)
        expand_x = (input_x.max() - input_x.min()) * expand_x + input_x.min()
        expand_y = self.synthetic_model.get_true_result(expand_x)

        L_c, L_g, L_c_selected, L_g_selected, event_mask, padded_filtered_labels, padded_selected_filtered_labels, padded_filtered_input_x, padded_filtered_input_y, \
        padded_selected_filtered_input_x, padded_selected_filtered_input_y, expand_x, expand_y \
            = move_from_tensor_to_ndarray(L_c, L_g, L_c_selected, L_g_selected, \
                                          event_mask, padded_filtered_labels, padded_selected_filtered_labels, \
                                          padded_filtered_input_x, padded_filtered_input_y, \
                                          padded_selected_filtered_input_x, padded_selected_filtered_input_y, \
                                          expand_x, expand_y)

        data = {
            'L_c': L_c,
            'L_g': L_g,
            'L_c_selected': L_c_selected, 
            'L_g_selected': L_g_selected,
            'event_mask': event_mask,
            'padded_filtered_labels': padded_filtered_labels,
            'padded_filtered_input_x': padded_filtered_input_x,
            'padded_filtered_input_y': padded_filtered_input_y,
            'padded_selected_filtered_labels': padded_selected_filtered_labels,
            'padded_selected_filtered_input_x': padded_selected_filtered_input_x,
            'padded_selected_filtered_input_y': padded_selected_filtered_input_y,
            'expand_x': expand_x,
            'expand_y': expand_y
        }

        plots = plot_removed_events(data, opt)


        return plots
    

    def all_filtered_and_selected_events(self, input_data, opt):
        '''
        Extract data from the input minibatch.
        '''
        input_x, input_y, label = self.extract_plot_data(input_data)

        (L_c, L_g), event_mask, \
        (padded_filtered_labels,), \
        (padded_filtered_input_x, padded_filtered_input_y) \
            = self.nes_module(self.model, (input_x, input_y), \
                              discrete_inputs = (label,), \
                              continuous_inputs = (input_x, input_y), evaluate = True)

        (L_c_selected, L_g_selected), \
        _, \
        (padded_selected_filtered_labels,), \
        (padded_selected_filtered_input_x, padded_selected_filtered_input_y) \
            = self.nes_module(self.model, (input_x, input_y), \
                              discrete_inputs = (label,), \
                              continuous_inputs = (input_x, input_y), evaluate = True, pick_essential_events = True)
        
        steps = 1000
        expand_x = torch.linspace(start = 0, end = 1, steps = steps, device = self.device)
        expand_x = (input_x.max() - input_x.min()) * expand_x + input_x.min()
        expand_y = self.synthetic_model.get_true_result(expand_x)

        L_c, L_g, L_c_selected, L_g_selected, event_mask, padded_filtered_labels, padded_selected_filtered_labels, padded_filtered_input_x, padded_filtered_input_y, \
        padded_selected_filtered_input_x, padded_selected_filtered_input_y, expand_x, expand_y \
            = move_from_tensor_to_ndarray(L_c, L_g, L_c_selected, L_g_selected, \
                                          event_mask, padded_filtered_labels, padded_selected_filtered_labels, \
                                          padded_filtered_input_x, padded_filtered_input_y, \
                                          padded_selected_filtered_input_x, padded_selected_filtered_input_y, \
                                          expand_x, expand_y)

        data = {
            'L_c': L_c,
            'L_g': L_g,
            'L_c_selected': L_c_selected, 
            'L_g_selected': L_g_selected,
            'event_mask': event_mask,
            'padded_filtered_labels': padded_filtered_labels,
            'padded_filtered_input_x': padded_filtered_input_x,
            'padded_filtered_input_y': padded_filtered_input_y,
            'padded_selected_filtered_labels': padded_selected_filtered_labels,
            'padded_selected_filtered_input_x': padded_selected_filtered_input_x,
            'padded_selected_filtered_input_y': padded_selected_filtered_input_y,
            'expand_x': expand_x,
            'expand_y': expand_y
        }

        plots = plot_removed_events(data, opt)


        return plots


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
        input_x, input_y, label = minibatch
        loss, L_c, L_p, L_g = model(         
                task_name = 'train', input_x = input_x, input_y = input_y, label = label
        )
        
        # torch.autograd.set_detect_anomaly(True)
        loss.backward()
        
        loss = loss.item()
        L_c = L_c.item()
        L_p = L_p.item()
        L_g = L_g.item()

        return loss, L_c, L_p, L_g
    

    def evaluation_step(model, minibatch, device):
        ''' Epoch operation in evaluation phase '''
    
        model.eval()
        input_x, input_y, label = minibatch
        loss, L_c, L_p, L_g, average_label = model(         
                task_name = 'evaluate', input_x = input_x, input_y = input_y, label = label
        )
    
        loss = loss.item()
        L_c = L_c.item()
        L_p = L_p.item()
        L_g = L_g.item()
        average_label = average_label.item()

        return loss, L_c, L_p, L_g, average_label


    def postprocess(input, procedure):
        def train_postprocess(input):
            '''
            Training process
            [absolute loss, relative loss, events loss]
            '''
            return [input[0], input[1], input[2], input[3]]
        
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
            format_dict['Loss'] = input[0]
            format_dict['L_c'] = input[1]
            format_dict['L_p'] = input[2]
            format_dict['L_g'] = input[3]
            format_dict['num_format'] = {'Loss': ':6.5f', 'L_c': ':6.5f', 'L_p': ':6.5f', 'L_g': ':6.5f'}
            return format_dict

        def test_log_print_format(input):
            format_dict = {}
            format_dict['Loss'] = input[0]
            format_dict['L_c'] = input[1]
            format_dict['L_p'] = input[2]
            format_dict['L_g'] = input[3]
            format_dict['average_label'] = input[4]
            format_dict['num_format'] = {'Loss': ':6.5f', 'L_c': ':6.5f', 'L_p': ':6.5f', 'L_g': ':6.5f', 
                                         'average_label': ':6.5f'}
            return format_dict
        
        return (train_log_print_format(input) if procedure == 'Training' else test_log_print_format(input))

    format_dict_length = 5
    
    def choose_metric(evaluation_report_format_dict, test_report_format_dict):
        '''
        [relative loss on evaluation dataset, relative loss on test dataset, event loss on test dataset]
        '''
        return [evaluation_report_format_dict['L_p'], 
                test_report_format_dict['L_p']], \
               ['evaluation_L_p_Loss', 'test_L_p_Loss']
    
    metric_number = 2 # metric number is the length of the output of choose_metric