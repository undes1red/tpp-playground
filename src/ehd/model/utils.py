from abc import ABCMeta, abstractmethod
from einops import rearrange, reduce, repeat
import torch
import numpy as np
import torch.nn as nn


class BasicModule(nn.Module, metaclass = ABCMeta):
    '''
    The parent of all model classes.
    '''
    @abstractmethod
    def forward(self, *args):
        '''
        '''
    
    @staticmethod
    @abstractmethod
    def train_step(model, minibatch, device):
        '''
        Please tell us how your model propagates and obtains a proper loss value using one minibatch from the training dataset.
        '''

    @staticmethod
    @abstractmethod
    def evaluation_step(model, minibatch, device):
        '''
        Please tell us how your model propagates and obtains a proper loss value using one minibatch from the evaluation dataset.
        '''

    @staticmethod
    @abstractmethod
    def postprocess(input):
        '''
        The input is the output of function train_step() or function evaluation_step(). You should return a list
        '''

    '''
    The input of log_print_format() and logfile_print_format() is the output object of function postprocess()
    '''
    @staticmethod
    @abstractmethod
    def log_print_format(input):
        '''
        The output format definition. The rule-defining dict should contain objects listed below:
        1. 'num_format': Please, do not modify the name because the architecture will detect this key and use the corresponding subdict as the output format definition.
        2. What you want to output. You should register the name of each number in list 'input' as a key and each matching number as a value.
        Caveats: All used names should have their own format definition. If you really don't need it for some special outputs, please set it to an empty string ''.
        e.x.:
        input = [a, b]. Expected output: loss_a: a, loss_b: b. Both a and b should keep 5 decimal places.
        The format_dict should be like this:
        {
            'loss_a': a,
            'loss_b': b,
            'num_format': {'loss_a': ':.5f', 'relative_loss': ':.5f'}
        }
        '''
    
    # The largest length of the format_dict
    format_dict_length = 0

    
    metric_number = 0 # metric number is the length of the output of choose_metric
    '''
    evaluation_report and test_report have the same variable mapping with postprocess.
    '''
    @staticmethod
    @abstractmethod
    def choose_metric(evaluation_report, test_report):
        '''
        Choose the metric values that you want to employ for model performance comparison.
    
        You'd better to mark the name of each object in the output list as a reminder, like:
        [relative loss on evaluation dataset, relative loss on test dataset]
        '''

'''
commonly used functions
'''
def move_from_tensor_to_ndarray(*kwargs):
    def move_tensor(x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        else:
            return x

    if len(kwargs) == 1:
        tmp_results = move_tensor(kwargs[0])
    else:
        tmp_results = []
        for object in kwargs:
            tmp_results.append(move_tensor(object))

    return tmp_results


def check_tensor(x):
    '''
    Ensure that the input tensor does not contain: negative numbers, inf, and nan.
    
    Args:
    * x  type: torch.tensor shape: any shape
         the input tensor.

    Outputs:
      No outputs available.
    '''
    try:
        assert (x < 0).any() == False, 'Negative numbers detected!'
        assert torch.isfinite(x).all() == True, 'inf detected in input!'
        assert torch.isnan(x).any() == False, 'Nan detected in input!'
    except Exception as e:
        print(x)
        raise e


'''
custom metrics
'''
def L1_distance_across_events(input, resolution, num_events, time_next):
    '''
    This function calculates the L^1 distance between two functions in scattered form.
    Input:
    1. input:      function values
                   [seq_len * resolution, num_events]
    2. resolution: int
                   the number of points from [t_{i - 1}, t_i]
    3. num_events: int
                   the number of event types
    4. time_next:  [seq_len, num_events]
                   the length of all intervals with interpolations.
    '''

    input = rearrange(input, '(s r) ne -> ne s r', r = resolution)             # [num_events, seq_len, resolution]
    intensity_1 = repeat(input, 'ne s r -> ne new_d s r', new_d = num_events)  # [num_events, num_events, seq_len, resolution]
    intensity_2 = repeat(input, 'ne s r -> new_d ne s r', new_d = num_events)  # [num_events, num_events, seq_len, resolution]
    delta_intensity = np.abs(intensity_1 - intensity_2)                        # [num_events, num_events, seq_len, resolution]

    gap = time_next.detach().cpu().numpy() / (resolution - 1)                  # [seq_len]
    gap = rearrange(gap, 's -> 1 1 s 1')                                       # [num_events, num_events, seq_len, 1]

    L1 = reduce((delta_intensity * gap)[:, :, :, :-1], 'ne1 ne2 s r -> ne1 ne2', 'sum')
                                                                               # [num_events, num_events]
    # round off the value smaller than 1e-6
    L1[L1 < 1e-6] = 0

    return L1


def L1_distance_between_two_funcs(x, y, timestamp, resolution):
    '''
    This function calculates the L^1 distance between two functions.
    Input:
    1. x:          function values
                   [seq_len * resolution, num_events]
    2. y:          function values
                   the number of points from [t_{i - 1}, t_i]
    3. time:       \\Delta t
                   the number of event types
    '''

    function_interval = np.abs(x - y).reshape(-1, resolution)[:, :-1]          # [batch_size * seq_len, resolution - 1]
    timestamp = timestamp.reshape(-1, resolution)[:, 1:]                       # [batch_size * seq_len, resolution - 1]

    L1 = (function_interval * timestamp).sum()

    # round up the value smaller than 1e-6
    if L1 < 1e-6:
        L1 = 0

    return L1