from abc import ABCMeta, abstractmethod
import torch.nn as nn


class BasicModule(nn.Module, metaclass = ABCMeta):
    '''
    The parent of all model classes.
    '''
    @abstractmethod
    def forward(self, *args):
        raise NotImplementedError()

    @staticmethod
    def train_step(model, minibatch, device):
        '''
        Please tell us how your model propagates and obtains a proper loss value using one minibatch from the training dataset.
        '''
        raise NotImplementedError()

    @staticmethod
    def evaluation_step(model, minibatch, device):
        '''
        Please tell us how your model propagates and obtains a proper loss value using one minibatch from the evaluation dataset.
        '''
        raise NotImplementedError()

    @staticmethod
    def postprocess(input):
        '''
        The input is the output of function train_step() or function evaluation_step().
        '''
        pass

    '''
    The input of log_print_format() and logfile_print_format() is the output object of function postprocess()
    '''
    @staticmethod
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
        raise NotImplementedError()

    
    '''
    Q: Why is the print format function different from the file print format function?
    A: Because FileLogger needs to know what it should output and prepare the log file before the training procedure begins. Item 'step' in dict 'logfile_format'
    is reserved to record the training progress so you should always have it in 'logfile_format'.
    Other stuff are the same as what log_print_format() does.
    '''
    logfile_format = {'step': ''}

    @staticmethod
    def logfile_print_format(input):
        raise NotImplementedError()

    
    metric_number = 0 # metric number is the length of the output of choose_metric
    '''
    evaluation_report and test_report have the same variable mapping with postprocess.
    '''
    @staticmethod
    def choose_metric(evaluation_report, test_report):
        '''
        Choose the metric values that you want to employ for model performance comparison.
    
        You'd better to mark the name of each object in the output list as a reminder, like:
        [relative loss on evaluation dataset, relative loss on test dataset]
        '''
        raise NotImplementedError()

