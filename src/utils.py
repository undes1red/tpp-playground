from functools import reduce
import math, logging, argparse, os, json
import torch

import torch.optim.lr_scheduler as lrs

# Logger settings
def getEventLogger(name, root):
    logger = logging.getLogger(name)
    if root:
        logger.parent = None
        logger.root = logger

    logger.setLevel(logging.DEBUG)
    if (logger.hasHandlers()):
        logger.handlers.clear()
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)

    return logger

def getFileLogger(name, file, root):
    logger = logging.getLogger(name)
    if root:
        logger.parent = None
        logger.root = logger

    logger.setLevel(logging.DEBUG)
    if (logger.hasHandlers()):
        logger.handlers.clear()
    # create console handler and set level to debug
    ch = logging.FileHandler(file, mode = 'w')
    ch.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(message)s')
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)

    return logger

def getLogger(name = None, file = None, root = True):
    if file:
        return getFileLogger(name, file, root)
    else:
        return getEventLogger(name, root)

logger_ = getLogger(__name__)

def add(a, b):
    return a + b

def mean(iter):
    return reduce(add, iter)/len(iter)

# For Lambda scheduler
def get_lr_sheduler(optimizer, num_warmup_steps, num_training_steps, num_cycles, last_epoch):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return lrs.LambdaLR(optimizer, lr_lambda = lr_lambda, last_epoch = last_epoch)

# Definition of path parsing action.
class path(argparse.Action):
    def __call__(self, parser, namespace, values, option_string = None):
        setattr(namespace, self.dest, os.path.abspath(values))


def print_performances(logger, procedure, num_format = None, **kwargs):
    if num_format is None or len(num_format) != len(kwargs):
        logger_.exception('Bad num_format dictoinary.')

    info = f'{procedure:12} '
    for key in kwargs.keys():
        info += ' ,' + key + ': {' + key + num_format[key] + '}'
    logger.info(info.format_map(kwargs))


class FileLogger(object):
    def __init__(self, print_format, **kwargs):
        self.loggers = dict()
        for name, path in kwargs.items():
            self.loggers[name] = getLogger(name, path)
        self.print_format = print_format
        self.print_item = self.print_format.keys()

        # Initial info
        for logger in self.loggers.values():
            logger.info(', '.join(self.print_item))

    def print(self, logger_name, **kwargs):
        logger = self.loggers[logger_name]

        info = ''
        for key in self.print_item:
            info += '{' + key + self.print_format[key] + '}, '
        logger.info(info.format_map(kwargs))


def read_json(json_path):
    with open(json_path, 'r') as f:
        a = json.load(f)
    return a

def suffix(opt, *args):
    output = {}
    for item in args:
        output[item] = getattr(opt, item)
    
    return output

def lst_add_lst(list1, list2):
    assert len(list1) == len(list2)
    return [sum(x) for x in zip(list1, list2)]

def lst_divide(lst, denominator):
    if isinstance(denominator, list):
        assert len(lst) == len(denominator)
        return [x/y for x, y in zip(lst, denominator)]
    return [x/denominator for x in lst]

def evaluation(data, model, model_class, device, output_length):
    r = range(1, len(data) + 1)
    data_itr = iter(data)
    sum_ = [0] * output_length
    
    # for _ in tqdm(r, desc=desc, disable=True):
    for _ in r:
        minibatch = next(data_itr)
        batch_sum = model_class.evaluation_step(model, minibatch, device)
        sum_ = lst_add_lst(sum_, batch_sum)

    return lst_divide(sum_, len(data))

class Metric():
    def __init__(self, metric_number, smaller_is_better = None):
        self.metric_number = metric_number
        self.map = {True:1, False: -1}
        self.best_metric = [math.inf] * self.metric_number
        if smaller_is_better is None:
            self.mask = [1] * self.metric_number
        else:
            assert len(smaller_is_better) == self.metric_number
            self.mask = [self.map[item] for item in smaller_is_better]
    
    def compare(self, input_metric):
        assert len(input_metric) == len(self.mask)
        tmp = lst_divide(input_metric, self.mask)
        output = True

        for input_number, recorded in zip(tmp, self.best_metric):
            if input_number >= recorded:
                output = False
                break
        
        if output:
            self.best_metric = input_metric
        
        return output
    
    def show(self):
        return self.best_metric