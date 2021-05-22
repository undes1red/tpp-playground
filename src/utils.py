from functools import reduce
import math, logging, argparse, os, json

import torch.optim.lr_scheduler as lrs

# Logger settings
def getEventLogger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
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

def getFileLogger(name, file):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
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

def getLogger(name, file = None):
    if file:
        return getFileLogger(name, file)
    else:
        return getEventLogger(name)

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
        num_format = [':5.2f'] * len(kwargs)

    info = f'{procedure:12} '
    for idx, key in enumerate(kwargs.keys()):
        info += ' ,' + key + ': {' + key + num_format[idx] + '}'
    logger.info(info.format_map(kwargs))


class FileLogger(object):
    def __init__(self, print_item, **kwargs):
        self.loggers = dict()
        for name, path in kwargs.items():
            self.loggers[name] = getLogger(name, path)
        self.print_item = print_item

        # Initial info
        if isinstance(self.print_item, list):
            for logger in self.loggers.values():
                logger.info(', '.join(self.print_item))
        elif isinstance(self.print_item, dict):
            for name in self.print_item.keys():
                self.loggers[name].info(', '.join(self.print_item[name]))
        else:
            logger_.exception(
                'Wrong log index input type. The expected types are list or dict. Please check your input of print_item.'
            )

    def print(self, logger_name, num_format = None, **kwargs):
        logger = self.loggers[logger_name]
        if num_format is None or len(num_format) != len(kwargs):
            num_format = [':5.2f'] * len(kwargs)

        info = ''
        for idx, key in enumerate(self.print_item):
            info += '{' + key + num_format[idx] + '}, '
        logger.info(info.format_map(kwargs))

def read_json(json_path):
    with open(json_path, 'r') as f:
        a = json.load(f)
    return a

def suffix(opt, *args):
    output = ''
    for item in args:
        output += ('_' + str(getattr(opt, item)))
    
    return output