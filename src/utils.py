from functools import reduce
import math
import logging
import argparse
import os

import torch.optim.lr_scheduler as lrs

# Logger settings
def getLogger(name):
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

logger = getLogger(__name__)

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

# Calculate the total training step
def training_steps(dataset_size, epoch, batch):
    return math.ceil(dataset_size * epoch / batch)


# Definition of path parsing action.
class path(argparse.Action):
    def __call__(self, parser, namespace, values, option_string = None):
        setattr(namespace, self.dest, os.path.abspath(values))


def print_performances(procedure, num_format = None, **kwargs):
    if num_format is None or len(num_format) != len(kwargs):
        num_format = [':5.2f'] * len(kwargs)

    info = f'{procedure:12} '
    for idx, key in enumerate(kwargs.keys()):
        info += ' ,' + key + ': {' + key + num_format[idx] + '}'
    logger.info(info.format_map(kwargs))