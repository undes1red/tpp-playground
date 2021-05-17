from functools import reduce
import math

import torch.optim.lr_scheduler as lrs

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