'''A wrapper class for scheduled optimizer '''

from .. import utils

class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, scheduler = True, num_warmup_steps = None, num_training_steps = None, num_cycles = 0.5, last_epoch = -1):
        self._optimizer = optimizer
        if scheduler:
            self._scheduler = utils.get_lr_sheduler(optimizer = self._optimizer, num_warmup_steps = num_warmup_steps, num_training_steps = num_training_steps
                                                    , num_cycles = num_cycles, last_epoch = last_epoch)
        else:
            self._scheduler = None

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._optimizer.step()

        if self._scheduler:
            self._scheduler.step()

    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._optimizer.zero_grad()

    def get_lr(self):
        lr = []
        for items in self._optimizer.state_dict()['param_groups']:
            lr.append(items['lr'])

        return utils.mean(lr)