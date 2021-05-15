'''A wrapper class for scheduled optimizer '''

from .. import utils

class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, scheduler = None):
        self._optimizer = optimizer
        self._scheduler = scheduler


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