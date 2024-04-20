'''A wrapper for a scheduled optimizer '''

import math
from src.taskhost_utils import get_logger
from src.TPP.utils import mean, read_yaml
import torch.optim as optim

logger = get_logger(__name__)


class ScheduledOptim():
    '''
    A simple wrapper for using various optimizers to train models.
    We only support LambdaLR learning rate scheduler.
    '''
    def __init__(self, opt, model):
        if opt.custom_op:
            '''
            torch_optimizer is a supplementary optimizer collection compatible with pytorch.
            Visit https://github.com/jettify/pytorch-optimizer for more information about torch_optimizer.
            '''
            import torch_optimizer as top
            if not hasattr(top, opt.op_name) and not hasattr(optim, opt.op_name):
                raise logger.exception(f'The given optimizer {opt.op_name} is not found in neither PyTorch nor pytorch_optimizer. Please check your optimizer settings and try again.')
        else:
            if not hasattr(optim, opt.op_name):
                raise logger.exception(f"The given optimizer {opt.op_name} is not found. Maybe it is a custom optimizer. Please set --custom_op and try again.")
            
        '''
        Read in optimizer configurations.
        '''
        param = read_yaml(opt.optim_config)

        logger.info(f'The additional input optimizer hyperparameters are {param}')
        if hasattr(optim, opt.op_name):
            self._optimizer = getattr(optim, opt.op_name)(model.parameters(), opt.lr, **param)
        else:
            self._optimizer = top.get(opt.op_name)(model.parameters(), opt.lr, **param)
        
        if opt.lr_sched:
            self.n_warmup_steps = opt.n_warmup_steps
            self.n_training_steps = opt.n_training_steps
            self.n_cycles = opt.n_cycles
            self.last_epoch = opt.last_epoch
            self._scheduler = get_lr_sheduler(optimizer = self._optimizer, num_warmup_steps = self.n_warmup_steps, 
                                                    num_training_steps = self.n_training_steps,
                                                    num_cycles = self.n_cycles, last_epoch = self.last_epoch)
        else:
            self._scheduler = None


    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._optimizer.step()

        if self._scheduler:
            self._scheduler.step()


    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._optimizer.zero_grad(set_to_none=True)


    def get_lr(self):
        lr = []
        for items in self._optimizer.state_dict()['param_groups']:
            lr.append(items['lr'])

        return mean(lr)
    

    def state_dict(self):
        return {'optimizer': self._optimizer.state_dict(), 'scheduler': self._scheduler.state_dict()}
    

    def load_state_dict(self, state_dict):
        self._optimizer.load_state_dict(state_dict['optimizer'])
        self._scheduler.load_state_dict(state_dict['scheduler'])


def get_lr_sheduler(optimizer, num_warmup_steps, num_training_steps, num_cycles, last_epoch):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lr_lambda, last_epoch = last_epoch)