'''A wrapper class for scheduled optimizer '''

from .. import utils
import torch.optim as optim

logger = utils.getLogger(__name__)

class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''
    def __init__(self, opt, model):
        if opt.custom_op:
            import torch_optimizer as top
            if not hasattr(top, opt.op_name) and not hasattr(optim, opt.op_name):
                raise logger.exception(f'The given optimizer {opt.op_name} is not found in neither PyTorch nor pytorch_optimizer. Please check your optimizer settings and try again.')
        else:
            if not hasattr(optim, opt.op_name):
                raise logger.exception(f"The given optimizer {opt.op_name} is not found. Maybe it is a custom optimizer. Please set --custom_op and try again.")
    
        param = utils.read_json(opt.optim_json)
        logger.info(f'The additional input optimizer hyperparameters are {param}')
        if hasattr(optim, opt.op_name):
            self._optimizer = getattr(optim, opt.op_name)(model.parameters(), opt.lr, **param)
        else:
            self._optimizer = top.get(opt.op_name)(model.parameters(), opt.lr, **param)
        
        if opt.lr_sched:
            self.n_warmup_steps = opt.n_warmup_steps
            self.n_training_steps = utils.training_steps(opt.training_size, opt.epoch, opt.batch_size)
            self.n_cycles = opt.n_cycles
            self.last_epoch = opt.last_epoch
            self._scheduler = utils.get_lr_sheduler(optimizer = self._optimizer, num_warmup_steps = self.n_warmup_steps, 
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
        self._optimizer.zero_grad()

    def get_lr(self):
        lr = []
        for items in self._optimizer.state_dict()['param_groups']:
            lr.append(items['lr'])

        return utils.mean(lr)