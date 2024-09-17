'''A wrapper for a scheduled optimizer '''

import math
from src.toolbox.misc import get_logger, read_yaml
from src.toolbox.list_operation import list_mean
import torch.optim as optim

logger = get_logger(__name__)


def generate_optimizer_scheduler(opt, model):
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
        optimizer = getattr(optim, opt.op_name)(model.parameters(), opt.lr, **param)
    else:
        optimizer = top.get(opt.op_name)(model.parameters(), opt.lr, **param)
        
    if opt.lr_sched:
        scheduler = get_lr_sheduler(optimizer = optimizer, num_warmup_steps = opt.n_warmup_steps, 
                                    num_training_steps = opt.n_training_steps,
                                    num_cycles = opt.n_cycles, last_epoch = opt.last_epoch)
    else:
        scheduler = None

    return optimizer, scheduler


def step_and_update_lr(optimizer, scheduler):
    "Step with the inner optimizer"
    optimizer.step()

    if scheduler:
        scheduler.step()


def zero_grad(optimizer):
    "Zero out the gradients with the inner optimizer"
    optimizer.zero_grad(set_to_none = True)


def get_lr(optimizer):
    lr = []
    for items in optimizer.state_dict()['param_groups']:
        lr.append(items['lr'])

    return list_mean(lr)
    

def state_dict(optimizer, scheduler):
    if scheduler:
        return {'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}
    else:
        return {'optimizer': optimizer.state_dict(), 'scheduler': None}
    

def load_state_dict(optimizer, scheduler, state_dict):
    optimizer.load_state_dict(state_dict['optimizer'])

    if scheduler:
        scheduler.load_state_dict(state_dict['scheduler'])

    return optimizer, scheduler


def get_lr_sheduler(optimizer, num_warmup_steps, num_training_steps, num_cycles, last_epoch):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lr_lambda, last_epoch = last_epoch)