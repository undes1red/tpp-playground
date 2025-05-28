import torch

from src.toolbox.misc.get_logger import get_logger

logger = get_logger(__name__)


def compile_model(model, use_compile, *args, **kwargs):
    if use_compile:
        logger.warning('Optimizing the model by torch.compile(). This process may not work OOTB in some cases because of unsupported devices, out-of-date graphic drivers, wrong triton installation, specific model design, pytorch bugs, etc.')
        
    return torch.compile(
        model,
        *args,
        disable = not use_compile, 
        **kwargs)


def conditional_compile_func(func, compile_or_not = False):
    def wrapper(*args, **kwargs):
        return torch.compile(func, disable = not compile_or_not)(*args, **kwargs)
    
    return wrapper


def conditional_compile_class_method(func):
    def wrapper(*args, **kwargs):
        # self is always the first input given the decorated function is a method of a class.
        # we do not compile if compile_or_not is not found.
        if hasattr(args[0], 'compile_or_not'):
            compile_or_not = args[0].compile_or_not
        else:
            logger.debug('compile_or_not undefined in the class object! For compatibility, we will not compile the decorated function.')
            compile_or_not = False
        return torch.compile(func, disable = not compile_or_not)(*args, **kwargs)
    
    return wrapper