import torch

from src.toolbox.misc.get_logger import get_logger

logger = get_logger(__name__)

def compile_model(model, use_compile, *args, **kwargs):
    if use_compile:
        logger.warning('Optimizing the model by torch.compile(). This process may not work in some conditions because of unsupported devices, out-of-date graphic drivers, wrong triton installation, specific model design, pytorch bugs, etc.')
        
    return torch.compile(
        model,
        *args,
        disable = not use_compile, 
        **kwargs)