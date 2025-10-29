import torch

from src.toolbox.misc.get_logger import get_logger

logger = get_logger(__name__)


def compile_model(model, use_compile, backend=None, *args, **kwargs):
    if use_compile and backend:
        logger.warning(
            "Optimizing the model by torch.compile(). This process may not work OOTB in some cases because of unsupported devices, out-of-date graphic drivers, wrong triton installation, specific model design, pytorch bugs, etc."
        )

        return torch.compile(
            model, *args, backend=backend, dynamic=False, fullgraph=True, disable=not use_compile, **kwargs
        )
    return model


def conditional_compile_func(func, backend, compile_or_not=False, fullgraph=True):
    return torch.compile(func, backend=backend, dynamic=False, fullgraph=fullgraph, disable=not compile_or_not)
