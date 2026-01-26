import functools

import torch

from src.toolbox.misc.get_logger import get_logger

logger = get_logger(__name__)


def compile_model(model, use_compile, backend=None, fullgraph=True, *args, **kwargs):
    if use_compile and backend:
        logger.warning(
            "Optimizing the model by torch.compile(). This process may not work OOTB in some cases because of unsupported devices, out-of-date graphic drivers, wrong triton installation, specific model design, pytorch bugs, etc."
        )

        return torch.compile(
            model, *args, backend=backend, dynamic=False, fullgraph=fullgraph, disable=not use_compile, **kwargs
        )
    return model


def compile_func(compile_or_not, backend, *args, **kwargs):
    def decorator(func):
        if isinstance(compile_or_not, bool):
            return torch.compile(func, backend=backend, dynamic=False, disable=not compile_or_not, *args, **kwargs)

        if isinstance(compile_or_not, str):

            @functools.wraps(func)
            def wrapper(self, *f_args, **f_kwargs):
                # Check if we have already decided and compiled
                # We use a private attribute on the instance to store the (possibly compiled) function
                compiled_func_attr = f"_compiled_{func.__name__}"
                if not hasattr(self, compiled_func_attr):
                    # Determine if we should compile based on the attribute name provided
                    should_compile = getattr(self, compile_or_not)
                    picked_backend = getattr(self, backend)
                    if should_compile:
                        # Compile the function.
                        # Since it's a method, func takes 'self' as first argument.
                        # torch.compile(func) will return a compiled version of func.
                        compiled = torch.compile(func, backend=picked_backend, dynamic=False, *args, **kwargs)
                    else:
                        compiled = func
                    setattr(self, compiled_func_attr, compiled)

                # Call the stored function, passing 'self' explicitly because 'compiled' is the raw function
                return getattr(self, compiled_func_attr)(self, *f_args, **f_kwargs)

            return wrapper

        raise TypeError(f"compile_or_not must be bool or str, but got {type(compile_or_not)}")

    return decorator
