import torch


def compile_model(model, use_compile, *args, **kwargs):
    return torch.compile(
        model,
        *args,
        disable = not use_compile, 
        **kwargs)