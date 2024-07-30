import gc
import torch


def free_model_from_gpu(model):
    del model

    gc.collect()
    if next(model.parameters()).is_cuda:
        torch.cuda.empty_cache()