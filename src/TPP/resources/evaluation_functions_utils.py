import torch


def flatten(np_lists):
    results = []
    for np_list in np_lists:
        results += np_list.flatten().tolist()
    return results

def free_model_from_gpu(model):
    del model

    import gc         # garbage collect library
    gc.collect()
    torch.cuda.empty_cache()