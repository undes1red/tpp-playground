import torch

def move_from_tensor_to_ndarray(*kwargs):
    '''
    This function converts an arbitrary number of torch.tensor to np.array.
    This function can automaticly move cuda tensor to cpu.
    '''
    def move_tensor(x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        else:
            return x

    if len(kwargs) == 1:
        tmp_results = move_tensor(kwargs[0])
    else:
        tmp_results = []
        for object in kwargs:
            tmp_results.append(move_tensor(object))

    return tmp_results