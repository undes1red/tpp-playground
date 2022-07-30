import torch, random
import numpy as np

class move_data_to_the_correct_device:
    def __init__(self, device):
        self.device = device
    
    '''
    These following two functions try to mimic the official collate_fn().
    '''
    def __call__(self, data):
        from torch.utils.data._utils.collate import default_collate
        data_in_cpu = default_collate(data)
        return self.move_to_device(data_in_cpu)
    
    def move_to_device(self, data):
        data_in_correct_location = []
        for item in data:
            data_in_correct_location.append(item.to(self.device))
        return data_in_correct_location


# Referring to https://pytorch.org/docs/stable/notes/randomness.html#reproducibility
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


dataloader_modulepath = {
    # These following dataloaders fits all other models.
    'syn': ['synthetic.synthetic_dataset', 'syn_dataloader'],
    'syn_arg': ['synthetic_autoregressive.synthetic_autoregressive_dataset', 'synarg_dataloader'],
    'ctlstm': ['ctlstm.ctlstm_dataset', 'ctlstm_dataloader'],
    'cnf': ['cnf.cnf_dataset', 'cnf_dataloader'],
    'ifl': ['ifl.ifl_dataset', 'ifl_dataloader'],

    # 2021-10-14 update: legacy dataloaders are deprecated and are removed from the dataloader zoo.
    # Take your own risk to readd and use them.
}