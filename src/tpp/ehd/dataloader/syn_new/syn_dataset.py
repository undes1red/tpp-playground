import torch.utils as utils
import os
import pandas as pd
import numpy as np


class syn_dataset(utils.data.Dataset):
    '''
    Self defined dataset. The required pandas DataFrame are listed in start.py.
    But...what can we do if we need prediction? It is strange.
    '''
    def __init__(self, data, device, property_dict):
        super(syn_dataset, self).__init__()
        '''
        Data structure.
        x: input,
        y: output
        label: used for visualization.
        '''
        self.data = data
        self.x = self.data['x'].astype(np.float32)
        self.y = self.data['y'].astype(np.float32)
        self.label = self.data['label'].astype(np.int32)

        self.device = device


    def __getitem__(self, index):
        '''
        Synthetic dataloader is very simple. It doesn't have any event infomation at each timestamp,
        and only the time differences between two neighboring events are available.
        '''
        if isinstance(index, slice):
            return [
                self[idx] for idx in range(index.start or 0, index.stop or len(self), index.step or 1)
            ]
        else:
            return self.x[index], self.y[index], self.label[index]


    def __len__(self):
        return self.x.shape[0]
    
    
    data_collator = None


def read_data(path, file_names):
    data_raw = {}
    try:
        for file_name in file_names:
            file, _ = file_name.split('.')
            data_raw[file] = np.load(
                os.path.join(path, file_name), allow_pickle = True)
    except:
        raise TypeError(
            f"Wrong datafile format. Please check your data file in {path}")
    
    return data_raw


def syn_dataloader():
    '''
    Synthetic dataloader for all synthetic datasets.
    '''
    return [syn_dataset, read_data]