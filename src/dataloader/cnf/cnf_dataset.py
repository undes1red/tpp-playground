import torch
import torch.utils as utils
import os
import pandas as pd
import numpy as np


class CNFDataset(utils.data.Dataset):
    '''
    Self defined dataset. The required pandas DataFrame are listed in train.py.
    But...what can we do if we need prediction? It is strange.
    '''

    def __init__(self, data, device, have_mask = False):
        super(CNFDataset, self).__init__()
        self.data = data
        self.device = device
        # All input data has the same sequence length.
        self.sequence_length = len(self.data.iloc[0].time_seq)

        # Data preprocessing
        if not have_mask:
            self.data['mask'] = [np.ones(self.sequence_length)] * self.data.shape[0]

        self.data.time_seq = self.data.time_seq.apply(np.array, dtype = np.float32)
        self.data.score = self.data.score.apply(np.array, dtype = np.float32)
        self.data['mask'] = self.data['mask'].apply(np.array, dtype = np.int8)
        self.data.event = self.data.event.apply(np.array, dtype = np.float32)
        self.data.event = self.data.event.apply(np.reshape, newshape = (-1, 1))

    def __getitem__(self, index):
        # score is the global fact. So we need to modify the first part of the minibatch
        if isinstance(index, slice):
            return [
                self[idx] for idx in range(index.start or 0, index.stop or len(self), index.step or 1)
            ]
        else:
            event_tensor = torch.tensor(self.data.iloc[index].event)
            time_tensor = torch.from_numpy(self.data.iloc[index].time_seq)
            mask_tensor = torch.from_numpy(self.data.iloc[index]['mask'])
            return [event_tensor, time_tensor, mask_tensor], \
                   torch.from_numpy(self.data.iloc[index].score)

    def __len__(self):
        return self.data.shape[0]


def read_data(path, file_names):
    data_raw = {}
    is_csv = file_names[0].split('.')[-1] == 'csv'
    try:
        if is_csv:
            for file_name in file_names:
                file, type = file_name.split('.')
                data_raw[file] = pd.read_csv(
                    os.path.join(path, file + '.' + type))
        else:
            for file_name in file_names:
                file, type = file_name.split('.')
                data_raw[file] = pd.read_json(
                    os.path.join(path, file + '.' + type))
    except:
        raise TypeError(
            f"Wrong datafile format. Please check your data file in {path}")
    
    return data_raw