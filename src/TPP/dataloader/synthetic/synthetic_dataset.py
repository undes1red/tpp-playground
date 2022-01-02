import torch
import torch.utils as utils
import os
import pandas as pd
import numpy as np

def insert(per_line, number):
    return np.concatenate([np.array([number]), per_line])

class SynDataset(utils.data.Dataset):
    '''
    Self defined dataset. The required pandas DataFrame are listed in train.py.
    But...what can we do if we need prediction? It is strange.
    '''

    def __init__(self, data, device, plot = False, number_of_events = 10):
        super(SynDataset, self).__init__()
        self.data = data
        self.device = device
        self.plot = plot
        self.number_of_events = number_of_events

        # Data preprocessing
        self.data.time_seq = self.data.time_seq.apply(insert, number = 0)
        self.data.time_seq = self.data.time_seq.apply(np.diff)
        self.data.time_seq = self.data.time_seq.apply(insert, number = 0)
        self.data.event = self.data.event.apply(insert, number = self.number_of_events)

        self.data.time_seq = self.data.time_seq.apply(np.array, dtype = np.float32)
        self.data.score = self.data.score.apply(np.array, dtype = np.float32)
        self.data.intensity = self.data.intensity.apply(np.array, dtype = np.float32)
        self.data.event = self.data.event.apply(np.array, dtype = np.float32)

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
            if self.plot:
                return torch.from_numpy(self.data.iloc[index].time_seq).to(self.device), \
                       torch.from_numpy(self.data.iloc[index].event).to(self.device), \
                       torch.from_numpy(self.data.iloc[index].score).to(self.device),\
                       torch.from_numpy(self.data.iloc[index].intensity).to(self.device)
            else:
                return torch.from_numpy(self.data.iloc[index].time_seq).to(self.device), \
                       torch.from_numpy(self.data.iloc[index].event).to(self.device), \
                       torch.from_numpy(self.data.iloc[index].score).to(self.device)

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

def syn_dataloader():
    '''
    Synthetic dataloader for all synthetic datasets.
    '''
    return [SynDataset, read_data]