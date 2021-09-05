import torch
import torch.utils as utils
import os
import pandas as pd
import numpy as np

def data_preprocess(per_line):
    return np.concatenate([np.zeros(1), per_line])

class SynDataset(utils.data.Dataset):
    '''
    Self defined dataset. The required pandas DataFrame are listed in train.py.
    But...what can we do if we need prediction? It is strange.
    '''

    def __init__(self, data, device, plot = False):
        super(SynDataset, self).__init__()
        self.data = data
        self.device = device
        self.plot = plot

        # Data preprocessing
        self.data.time_seq = self.data.time_seq.apply(data_preprocess)
        self.data.time_seq = self.data.time_seq.apply(np.diff)
        self.data.time_seq = self.data.time_seq.apply(data_preprocess)

        self.data.time_seq = self.data.time_seq.apply(np.array, dtype = np.float32)
        self.data.score = self.data.score.apply(np.array, dtype = np.float32)
        self.data.intensity = self.data.intensity.apply(np.array, dtype = np.float32)

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
                return torch.from_numpy(self.data.iloc[index].time_seq), \
                       torch.from_numpy(self.data.iloc[index].score),\
                       torch.from_numpy(self.data.iloc[index].intensity)
            else:
                return torch.from_numpy(self.data.iloc[index].time_seq), \
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