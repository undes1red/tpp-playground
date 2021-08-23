import torch
import torch.utils as utils
import os
import pandas as pd


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

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [
                self[idx] for idx in range(index.start or 0, index.stop or len(self), index.step or 1)
            ]
        else:
            raw_time = torch.cat((torch.tensor([0.]), torch.tensor(self.data.iloc[index].time_seq)))
            diff_time = torch.cat((torch.tensor([0.]), torch.diff(raw_time)))
            if self.plot:
                return diff_time, torch.cat((torch.tensor([0.]), torch.tensor(self.data.iloc[index].score)))
            else:
                return diff_time, torch.cat((torch.tensor([0.]), torch.tensor(self.data.iloc[index].score))),\
                       torch.tensor(self.data.iloc[index].intensity)

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