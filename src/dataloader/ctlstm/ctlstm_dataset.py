import torch
import torch.utils as utils
import os
import pandas as pd


class CTLSTMDataset(utils.data.Dataset):
    '''
    Self defined dataset. The required pandas DataFrame are listed in train.py.
    But...what can we do if we need prediction? It is strange.
    '''

    def __init__(self, data, device):
        super(CTLSTMDataset, self).__init__()
        self.data = data
        self.device = device
        # All input data has the same sequence length. This sequence length does not contain special start and end events.
        self.sequence_length = len(self.data.iloc[0].time_seq)

    def __getitem__(self, index):
        # score is the global fact. So we need to modify the first part of the minibatch
        if isinstance(index, slice):
            return [
                self[idx] for idx in range(index.start or 0, index.stop or len(self), index.step or 1)
            ]
        else:
            # Currently, we just have dummy event labels. This will be fixed in the future by generating MTPP synthetic datasets.
            # Later, we will have native marked datasets.
            # Move to sequence datasets.
            event_tensor = torch.tensor(self.data.iloc[index].event)
            dtime_tensor = torch.diff(torch.tensor([0] + self.data.iloc[index].time_seq))
            dtime_tensor = torch.cat(
                (torch.tensor([0.0]), dtime_tensor, (torch.tensor([0.1])))
            )
            token_num_tensor = torch.tensor([self.sequence_length])
            duration_tensor = torch.sum(dtime_tensor, dim = -1)
            return [event_tensor, dtime_tensor, token_num_tensor, duration_tensor], \
                   torch.tensor(self.data.iloc[index].score)

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