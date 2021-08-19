import torch
import torch.utils as utils
import os
import pandas as pd
import numpy as np


class IflDataset(utils.data.Dataset):
    '''
    Self defined dataset. The required pandas DataFrame are listed in train.py.
    But...what can we do if we need prediction? It is strange.
    '''

    def __init__(self, data, device, start_time: int = None, end_time: int = None):
        super(IflDataset, self).__init__()
        self.data = data
        self.device = device
        # All input data has the same sequence length.
        self.sequence_length = len(self.data.iloc[0].time_seq)
        self.start_time = torch.tensor([start_time]) if start_time else torch.tensor([0])
        self.end_time = torch.tensor([end_time]) if end_time else torch.tensor([300])
        
        # Data normalization
        regenerated_data = pd.DataFrame(self.data['time_seq'].values.tolist())
        regenerated_data.insert(0, 'start', self.start_time.item())
        regenerated_data.insert(regenerated_data.columns.size, 'end', self.end_time.item())
        regenerated_data = np.log(regenerated_data.diff(axis = 1) + 1e-8).stack()
        self.mean = regenerated_data.mean()
        self.var = regenerated_data.var()

    def __getitem__(self, index):
        # score is the global fact. So we need to modify the first part of the minibatch
        if isinstance(index, slice):
            return [
                self[idx] for idx in range(index.start or 0, index.stop or len(self), index.step or 1)
            ]
        else:
            '''
            Based on the ifl model documents, the dataset has three parts
            1. event_tensor: Data tensors that contain event marks for available sequences.
            2. time_tensor: The timestamp of each event in relative style.
            3. mask_tensor: Tensors to mask out dummy events(under such circumstance the only dummy event is the last "process end" one.)
            4. mean
            5. var: these two variables are used for input data normalization.

            Seems that t_start and t_end are fixed and stay unchanged unless the dataset get changed.
            finally we should tell the model how many event types the dataset has.(Maybe this can be a model hyperparameter)
            '''
            event_tensor = torch.cat(
                (torch.tensor(self.data.iloc[index].event), torch.tensor([10]))
                )
            time_tensor = torch.diff(torch.cat(
                (self.start_time, torch.tensor(self.data.iloc[index].time_seq), self.end_time)
            ))
            mask_tensor = torch.cat(
                (torch.ones(self.sequence_length), torch.tensor([0]))
            )
            return [event_tensor, time_tensor, mask_tensor, self.mean, self.var], \
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