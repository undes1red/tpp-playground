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

    def __init__(self, data, device, event_num = 10, have_mask = False, start_time: int = None, end_time: int = None, input_norm_data = True, shift = False):
        super(IflDataset, self).__init__()
        self.data = data
        self.device = device
        # All input data has the same sequence length.
        self.sequence_length = len(self.data.iloc[0].time_seq)
        self.start_time = start_time if start_time else 0
        self.end_time = end_time if end_time else 350
        self.input_norm_data = input_norm_data
        self.event_num = event_num
        self.have_mask = have_mask

        # Use shift if the event sequences don't start at timestamp 0.
        # If enabled, the time interval between the first event and the start will always be 1s.
        self.shift = shift

        # Data normalization
        if input_norm_data:
            regenerated_data = pd.DataFrame(self.data['time_seq'].values.tolist())
            regenerated_data.insert(0, 'start', self.start_time)
            regenerated_data.insert(regenerated_data.columns.size, 'end', self.end_time)
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
            if self.shift:
                self.start_time = self.data.iloc[index]['time_seq'][0] - 1
                self.end_time += self.start_time

            event_tensor = torch.cat(
                (torch.tensor(self.data.iloc[index].event), torch.tensor([self.event_num]))
                )
            time_tensor = torch.diff(torch.cat(
                (torch.tensor([self.start_time]), torch.tensor(self.data.iloc[index].time_seq), torch.tensor([self.end_time]))
            )) + 1e-5
            if self.have_mask:
                mask_tensor = torch.cat(
                    (torch.tensor(self.data.iloc[index]['mask']), torch.tensor([0]))
                )
            else:
                mask_tensor = torch.cat(
                    (torch.ones(self.sequence_length), torch.tensor([0]))
                )

            return [event_tensor, time_tensor, mask_tensor, self.mean, self.var] if self.input_norm_data else [event_tensor, time_tensor, mask_tensor], \
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