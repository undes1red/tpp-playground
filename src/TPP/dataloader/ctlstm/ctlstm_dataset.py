import torch
import torch.utils as utils
import os
import pandas as pd
import numpy as np

def concate(per_line, item1 = np.array([]), item2 = np.array([])):
    return np.concatenate([item1, per_line, item2])

class CTLSTMDataset(utils.data.Dataset):
    '''
    Self defined dataset. The required pandas DataFrame are listed in train.py.
    But...what can we do if we need prediction? It is strange.

    event_number is the number of all existing events, dummy events are not included.

    You can set event_number = 1 to mask all events by forcing them to have the same label.
    '''

    def __init__(self, data, device, event_number = 10):
        super(CTLSTMDataset, self).__init__()
        self.data = data
        self.device = device
        # All input data has the same sequence length. This sequence length does not contain special start and end events.
        self.sequence_length = len(self.data.iloc[0].time_seq)
        self.token_num_tensor = torch.tensor([self.sequence_length], device = self.device)
        self.event_number = event_number

        # Data preprocessing
        # Add dummy start and end events.
        if self.event_number == 1:
            self.data.event = self.data.event.apply(np.zeros_like)
            self.data.event = self.data.event.apply(concate, item1 = np.array([self.event_number]), item2 = np.array([self.event_number + 1]))
        else:
            self.data.event = self.data.event.apply(concate, item1 = np.array([self.event_number]), item2 = np.array([self.event_number + 1]))

        self.data.time_seq = self.data.time_seq.apply(concate, item1 = np.array([0]))
        self.data.time_seq = self.data.time_seq.apply(np.diff)
        self.data.time_seq = self.data.time_seq.apply(concate, item1 = np.array([0]), item2 = np.array([0.1]))

        self.data.event = self.data.event.apply(np.array, dtype = np.int32)
        self.data.time_seq = self.data.time_seq.apply(np.array, dtype = np.float32)
        self.data['duation'] = self.data.time_seq.apply(np.sum, dtype = np.float32)
        self.data.intensity = self.data.intensity.apply(np.array, dtype = np.float32)

    def __getitem__(self, index):
        # score is the global fact. So we need to modify the first part of the minibatch
        if isinstance(index, slice):
            return [
                self[idx] for idx in range(index.start or 0, index.stop or len(self), index.step or 1)
            ]
        else:
            '''
            Outputs:
            event_tensor
            dtime_tensor
            token_num_tensor
            duration_tensor
            '''
            event_tensor = torch.from_numpy(self.data.iloc[index].event).to(self.device)
            dtime_tensor = torch.from_numpy(self.data.iloc[index].time_seq).to(self.device)
            intensity_tensor = torch.from_numpy(self.data.iloc[index].intensity).to(self.device)
            duation_tensor = torch.from_numpy(self.data.iloc[index].duation).to(self.device)

            
            return [event_tensor, dtime_tensor, self.token_num_tensor, duation_tensor], \
                   torch.tensor(self.data.iloc[index].score), intensity_tensor

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

def ctlstm_dataloader():
    '''
    Dataloader for ctlstm mainly. Perhaps, it can be used against another suitable models.
    '''
    return [CTLSTMDataset, read_data]