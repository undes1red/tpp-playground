import torch
import torch.utils as utils
import os
import pandas as pd
import numpy as np
from ..utils import move_data_to_the_correct_device


def concate(per_line, item1 = np.array([]), item2 = np.array([])):
    return np.concatenate([item1, per_line, item2])

class CTLSTMDataset(utils.data.Dataset):
    '''
    Self defined dataset. The required pandas DataFrame are listed in start.py.
    But...what can we do if we need prediction? It is strange.

    event_number is the number of all existing events, dummy events are not included.

    You can set event_number = 1 to mask all events by forcing them to have the same label.
    '''

    def __init__(self, data, device, num_events, plot = False):
        super(CTLSTMDataset, self).__init__()
        self.data = data
        self.device = device
        # All input data has the same sequence length. This sequence length does not contain special start and end events.
        # Update: 2022-02-22: This assumption does not hold anymore.
        # self.sequence_length = len(self.data.iloc[0].time_seq)
        # self.token_num_tensor = torch.tensor([self.sequence_length], device = self.device)
        self.event_number = num_events
        self.plot = plot

        # Data preprocessing
        # Add dummy start and end events.
        # [self.event_number(start dummy events), ...(events), self.event_number + 1(end dummy events)]
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
        self.data.score = self.data.score.apply(np.array, dtype = np.float32)

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
            event_tensor = self.data.iloc[index].event
            dtime_tensor = self.data.iloc[index].time_seq
            duation_tensor = self.data.iloc[index].duation
            score_tensor = self.data.iloc[index].score
            if self.plot:
                intensity_tensor = self.data.iloc[index].intensity
            
            if self.plot:
                return dtime_tensor, event_tensor, score_tensor, duation_tensor, intensity_tensor
            else:
                return dtime_tensor, event_tensor, score_tensor, duation_tensor

    def __len__(self):
        return self.data.shape[0]

    def __call__(self, data):
        '''
        The structure of data:
        [
            (time_seq, event, score, intensity if self.plot else it doesn't exist at all.)
        ]
        '''
        max_length_of_this_batch = max([item[0].size for item in data])
        mask = []
        padded_data = []
        for item in data:
            pad_length = max_length_of_this_batch - item[0].size
            mask = np.array([1] * item[0].size + [0] * pad_length)
            padded_time_seq = np.pad(item[0], (0, pad_length), mode = 'mean')
            padded_event = np.pad(item[1], (0, pad_length), mode = 'minimum')
            padded_score = np.pad(item[2], (0, pad_length), mode = 'constant', constant_values = 0)
            padded_item = [[padded_event, padded_time_seq, np.array(max_length_of_this_batch), item[3], mask], padded_score]
            if self.plot:
                padded_intensity = np.pad(item[-1], (0, pad_length), mode = 'constant', constant_values = 0)
                padded_item.append(padded_intensity)
            
            padded_data.append(tuple(padded_item))
        
        from torch.utils.data._utils.collate import default_collate
        padded_data = default_collate(padded_data)
        if self.plot:
            move = move_data_to_the_correct_device(device = self.device)
            padded_data = move.move_to_device(padded_data)
        
        return padded_data


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