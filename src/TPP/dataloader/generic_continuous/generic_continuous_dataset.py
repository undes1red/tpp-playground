import torch.utils as utils
import os
import pandas as pd
import numpy as np
from src.TPP.dataloader.utils import move_data_to_the_correct_device


def insert(per_line, number):
    return np.concatenate([np.array([number]), per_line])


def diff(per_line, shift):
    '''
    Avoid potential 0 output.
    '''
    return np.diff(per_line) + (1e-6 if shift else 0)


class generic_continuous_dataset(utils.data.Dataset):
    '''
    Self defined dataset. The required pandas DataFrame are listed in train.py.
    But...what can we do if we need prediction? It is strange.
    '''
    def __init__(self, data, device, num_events, plot = False, shift = False, shift_time = False, input_norm_data = False):
        super(generic_continuous_dataset, self).__init__()
        self.data = data
        self.device = device
        self.plot = plot
        self.number_of_events = num_events
        self.mean = 0
        self.var = 1

        # Data preprocessing
        if shift_time:
            '''
            Current stackoverflow specific
            '''
            for idx, item in enumerate(self.data.time_seq):
                first_event_abs_time = item[0]
                self.data.time_seq[idx].insert(0, first_event_abs_time - 0.8)
        else:
            self.data.time_seq = self.data.time_seq.apply(insert, number = 0)

        self.data.time_seq = self.data.time_seq.apply(diff, shift = shift)
        # if input_norm_data:
        #     self.data.time_seq = self.data.time_seq.apply(math.log)
        self.data.time_seq = self.data.time_seq.apply(insert, number = 0)
        self.data.event = self.data.event.apply(insert, number = [0, 0])

        # Data normalization
        # We need it because several datasets' inputs are just so huge that several model can never handle it.
        if input_norm_data:
            regenerated_data = pd.DataFrame(self.data['time_seq'].values.tolist())
            regenerated_data = (regenerated_data + 1e-8).stack()
            self.mean = regenerated_data.mean()
            self.var = regenerated_data.std()

        self.data.time_seq = self.data.time_seq.apply(np.array, dtype = np.float32)
        self.data.score = self.data.score.apply(np.array, dtype = np.float32)
        self.data.intensity = self.data.intensity.apply(np.array, dtype = np.float32)
        self.data.event = self.data.event.apply(np.array, dtype = np.int32)


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
                return self.data.iloc[index].time_seq, \
                       self.data.iloc[index].event, \
                       self.data.iloc[index].score,\
                       self.data.iloc[index].intensity
            else:
                return self.data.iloc[index].time_seq, \
                       self.data.iloc[index].event, \
                       self.data.iloc[index].score


    def __len__(self):
        return self.data.shape[0]
    
    
    def __call__(self, data):
        '''
        The structure of data:
        [
            (time_seq, event, score, mask, intensity if self.plot else it doesn't exist at all.)
        ], (mean, var)
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
            padded_item = [padded_time_seq, padded_event, padded_score, mask]
            if self.plot:
                padded_intensity = np.pad(item[3], (0, pad_length), mode = 'constant', constant_values = 0)
                padded_item.append(padded_intensity)
            
            padded_data.append(tuple(padded_item))
        
        from torch.utils.data._utils.collate import default_collate
        padded_data = default_collate(padded_data)
        if self.plot:
            move = move_data_to_the_correct_device(device = self.device)
            padded_data = move.move_to_device(padded_data)
        
        return padded_data, (self.mean, self.var)


def read_data(path, file_names):
    data_raw = {}
    try:
        for file_name in file_names:
            file, _ = file_name.split('.')
            data_raw[file] = pd.read_json(
                os.path.join(path, file_name))
    except:
        raise TypeError(
            f"Wrong datafile format. Please check your data file in {path}")
    
    return data_raw


def generic_continuous_dataloader():
    '''
    Synthetic dataloader for all synthetic datasets.
    '''
    return [generic_continuous_dataset, read_data]