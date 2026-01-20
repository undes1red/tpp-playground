import torch
import torch.utils as utils
import os
import numpy as np

from src.toolbox.misc import load_from_pkl
from src.tpp.dataloader.od_generic.utils import *


def prepend(per_line, number):
    return np.concatenate([np.array([number]), per_line])


def append(per_line, number):
    return np.concatenate([per_line, np.array([number])])


def head_and_tail(per_line, head, tail):
    return np.concatenate([np.array([head]), per_line, np.array([tail])])


def diff(per_line, prepend = np._NoValue, append = np._NoValue):
    '''
    Avoid potential 0 output.
    '''
    return np.diff(per_line, prepend = prepend, append = append)


class commission_dataset(utils.data.Dataset):
    '''
    Self defined dataset. The required pandas DataFrame are listed in start.py.
    But...what can we do if we need prediction? It is strange.
    '''
    def __init__(self, data, device, property_dict, shift = False, input_norm_data = False):
        super(commission_dataset, self).__init__()
        self.device = device
        # The dummy events, with event_id = self.number_of_events, are always here.
        self.number_of_events = property_dict['num_events']
        self.start_time = property_dict['t_0']
        self.end_time = property_dict['T']
        self.mean = property_dict['mean'] if input_norm_data else 0
        self.std = property_dict['std'] if input_norm_data else 1
        
        '''
        od_data_dict = {
            'complete_forward': original data,
            'commissions': outliers that should inserted into the event sequence
        }
        '''
        
        '''
        Convert data from list to np.array.
        '''
        self.events = data['events']
        self.time_seq = data['time_seq']
        self.commission = data['commission']
        
        assert len(self.events) == len(self.time_seq) == len(self.commission), 'Dataset size mismatches!'
        self.dataset_size = len(self.events)
        
        # Data preprocessing
        # we remove the end dummy event from the sequence when evaluate = True

        # Use T
        # self.data.time_seq = self.data.time_seq.apply(diff, prepend = self.start_time, append = self.end_time)
        # Do not use T
        self.time_seq = [np.diff(seq, prepend = self.start_time) for seq in self.time_seq]
        self.time_seq = [append(seq, 0.1) for seq in self.time_seq]
        self.time_seq = [seq + (1e-30 if shift else 0) for seq in self.time_seq]
        self.time_seq = [prepend(seq, 0) for seq in self.time_seq]
        self.events = [head_and_tail(seq, self.number_of_events, self.number_of_events) for seq in self.events]
        self.commission = [head_and_tail(seq, 0, 0) for seq in self.commission]

        '''
        self.data.time_seq = self.data.time_seq.apply(diff, prepend = self.start_time)
        self.data.time_seq = self.data.time_seq.apply(append, number = 0.1)
        self.data.event = self.data.event.apply(append, number = self.number_of_events)
        '''


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
            return self.events[index], \
                   self.time_seq[index], \
                   self.commission[index]


    def __len__(self):
        return self.dataset_size
    

    def data_collator(self, data):
        '''
        The structure of data:
        [
            (time_seq, event, score, mask, intensity if self.evaluate else it doesn't exist at all.)
        ], (mean, var)
        '''
        max_length_of_this_batch = max([len(item[0]) for item in data])
        mask = []
        padded_data = []
        for item in data:
            pad_length = max_length_of_this_batch - len(item[0])
            mask = np.array([1] * len(item[0]) + [0] * pad_length, dtype = np.bool)
            padded_event = np.pad(item[0], (0, pad_length), mode = 'constant', constant_values = 0)
            padded_time_seq = np.pad(item[1], (0, pad_length), mode = 'constant', constant_values = self.number_of_events)
            padded_commission = np.pad(item[2], (0, pad_length), mode = 'constant', constant_values = 0)
            padded_item = [padded_time_seq, padded_event, padded_commission, mask]
            
            padded_data.append(tuple(padded_item))
        
        from torch.utils.data._utils.collate import default_collate
        padded_data = default_collate(padded_data)
        
        return padded_data, (self.mean, self.std)


def read_data(path, file_names):
    data_raw = {}
    try:
        for file_name in file_names:
            file = file_name.split('.')[0]
            combined_path = os.path.join(path, file_name)
            data_raw[file] = load_from_pkl(combined_path, 'lzma')
    except:
        raise TypeError(
            f"Wrong datafile format. Please check your data file in {path}")
    
    return data_raw


def commission_dataloader():
    '''
    Synthetic dataloader for all synthetic datasets.
    '''
    return [commission_dataset, read_data]