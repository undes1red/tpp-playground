import torch
import torch.utils as utils
import os
import numpy as np

from src.toolbox.misc import load_from_pkl
from src.MDI.dataloader.missing.utils import *


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


class missing_dataset(utils.data.Dataset):
    '''
    Self defined dataset. The required pandas DataFrame are listed in start.py.
    But...what can we do if we need prediction? It is strange.
    '''
    def __init__(self, data, device, property_dict, missing_probability = [], num_of_missing_sample = 16, \
                 shift = False, input_norm_data = False):
        super(missing_dataset, self).__init__()
        self.device = device
        # The dummy events, with event_id = self.number_of_events, are always here.
        self.number_of_events = property_dict['num_events']
        self.start_time = property_dict['t_0']
        self.end_time = property_dict['T']
        self.mean = property_dict['mean'] if input_norm_data else 0
        self.std = property_dict['std'] if input_norm_data else 1
        
        '''
        od_data_dict = {
            'complete_forward': padded_data,
            'complete_backward': padded_backward_data,
            'observed_forward': padded_obs_data,
            'observed_backward': padded_backward_obs_data,
        }
        '''
        
        '''
        Convert data from list to np.array.
        '''
        self.complete_forward = data['complete_forward']
        self.complete_backward = data['complete_backward']
        self.observed_forward = data['observed_forward']
        self.observed_backward = data['observed_backward']
        
        assert len(self.complete_forward) == len(self.complete_backward) == len(self.observed_forward) == len(self.observed_backward), 'Dataset size mismatches!'
        self.dataset_size = len(self.complete_forward)


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
            return self.complete_forward[index], \
                   self.complete_backward[index], \
                   self.observed_forward[index], \
                   self.observed_backward[index]


    def __len__(self):
        return self.dataset_size
    
    
    def data_collator(self, data):
        '''
        The structure of data:
        [
            (time_seq, event, score, mask, intensity if self.evaluate else it doesn't exist at all.)
        ], (mean, var)
        '''
        max_length_of_this_batch = max([item[0][0].size for item in data])
        mask = []
        padded_data = []
        padded_backward_data = []
        padded_obs_data = []
        padded_backward_obs_event_seq = []
        
        for item in data:
            pad_length = max_length_of_this_batch - item[0][0].size
            mask = np.array([1] * item[0][0].size + [0] * pad_length)
            
            # The complete sequence, forward.
            # item[0] contains the complete forward sequence.
            # item[0][0]: complete time seq
            # item[0][1]: complete event seq
            padded_time_seq = np.pad(item[0][0], (0, pad_length), mode = 'constant', constant_values = 0)
            padded_event = np.pad(item[0][1], (0, pad_length), mode = 'constant', constant_values = self.number_of_events)
            padded_item = [padded_time_seq, padded_event, mask]

            # The complete sequence, backward.
            # item[1] contains the complete forward sequence.
            # item[1][0]: complete time seq
            # item[1][1]: complete event seq
            padded_backward_time_seq = np.pad(item[1][0], (0, pad_length), mode = 'constant', constant_values = 0)
            padded_backward_event = np.pad(item[1][1], (0, pad_length), mode = 'constant', constant_values = self.number_of_events)
            backward_padded_item = [padded_backward_time_seq, padded_backward_event, mask]
            
            padded_data.append(tuple(padded_item))
            padded_backward_data.append(tuple(backward_padded_item))
            
            # The observed sequence, forward.
            # item[2] contains the observed forward sequence.
            # item[2][0]: observed time seq
            # item[2][1]: observed event seq
            # item[2][2]: the padding mask of observed seq
            # item[2][3]: missing_mask, showing which events in the complete forward event seq are missing.
            # item[2][4]: log_censor_prob, the probability of this type of missing occurring.
            padded_obs_data.append(item[2])

            # The observed sequence, backward.
            # item[3] contains the observed backward sequence.
            # item[3][0]: observed time seq
            # item[3][1]: observed event seq
            # item[3][2]: the padding mask of observed seq
            # item[3][3]: missing_mask, showing which events in the complete backward event seq are missing.
            # item[3][4]: log_censor_prob, the probability of this type of missing occurring.
            padded_backward_obs_event_seq.append(item[3])
               
        from torch.utils.data._utils.collate import default_collate
        padded_data = default_collate(padded_data)
        padded_backward_data = default_collate(padded_backward_data)
        # What you get is a bunch of [batch_size, seq_len] tensors.
        padded_obs_data = [(torch.from_numpy(per_item[0]), torch.from_numpy(per_item[1]), torch.from_numpy(per_item[2]), torch.from_numpy(per_item[3]), torch.from_numpy(per_item[4])) for per_item in padded_obs_data]
        padded_backward_obs_event_seq = [(torch.from_numpy(per_item[0]), torch.from_numpy(per_item[1]), torch.from_numpy(per_item[2]), torch.from_numpy(per_item[3].copy()), torch.from_numpy(per_item[4])) for per_item in padded_backward_obs_event_seq]

        return padded_data, padded_backward_data, padded_obs_data, padded_backward_obs_event_seq, (self.mean, self.std)


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


def missing_dataloader():
    '''
    Synthetic dataloader for all synthetic datasets.
    '''
    return [missing_dataset, read_data]