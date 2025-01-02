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
    def __init__(self, data, device, property_dict, missing_probability, num_of_missing_sample = 16, \
                 shift = False, input_norm_data = False):
        super(missing_dataset, self).__init__()
        self.device = device
        # The dummy events, with event_id = self.number_of_events, are always here.
        self.missing_probability = np.array(missing_probability + [0.0])
        self.num_of_missing_sample = num_of_missing_sample
        self.number_of_events = property_dict['num_events']
        self.start_time = property_dict['t_0']
        self.end_time = property_dict['T']
        self.mean = property_dict['mean'] if input_norm_data else 0
        self.std = property_dict['std'] if input_norm_data else 1

        '''
        Convert data from list to np.array.
        '''
        self.time_seq = data['time_seq']
        self.event = data['event']
        
        assert len(self.time_seq) == len(self.event), 'Dataset size mismatches!'
        self.dataset_size = len(self.time_seq)

        self.time_seq = [np.diff(seq, prepend = self.start_time) for seq in self.time_seq]
        self.time_seq = [append(seq, 0.1) for seq in self.time_seq]
        self.time_seq = [seq + (1e-30 if shift else 0) for seq in self.time_seq]
        self.time_seq = [prepend(seq, 0) for seq in self.time_seq]
        
        self.event = [head_and_tail(seq, head = self.number_of_events, tail = self.number_of_events) for seq in self.event]

        '''
        Fix datatype
        '''
        self.time_seq = [np.array(seq, dtype = np.float64) for seq in self.time_seq]
        self.event = [np.array(seq, dtype = np.int64) for seq in self.event]


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
            return self.time_seq[index], \
                   self.event[index]


    def __len__(self):
        return self.dataset_size
    
    
    def data_collator(self, data):
        '''
        The structure of data:
        [
            (time_seq, event, score, mask, intensity if self.evaluate else it doesn't exist at all.)
        ], (mean, var)
        '''
        max_length_of_this_batch = max([item[0].size for item in data])
        mask = []
        padded_data = []
        padded_backward_data = []
        padded_obs_data = []
        padded_backward_obs_event_seq = []
        
        for item in data:
            pad_length = max_length_of_this_batch - item[0].size
            mask = np.array([1] * item[0].size + [0] * pad_length)
            
            # The complete sequence, forward.
            padded_time_seq = np.pad(item[0], (0, pad_length), mode = 'constant', constant_values = 0)
            padded_event = np.pad(item[1], (0, pad_length), mode = 'constant', constant_values = self.number_of_events)
            padded_item = [padded_time_seq, padded_event, mask]

            # The complete sequence, backward.
            backward_time_seq = np.concatenate((np.array([0.0]), np.flip(item[0][1:])))
            backward_event = np.flip(item[1])
            padded_backward_time_seq = np.pad(backward_time_seq, (0, pad_length), mode = 'constant', constant_values = 0)
            padded_backward_event = np.pad(backward_event, (0, pad_length), mode = 'constant', constant_values = self.number_of_events)
            backward_padded_item = [padded_backward_time_seq, padded_backward_event, mask]
            
            # Sample missing events according to the missing probability.
            # What you get here is a bunch of [batch_size, number_of_missing_samples, seq_len]
            missing_mask, log_censor_prob \
                = sample_particles(item[1], self.num_of_missing_sample, missing_probability = self.missing_probability)
                                                                               # [num_of_missing_sample, seq_len] + [num_of_missing_sample]
            backward_missing_mask = np.flip(missing_mask, axis = -1)           # [num_of_missing_sample, seq_len]
            
            longest_obs = missing_mask.sum(axis=1).max()
            obs_event_seq = np.empty([self.num_of_missing_sample, longest_obs], dtype = np.int64)
            obs_event_seq.fill(self.number_of_events)
            obs_time_seq = np.zeros([self.num_of_missing_sample, longest_obs])
            
            backward_obs_event_seq = np.empty([self.num_of_missing_sample, longest_obs], dtype = np.int64)
            backward_obs_event_seq.fill(self.number_of_events)
            backward_obs_time_seq = np.zeros([self.num_of_missing_sample, longest_obs])
            
            obs_mask_seq = np.zeros([self.num_of_missing_sample, longest_obs], dtype = np.int64)
            for idx, (missing_mask_per_sample, backward_missing_mask_per_sample) in enumerate(zip(missing_mask, backward_missing_mask)):
                # observed time, forward.
                cum_time = item[0].cumsum(axis = -1)
                obs_time = np.diff(cum_time[missing_mask_per_sample], axis = -1, prepend = 0)
                obs_time_seq[idx, :missing_mask_per_sample.sum()] = obs_time
                
                # observed time, backward.
                backward_obs_time = np.concatenate((np.array([0.0]), np.flip(obs_time[1:])))
                backward_obs_time_seq[idx, :missing_mask_per_sample.sum()] = backward_obs_time
                
                # the gap between the current event to the latest observed events
                # If the current event is observed, this value will be 0.
                # If the current event is missing, this value will be the gap between this event to the latest observed events in the backward sequence.
                # We avoid using cumsum() to avoid nasty float number calculation errors.
                
                
                # observed events, forward.
                obs_event_seq[idx, :missing_mask_per_sample.sum()] = item[1][missing_mask_per_sample]
                
                # observed events, backward.
                backward_obs_event_seq[idx, :missing_mask_per_sample.sum()] = np.flip(item[1][missing_mask_per_sample])
                
                # mask.
                # Mask tensors stay the same for forward and backward time and event sequences.
                obs_mask_seq[idx, :missing_mask_per_sample.sum()] = 1

            padded_data.append(tuple(padded_item))
            padded_backward_data.append(tuple(backward_padded_item))
            padded_obs_data.append(tuple([obs_time_seq, obs_event_seq, obs_mask_seq, missing_mask, log_censor_prob]))
            padded_backward_obs_event_seq.append(tuple([backward_obs_time_seq, backward_obs_event_seq, obs_mask_seq, backward_missing_mask, log_censor_prob]))
                
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