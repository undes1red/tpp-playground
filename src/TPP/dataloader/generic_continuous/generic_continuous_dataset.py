import torch.utils as utils
import os
import pandas as pd
import numpy as np


def prepend(per_line, number):
    return np.concatenate([np.array([number]), per_line])


def append(per_line, number):
    return np.concatenate([per_line, np.array([number])])


def diff(per_line, prepend = np._NoValue, append = np._NoValue):
    '''
    Avoid potential 0 output.
    '''
    return np.diff(per_line, prepend = prepend, append = append)


class generic_continuous_dataset(utils.data.Dataset):
    '''
    Self defined dataset. The required pandas DataFrame are listed in start.py.
    But...what can we do if we need prediction? It is strange.
    '''
    def __init__(self, data, device, property_dict, evaluate = False, shift_time = False, norm_time = False, norm_coordinate = False):
        super(generic_continuous_dataset, self).__init__()
        self.data = data
        self.device = device
        self.evaluate = evaluate
        self.dim_events = property_dict['dim_events']
        self.start_time = property_dict['t_0']
        self.end_time = property_dict['T']
        self.mean_time = property_dict['mean_time'] if norm_time else 0
        self.std_time = property_dict['std_time'] if norm_time else 1
        self.mean_coordinate = np.array(property_dict['mean_coordinate'], dtype = np.float32) if norm_coordinate else np.array([0.0] * self.dim_events, dtype = np.float32)
        self.std_coordinate = np.array(property_dict['std_coordinate'], dtype = np.float32) if norm_coordinate else np.array([1.0] * self.dim_events, dtype = np.float32)


        '''
        Convert data from list to np.array.
        '''
        self.data.time_seq = self.data.time_seq.apply(np.array, dtype = np.float32)
        self.data.score = self.data.score.apply(np.array, dtype = np.float32)
        self.data.intensity = self.data.intensity.apply(np.array, dtype = np.float32)
        self.data.event = self.data.event.apply(np.array, dtype = np.float32)

        # Data preprocessing
        # we remove the end dummy event from the sequence when evaluate = True
        if self.evaluate:
            self.data.time_seq = self.data.time_seq.apply(diff, prepend = self.start_time)
        else:
            self.data.time_seq = self.data.time_seq.apply(diff, prepend = self.start_time, append = self.end_time)
            self.data.event = self.data.event.apply(append, number = self.mean_coordinate)

        self.data.time_seq = self.data.time_seq + (1e-30 if shift_time else 0)
        self.data.time_seq = self.data.time_seq.apply(prepend, number = 0)
        self.data.event = self.data.event.apply(prepend, number = self.mean_coordinate)

        self.data.time_seq = self.data.time_seq.apply(np.array, dtype = np.float32)
        self.data.score = self.data.score.apply(np.array, dtype = np.float32)
        self.data.intensity = self.data.intensity.apply(np.array, dtype = np.float32)
        self.data.event = self.data.event.apply(np.array, dtype = np.float32)


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
            if self.evaluate:
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
        ], (mean, std)
        '''
        max_length_of_this_batch = max([item[0].size for item in data])
        mask = []
        padded_data = []
        for item in data:
            pad_length = max_length_of_this_batch - item[0].size
            mask = np.array([1] * item[0].size + [0] * pad_length)
            padded_time_seq = np.pad(item[0], (0, pad_length), mode = 'mean')
            padded_event = np.pad(item[1], ((0, pad_length), (0, 0)), mode = 'constant', constant_values = [0] * self.dim_events)
            padded_score = np.pad(item[2], (0, pad_length), mode = 'constant', constant_values = 0)
            padded_item = [padded_time_seq, np.array_split(padded_event, self.dim_events, axis = -1), padded_score, mask]
            if self.evaluate:
                padded_intensity = np.pad(item[3], (0, pad_length), mode = 'constant', constant_values = 0)
                padded_item.append(padded_intensity)
            
            padded_data.append(tuple(padded_item))
        
        from torch.utils.data._utils.collate import default_collate
        padded_data = default_collate(padded_data)
        
        return padded_data, (self.mean_coordinate, self.std_coordinate), (self.mean_time, self.std_time)


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