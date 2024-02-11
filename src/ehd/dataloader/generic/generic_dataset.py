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


class generic_dataset(utils.data.Dataset):
    '''
    Self defined dataset. The required pandas DataFrame are listed in start.py.
    But...what can we do if we need prediction? It is strange.
    '''
    def __init__(self, data, device, property_dict, evaluate = False, shift = False, input_norm_data = False):
        super(generic_dataset, self).__init__()
        self.data = data
        self.device = device
        self.evaluate = evaluate
        self.number_of_events = property_dict['num_events']
        self.start_time = property_dict['t_0']
        self.end_time = property_dict['T']
        self.mean = property_dict['mean'] if input_norm_data else 0
        self.std = property_dict['std'] if input_norm_data else 1

        '''
        Convert data from list to np.array.
        '''
        self.data.time_seq = self.data.time_seq.apply(np.array, dtype = np.float32)
        self.data.score = self.data.score.apply(np.array, dtype = np.float32)
        self.data.intensity = self.data.intensity.apply(np.array, dtype = np.float32)
        self.data.event = self.data.event.apply(np.array, dtype = np.int64)

        # Data preprocessing
        self.data.time_seq = self.data.time_seq + (1e-30 if shift else 0)

        self.data.event = self.data.event.apply(append, number = self.number_of_events)
        self.data.time_seq = self.data.time_seq.apply(append, number = 0)

        self.data.event = self.data.event.apply(prepend, number = self.number_of_events)
        self.data.time_seq = self.data.time_seq.apply(prepend, number = 0)

        self.data['mask'] = self.data['mask'].apply(append, number = 1)
        self.data['mask'] = self.data['mask'].apply(prepend, number = 1)

        '''
        Fix datatype
        '''
        self.data.time_seq = self.data.time_seq.apply(np.array, dtype = np.float32)
        self.data.score = self.data.score.apply(np.array, dtype = np.float32)
        self.data.intensity = self.data.intensity.apply(np.array, dtype = np.float32)
        self.data.event = self.data.event.apply(np.array, dtype = np.int64)
        self.data['mask'] = self.data['mask'].apply(np.array, dtype = np.int32)


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
            return self.data.iloc[index].time_seq, \
                   self.data.iloc[index].event, \
                   self.data.iloc[index].score, \
                   self.data.iloc[index]['mask']


    def __len__(self):
        return self.data.shape[0]
    
    
    def data_collator(self, data):
        '''
        The structure of data:
        [
            (time_seq, event, score, mask, intensity if self.evaluate else it doesn't exist at all.)
        ], (mean, var)
        '''
        from torch.utils.data._utils.collate import default_collate
        data = default_collate(data)
        
        return data, (self.mean, self.std)


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


def generic_dataloader():
    '''
    Synthetic dataloader for all synthetic datasets.
    '''
    return [generic_dataset, read_data]