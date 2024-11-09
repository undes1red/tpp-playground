import torch.utils as utils
import os
import numpy as np

from src.toolbox.misc import load_from_pkl


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
        self.time_seq = data['time_seq']
        self.score = data['score']
        self.intensity = data['intensity']
        self.event = data['event']
        
        assert len(self.time_seq) == len(self.score) == len(self.intensity) == len(self.event), 'Dataset size mismatches!'
        self.dataset_size = len(self.time_seq)

        # Data preprocessing
        # we remove the end dummy event from the sequence when evaluate = True
        if self.evaluate:
            self.time_seq = [np.diff(seq, prepend = self.start_time) for seq in self.time_seq]
            '''
            self.data.time_seq = self.data.time_seq.apply(diff, prepend = self.start_time)
            '''
        else:
            # Use T
            # self.data.time_seq = self.data.time_seq.apply(diff, prepend = self.start_time, append = self.end_time)
            # Do not use T
            self.time_seq = [np.diff(seq, prepend = self.start_time) for seq in self.time_seq]
            self.time_seq = [append(seq, 0.1) for seq in self.time_seq]
            self.event = [append(seq, self.number_of_events) for seq in self.event]

            '''
            self.data.time_seq = self.data.time_seq.apply(diff, prepend = self.start_time)
            self.data.time_seq = self.data.time_seq.apply(append, number = 0.1)
            self.data.event = self.data.event.apply(append, number = self.number_of_events)
            '''

        self.time_seq = [seq + (1e-30 if shift else 0) for seq in self.time_seq]
        self.time_seq = [prepend(seq, 0) for seq in self.time_seq]
        self.event = [prepend(seq, self.number_of_events) for seq in self.event]
        '''
        self.time_seq = self.data.time_seq.apply(prepend, number = 0)
        self.data.event = self.data.event.apply(prepend, number = self.number_of_events)
        '''

        '''
        Fix datatype
        '''
        self.time_seq = [np.array(seq, dtype = np.float32) for seq in self.time_seq]
        self.score = [np.array(seq, dtype = np.float32) for seq in self.score]
        self.intensity = [np.array(seq, dtype = np.float32) for seq in self.intensity]
        self.event = [np.array(seq, dtype = np.int64) for seq in self.event]
        '''
        self.data.time_seq = self.data.time_seq.apply(np.array, dtype = np.float32)
        self.data.score = self.data.score.apply(np.array, dtype = np.float32)
        self.data.intensity = self.data.intensity.apply(np.array, dtype = np.float32)
        self.data.event = self.data.event.apply(np.array, dtype = np.int64)
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
            if self.evaluate:
                return self.time_seq[index], \
                       self.event[index], \
                       self.score[index],\
                       self.intensity[index]
            else:
                return self.time_seq[index], \
                       self.event[index], \
                       self.score[index]


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
        for item in data:
            pad_length = max_length_of_this_batch - item[0].size
            mask = np.array([1] * item[0].size + [0] * pad_length, dtype = np.bool)
            padded_time_seq = np.pad(item[0], (0, pad_length), mode = 'constant', constant_values = 0)
            padded_event = np.pad(item[1], (0, pad_length), mode = 'constant', constant_values = self.number_of_events)
            padded_score = np.pad(item[2], (0, pad_length), mode = 'constant', constant_values = 0)
            padded_item = [padded_time_seq, padded_event, padded_score, mask]
            if self.evaluate:
                padded_intensity = np.pad(item[3], (0, pad_length), mode = 'constant', constant_values = 0)
                padded_item.append(padded_intensity)
            
            padded_data.append(tuple(padded_item))
        
        from torch.utils.data._utils.collate import default_collate
        padded_data = default_collate(padded_data)
        
        return padded_data, (self.mean, self.std)


def read_data(path, file_names):
    data_raw = {}
    try:
        for file_name in file_names:
            file, _ = file_name.split('.', 1)
            data_raw[file] = load_from_pkl(os.path.join(path, file_name), compression = 'lzma')
    except:
        raise TypeError(
            f"Wrong datafile format. Please check your data file in {path}")
    
    return data_raw


def generic_dataloader():
    '''
    Synthetic dataloader for all synthetic datasets.
    '''
    return [generic_dataset, read_data]