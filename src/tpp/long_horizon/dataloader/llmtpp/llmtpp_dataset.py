import torch.utils as utils
import os
import numpy as np


class llmtpp_dataset(utils.data.Dataset):
    '''
    Self defined dataset. The required pandas DataFrame are listed in start.py.
    But...what can we do if we need prediction? It is strange.
    '''
    def __init__(self, data, device, property_dict, shift = False, input_norm_data = False):
        super(llmtpp_dataset, self).__init__()
        self.device = device
        self.number_of_events = property_dict['num_events']
        self.lh_length = property_dict['hypro_length']
        self.mean = property_dict['mean'] if input_norm_data else 0
        self.std = property_dict['std'] if input_norm_data else 1

        '''
        Convert data from list to np.array.
        '''
        self.real_time = data['input_time']
        self.real_events = data['input_events']
        
        '''
        self.sampled_time = data['tau_sampled']
        self.sampled_events = data['events_sampled']
        '''
        
        self.real_time = [data + (1e-30 if shift else 0) for data in self.real_time]
        '''
        self.sampled_time = [data + (1e-30 if shift else 0) for data in self.sampled_time]
        '''
        
        for seq_idx in range(len(self.real_time)):
            self.real_time[seq_idx][:, 0] = 0
            '''
            self.sampled_time[seq_idx][:, 0] = 0
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
            '''
            return [self.real_time[index], self.real_events[index]], \
                   [self.sampled_time[index], self.sampled_events[index]]
            '''
            return [self.real_time[index][:, :-self.lh_length], self.real_events[index][:, :-self.lh_length], \
                    self.real_time[index][:, -self.lh_length:], self.real_events[index][:, -self.lh_length:]]


    def __len__(self):
        return len(self.real_time)
    
    
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
        
        for item in data:
            history_time, history_events, observed_time, observed_events = item
            history_time = np.array(history_time).squeeze()
            history_events = np.array(history_events).squeeze()
            observed_time = np.array(observed_time).squeeze()
            observed_events = np.array(observed_events).squeeze()
            
            seq_len = history_time.size
            pad_length = max_length_of_this_batch - seq_len
            
            mask = np.array([1] * seq_len + [0] * pad_length)
            padded_history_time_seq = np.pad(history_time, ((0, pad_length)), mode = 'constant', constant_values = 0)
            padded_history_events = np.pad(history_events, ((0, pad_length)), mode = 'constant', constant_values = self.number_of_events)
            
            # shuffle the data so the true sequence may appear at any position.
            padded_item = tuple([padded_history_time_seq, padded_history_events, mask, observed_time, observed_events])
            padded_data.append(padded_item)
        
        from torch.utils.data._utils.collate import default_collate
        padded_data = default_collate(padded_data)
        
        return padded_data, (self.mean, self.std)


def read_data(path, file_names):
    from src.toolbox.misc import load_from_pkl
    
    data_raw = {}
    try:
        for file_name in file_names:
            file = file_name.split('.')[0]
            combined_path = os.path.join(path, file_name)
            data_raw[file] = load_from_pkl(combined_path, 'bz2')
    except:
        raise TypeError(
            f"Wrong datafile format. Please check your data file in {path}")
    
    return data_raw


def llmtpp_dataloader():
    '''
    Synthetic dataloader for all synthetic datasets.
    '''
    return [llmtpp_dataset, read_data]