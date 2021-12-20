import torch
import torch.utils as utils
import os
import pandas as pd
import numpy as np

def concate(per_line, item1 = np.array([]), item2 = np.array([])):
    return np.concatenate([item1, per_line, item2])

def concate_shift(per_line, item1, item2):
    return np.concatenate([item1 + per_line[0] - 1, per_line, item2 + item1 + per_line[0] - 1])


class IflDataset(utils.data.Dataset):
    '''
    Self defined dataset. The required pandas DataFrame are listed in train.py.
    But...what can we do if we need prediction? It is strange.
    '''

    def __init__(self, data, device, event_num = 10, have_mask = False, start_time: int = None, end_time: int = None, input_norm_data = True, shift = False):
        super(IflDataset, self).__init__()
        self.data = data
        self.device = device
        # All input data has the same sequence length.
        # Use shift if the event sequences don't start at timestamp 0.
        # If enabled, the time interval between the first event and the start will always be 1s.
        self.sequence_length = len(self.data.iloc[0].time_seq)
        self.start_time = start_time if start_time else 0
        self.end_time = end_time if end_time else 350
        self.input_norm_data = input_norm_data
        self.event_num = event_num
        self.have_mask = have_mask

        # Data normalization
        if input_norm_data:
            regenerated_data = pd.DataFrame(self.data['time_seq'].values.tolist())
            regenerated_data.insert(0, 'start', self.start_time)
            regenerated_data.insert(regenerated_data.columns.size, 'end', self.end_time)
            regenerated_data = np.log(regenerated_data.diff(axis = 1) + 1e-8).stack()
            self.mean = regenerated_data.mean()
            self.var = regenerated_data.var()

        # Data preprocessing
        self.data.event = self.data.event.apply(concate, item2 = np.array([self.event_num]))
        if shift:
            self.data.time_seq = self.data.time_seq.apply(concate_shift, item1 = np.array([self.start_time]), item2 = np.array([self.start_time]))
        else:
            self.data.time_seq = self.data.time_seq.apply(concate, item1 = np.array([self.start_time]), item2 = np.array([self.end_time]))
        self.data.time_seq = self.data.time_seq.apply(np.diff) + 1e-5

        # pd.DataFrame already has a method called 'mask', so self.data.mask will fail and
        # one should use self.data['mask'] instead.
        if self.have_mask:
            self.data['mask'] = self.data['mask'].apply(concate, item2 = np.zeros(1))
        else:
            self.data['mask'] = [np.concatenate((np.ones(self.sequence_length), np.zeros(1)))] * self.data.time_seq.size
        
        self.data.time_seq = self.data.time_seq.apply(np.array, dtype = np.float32)
        self.data.score = self.data.score.apply(np.array, dtype = np.float32)
        self.data.event = self.data.event.apply(np.array, dtype = np.int32)
        self.data['mask'] = self.data['mask'].apply(np.array, dtype = np.int8)

    def __getitem__(self, index):
        # score is the global fact. So we need to modify the first part of the minibatch
        if isinstance(index, slice):
            return [
                self[idx] for idx in range(index.start or 0, index.stop or len(self), index.step or 1)
            ]
        else:
            '''
            Based on the ifl model documents, the dataset has following parts
            1. event_tensor: Data tensors that contain event marks for available sequences.
            2. time_tensor: The timestamp of each event in relative style.
            3. mask_tensor: Tensors to mask out dummy events(under such circumstance the only dummy event is the last "process end" one.)
            4. mean
            5. var: these two variables are used for input data normalization.

            Seems that t_start and t_end are fixed and stay unchanged unless the dataset get changed.
            finally we should tell the model how many event types the dataset has.(Maybe this can be a model hyperparameter)
            '''
            event_tensor = torch.from_numpy(self.data.iloc[index].event)
            time_tensor = torch.from_numpy(self.data.iloc[index].time_seq)
            mask_tensor = torch.from_numpy(self.data.iloc[index]['mask'])

            return [event_tensor, time_tensor, mask_tensor, self.mean, self.var] if self.input_norm_data else [event_tensor, time_tensor, mask_tensor], \
                   torch.from_numpy(self.data.iloc[index].score)

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