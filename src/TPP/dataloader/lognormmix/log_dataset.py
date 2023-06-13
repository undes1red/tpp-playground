import os
import torch.utils as utils
import pandas as pd
import numpy as np


def concatenate(per_line, item1 = np.array([]), item2 = np.array([])):
    return np.concatenate([item1, per_line, item2])


def concatenate_shift(per_line, item1, item2):
    return np.concatenate([item1 + per_line[0] - 1, per_line, item2 + item1 + per_line[0] - 1])


class LogNormDataset(utils.data.Dataset):
    '''
    Self defined dataset. The required pandas DataFrame are listed in start.py.
    But...what can we do if we need prediction? It is strange.
    '''

    def __init__(self, data, device, property_dict, input_norm_data = True, evaluate = False, shift = False):
        super(LogNormDataset, self).__init__()
        self.data = data
        self.device = device
        self.evaluate = evaluate
        self.start_time = property_dict['t_0']
        self.end_time = property_dict['T']
        self.event_num = property_dict['num_events']
        self.shift = shift
        self.input_norm_data = input_norm_data
        self.mean = 0
        self.std = 1
        self.epsilon = 1e-30

        '''
        Convert data from list to np.array.
        '''
        self.data.time_seq = self.data.time_seq.apply(np.array, dtype = np.float32)
        self.data.score = self.data.score.apply(np.array, dtype = np.float32)
        self.data.event = self.data.event.apply(np.array, dtype = np.int32)
        self.data.intensity = self.data.intensity.apply(np.array, dtype = np.float32)

        # Data preprocessing
        self.data.event = self.data.event.apply(concatenate, item2 = np.array([self.event_num]))
        self.data.time_seq = self.data.time_seq.apply(np.diff,  prepend = self.start_time, append = self.end_time) + (self.epsilon if shift else 0)

        # Data normalization
        if input_norm_data:
            time_inteval = np.array([])
            for item in self.data['time_seq'].values.tolist():
                time_inteval = np.concatenate((time_inteval, item[:-1]))
            regenerated_data = np.log(time_inteval + self.epsilon)
            self.mean = regenerated_data.mean()
            self.std = regenerated_data.std()
        
        '''
        Fix datatype
        '''
        self.data.time_seq = self.data.time_seq.apply(np.array, dtype = np.float32)
        self.data.score = self.data.score.apply(np.array, dtype = np.float32)
        self.data.event = self.data.event.apply(np.array, dtype = np.int32)
        self.data.intensity = self.data.intensity.apply(np.array, dtype = np.float32)


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
            event_tensor = self.data.iloc[index].event
            time_tensor = self.data.iloc[index].time_seq
            score = self.data.iloc[index].score

            if self.evaluate:
                return event_tensor, time_tensor, score, self.data.iloc[index].intensity
            else:
                return event_tensor, time_tensor, score


    def __len__(self):
        return self.data.shape[0]


    def __call__(self, data):
        '''
        The custom collate_fn() for IFL datasets.
        data: [event_tensor, time_tensor, score, self.data.iloc[index].intensity if self.evaluate]
        '''
        max_length_of_this_batch = max([item[0].size for item in data])
        mask = []
        padded_data = []

        for item in data:
            pad_length = max_length_of_this_batch - item[0].size
            '''
            The final dummy event should be excluded.
            '''
            mask = np.array([1] * (item[0].size - 1) + [0] * (pad_length + 1))
            padded_time_seq = np.pad(item[1], (0, pad_length), mode = 'mean')
            padded_event = np.pad(item[0], (0, pad_length), mode = 'minimum')
            padded_score = np.pad(item[2], (0, pad_length), mode = 'constant', constant_values = 0)
            padded_item = [padded_event, padded_time_seq, padded_score, mask]
            if self.evaluate:
                intensity = np.pad(item[3], (0, pad_length), mode = 'constant', constant_values = 0)
                padded_item.append(intensity)

            padded_data.append(tuple(padded_item))
            
        from torch.utils.data._utils.collate import default_collate
        padded_data = default_collate(padded_data)

        return padded_data, (self.mean, self.std) if self.input_norm_data else None


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


def log_dataloader():
    '''
    Dataloader for IFL model.
    '''
    return [LogNormDataset, read_data]