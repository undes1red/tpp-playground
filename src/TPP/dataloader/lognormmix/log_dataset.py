import os
import torch.utils as utils
import numpy as np

from src.toolbox.misc import load_from_pkl


def concatenate(per_line, item1 = np.array([]), item2 = np.array([])):
    return np.concatenate([item1, per_line, item2])


class LogNormDataset(utils.data.Dataset):
    '''
    Self defined dataset. The required pandas DataFrame are listed in start.py.
    But...what can we do if we need prediction? It is strange.
    '''

    def __init__(self, data, device, property_dict, input_norm_data = False, evaluate = False, shift = True):
        super(LogNormDataset, self).__init__()
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
        '''
        self.data.time_seq = self.data.time_seq.apply(np.array, dtype = np.float32)
        self.data.score = self.data.score.apply(np.array, dtype = np.float32)
        self.data.event = self.data.event.apply(np.array, dtype = np.int32)
        self.data.intensity = self.data.intensity.apply(np.array, dtype = np.float32)
        '''
        self.time_seq = data['time_seq']
        self.score = data['score']
        self.intensity = data['intensity']
        self.event = data['event']
        self.dataset_size = len(self.time_seq)
        
        # Data preprocessing
        self.event = [concatenate(item, item2 = np.array([self.event_num])) for item in self.event]
        self.time_seq = [np.diff(item, prepend = self.start_time, append = self.end_time) + (self.epsilon if shift else 0) for item in self.time_seq]
        
        '''
        self.event = self.data.event.apply(concatenate, item2 = np.array([self.event_num]))
        self.time_seq = self.data.time_seq.apply(np.diff, prepend = self.start_time, append = self.end_time)
        self.time_seq = self.data.time_seq + (self.epsilon if shift else 0)
        '''
        
        # Data normalization
        if input_norm_data:
            time_inteval = np.array([])
            for item in self.time_seq:
                time_inteval = np.concatenate((time_inteval, item[:-1]))
            regenerated_data = np.log(time_inteval + self.epsilon)
            self.mean = regenerated_data.mean()
            self.std = regenerated_data.std()
            del time_inteval, regenerated_data
        
        '''
        Fix datatype
        '''
        self.time_seq = [np.array(seq, dtype = np.float32) for seq in self.time_seq]
        self.score = [np.array(seq, dtype = np.float32) for seq in self.score]
        self.intensity = [np.array(seq, dtype = np.float32) for seq in self.intensity]
        self.event = [np.array(seq, dtype = np.int32) for seq in self.event]


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
            event_tensor = self.event[index]
            time_tensor = self.time_seq[index]
            score = self.score[index]

            if self.evaluate:
                return event_tensor, time_tensor, score, self.intensity[index]
            else:
                return event_tensor, time_tensor, score


    def __len__(self):
        return self.dataset_size


    def data_collator(self, data):
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
            mask = np.array([1] * (item[0].size - 1) + [0] * (pad_length + 1), dtype = np.bool)
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
    try:
        for file_name in file_names:
            file, _ = file_name.split('.', 1)
            data_raw[file] = load_from_pkl(os.path.join(path, file_name), compression = 'lzma')
    except:
        raise TypeError(
            f"Wrong datafile format. Please check your data file in {path}")
    
    return data_raw


def log_dataloader():
    '''
    Dataloader for IFL model.
    '''
    return [LogNormDataset, read_data]