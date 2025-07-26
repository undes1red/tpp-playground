import torch.utils as utils
import os
import numpy as np

import datetime
from src.toolbox.misc import load_from_pkl


def prepend(per_line, number):
    return np.concatenate([np.array([number]), per_line])


def append(per_line, number):
    return np.concatenate([per_line, np.array([number])])


def diff(per_line, prepend = np._NoValue, append = np._NoValue):
    # Avoid potential 0 output.
    return np.diff(per_line, prepend = prepend, append = append)


def shift_the_first_absolute_time(original_time_seq, shift_time):
    # We pick the first original time and shift it by shift_time unit as the absolute time of the start dummy event.
    last_time = datetime.datetime.strptime(original_time_seq[0], '%y-%m-%d %H:%M:%S')
    time_delta = datetime.timedelta(days = shift_time[0])
    shifted_time = last_time - time_delta
    time_str = shifted_time.strftime('%y-%m-%d %H:%M:%S')
    return time_str


def shift_the_last_absolute_time(original_time_seq):
    # We pick the last original time and shift it by 0.1 unit.
    last_time = datetime.datetime.strptime(original_time_seq[-1], '%y-%m-%d %H:%M:%S')
    time_delta = datetime.timedelta(days = 0.1)
    shifted_time = last_time + time_delta
    time_str = shifted_time.strftime('%y-%m-%d %H:%M:%S')
    return time_str
    

class generic_dataset(utils.data.Dataset):
    '''
    Self defined dataset. The required pandas DataFrame are listed in start.py.
    But...what can we do if we need prediction? It is strange.
    '''
    def __init__(self, data, device, property_dict, relative_time = True, evaluate = False, shift = False, input_norm_data = False):
        super(generic_dataset, self).__init__()
        self.device = device
        self.evaluate = evaluate
        self.number_of_events = property_dict['num_events']
        self.start_time = property_dict['t_0']
        self.end_time = property_dict['T']
        self.embedding_size = property_dict['embedding_size']
        self.mean = property_dict['mean'] if input_norm_data else 0
        self.std = property_dict['std'] if input_norm_data else 1

        # Convert data from list to np.array
        # relative time and absolute time
        self.time_seq = data['time_seq']
        self.original_time_seq = data['original_time_seq']
        
        # event marks.
        self.event = data['event']
        # original marks.
        self.original_event = data['original_event']
        
        # the note for each event. This note should be concise.
        self.note = data['text']
        self.embeddings = data['embeddings']
        
        # metadata of each sequence. Maybe useful during evaluation.
        self.metadata = data['metadata']

        # things only useful on experiments on synthetic data.
        self.score = data['score']
        self.intensity = data['intensity']
        
        assert len(self.time_seq) == len(self.original_time_seq) == \
               len(self.event) == len(self.note) == \
               len(self.score) == len(self.intensity), 'Dataset size mismatches!'
        self.dataset_size = len(self.time_seq)

        # Data preprocessing
        # we remove the end dummy event from the sequence when evaluate = True
        if relative_time:
            if self.evaluate:
                self.time_seq = [np.diff(seq, prepend = self.start_time) for seq in self.time_seq]
            else:
                # Use T
                # self.data.time_seq = self.data.time_seq.apply(diff, prepend = self.start_time, append = self.end_time)
                # Do not use T
                self.time_seq = [np.diff(seq, prepend = self.start_time) for seq in self.time_seq]
                self.time_seq = [append(seq, 0.1) for seq in self.time_seq]
                self.original_time_seq = [append(seq, shift_the_last_absolute_time(seq)) for seq in self.original_time_seq]
                
                self.event = [append(seq, self.number_of_events) for seq in self.event]
                self.original_event = [append(seq, ' ') for seq in self.original_event]
                self.note = [append(seq, 'This is the end of the event sequence.') for seq in self.note]
                self.embeddings = [append(seq, [0] * self.embedding_size) for seq in self.embeddings]
                self.score = [append(seq, 0) for seq in self.score]
                '''
                self.data.time_seq = self.data.time_seq.apply(diff, prepend = self.start_time)
                self.data.time_seq = self.data.time_seq.apply(append, number = 0.1)
                self.data.event = self.data.event.apply(append, number = self.number_of_events)
                '''

            self.time_seq = [seq + (1e-30 if shift else 0) for seq in self.time_seq]
            self.original_time_seq = [prepend(seq, shift_the_first_absolute_time(seq, relative_time_seq)) for seq, relative_time_seq in zip(self.original_time_seq, self.time_seq)]
            self.time_seq = [prepend(seq, 0) for seq in self.time_seq]

            self.event = [prepend(seq, self.number_of_events) for seq in self.event]
            self.original_event = [prepend(seq, ' ') for seq in self.original_event]
            self.note = [prepend(seq, 'This is the start of the event sequence.') for seq in self.note]
            self.embeddings = [prepend(seq, [0] * self.embedding_size) for seq in self.embeddings]
            self.score = [prepend(seq, 0) for seq in self.score]
        else:
            if self.evaluate:
                self.time_seq = [prepend(seq, self.start_time) for seq in self.time_seq]
            else:
                # Use T
                # self.data.time_seq = self.data.time_seq.apply(diff, prepend = self.start_time, append = self.end_time)
                # Do not use T
                self.time_seq = [prepend(seq, self.start_time) for seq in self.time_seq]
                self.time_seq = [append(seq, seq[-1] + 0.1) for seq in self.time_seq]
                self.event = [append(seq, self.number_of_events) for seq in self.event]
                self.original_event = [append(seq, ' ') for seq in self.original_event]
                '''
                self.data.time_seq = self.data.time_seq.apply(diff, prepend = self.start_time)
                self.data.time_seq = self.data.time_seq.apply(append, number = 0.1)
                self.data.event = self.data.event.apply(append, number = self.number_of_events)
                '''

            self.event = [prepend(seq, self.number_of_events) for seq in self.event]
            self.original_event = [prepend(seq, ' ') for seq in self.original_event]

        # Fix datatype.
        # We do not output the metadata.
        self.time_seq = [np.array(seq, dtype = np.float32) for seq in self.time_seq]
        self.original_time_seq = [np.array(seq, dtype = str) for seq in self.original_time_seq]
        
        self.note = [np.array(seq, dtype = str) for seq in self.note]
        self.embeddings = [np.array(seq, dtype = np.float32) for seq in self.embeddings]
        self.event = [np.array(seq, dtype = np.int64) for seq in self.event]
        self.original_event = [np.array(seq, dtype = str) for seq in self.original_event]
        
        self.score = [np.array(seq, dtype = np.float32) for seq in self.score]
        self.intensity = [np.array(seq, dtype = np.float32) for seq in self.intensity]


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
                       self.original_time_seq[index], \
                       self.event[index], \
                       self.original_event[index], \
                       self.note[index], \
                       self.embeddings[index], \
                       self.score[index],\
                       self.intensity[index]
            else:
                return self.time_seq[index], \
                       self.original_time_seq[index], \
                       self.event[index], \
                       self.original_event[index], \
                       self.note[index], \
                       self.embeddings[index], \
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
        
        # The data batch.
        data_size = len(data)
        padded_time_seq = np.zeros((data_size, max_length_of_this_batch), dtype = np.float32)
        padded_event = np.zeros((data_size, max_length_of_this_batch), dtype = np.int64)
        padded_embeddings = np.zeros((data_size, max_length_of_this_batch, self.embedding_size), dtype = np.float32)
        padded_score = np.zeros((data_size, max_length_of_this_batch), dtype = np.float32)
        padded_mask = np.zeros((data_size, max_length_of_this_batch), dtype = np.bool)
        
        padded_original_time_seq = []
        padded_original_event_seq = []
        padded_notes = []
        
        if self.evaluate:
            padded_intensity = np.zeros((data_size, max_length_of_this_batch - 1), dtype = np.float32)
        
        for idx, item in enumerate(data):
            pad_length = max_length_of_this_batch - item[0].size
            padded_mask[idx] = np.array([1] * item[0].size + [0] * pad_length, dtype = np.bool)
            padded_time_seq[idx] = np.pad(item[0], (0, pad_length), mode = 'constant', constant_values = 0)
            padded_event[idx] = np.pad(item[2], (0, pad_length), mode = 'constant', constant_values = self.number_of_events)
            padded_embeddings[idx] = np.pad(item[5], ((0, pad_length), (0, 0)), mode = 'constant', constant_values = 0)
            padded_score[idx] = np.pad(item[6], (0, pad_length), mode = 'constant', constant_values = 0)
            
            padded_original_time_seq.append(np.pad(item[1], (0, pad_length), mode = 'constant', constant_values = ' '))
            padded_original_event_seq.append(np.pad(item[3], (0, pad_length), mode = 'constant', constant_values = ' '))
            padded_notes.append(np.pad(item[4], (0, pad_length), mode = 'constant', constant_values = ' '))

            if self.evaluate:
                padded_intensity[idx] = np.pad(item[7], (0, pad_length), mode = 'constant', constant_values = 0)
        
        # Move numpy.array to torch.tensor.
        from torch import from_numpy
        padded_time_seq = from_numpy(padded_time_seq)
        padded_event = from_numpy(padded_event)
        padded_embeddings = from_numpy(padded_embeddings)
        padded_score = from_numpy(padded_score)
        padded_mask = from_numpy(padded_mask)
        
        if self.evaluate:
            padded_intensity = from_numpy(padded_intensity)
        
        padded_original_time_seq = np.stack(padded_original_time_seq, axis = 0)
        padded_original_event_seq = np.stack(padded_original_event_seq, axis = 0)
        padded_notes = np.stack(padded_notes, axis = 0)
        
        if self.evaluate:
            return (padded_time_seq, padded_original_time_seq, padded_event, padded_original_event_seq, \
                    padded_notes, padded_embeddings, padded_score, padded_mask, padded_intensity), (self.mean, self.std)
        else:
            return (padded_time_seq, padded_original_time_seq, padded_event, padded_original_event_seq, \
                    padded_notes, padded_embeddings, padded_score, padded_mask), (self.mean, self.std)


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