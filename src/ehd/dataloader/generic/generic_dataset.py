from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Self

import numpy as np
import torch.utils as utils

from src.toolbox.misc import load_from_pkl


def prepend_and_append(per_line: np.ndarray, prepend_item = None, append_item = None):
    if prepend_item is None and append_item is None:
        return np.array([prepend_item])
    if prepend_item is not None and append_item is None:
        return np.concatenate([np.array([prepend_item]), per_line])
    if prepend_item is None and append_item is not None:
        return np.concatenate([per_line, np.array([append_item])])
    return np.concatenate([np.array([prepend_item]), per_line, np.array([append_item])])


class GenericDataset(utils.data.Dataset):
    '''
    Self defined dataset. The required pandas DataFrame are listed in start.py.
    But...what can we do if we need prediction? It is strange.
    '''
    def __init__(self, data, device, property_dict, evaluate = False, input_norm_data = False):
        super().__init__()
        self.device = device
        self.evaluate = evaluate
        self.number_of_marks = property_dict['num_marks']
        self.start_time = property_dict['t_0']
        self.end_time = property_dict['T']
        self.mean = property_dict['mean'] if input_norm_data else 0
        self.std = property_dict['std'] if input_norm_data else 1

        # Convert data from list to np.array.
        self.time_seq = data["time_seq"]
        # compatible with old names.
        self.marks = data["event"]
        self.mask = data["mask"]

        self.dataset_size = len(self.time_seq)

        # Data preprocessing
        self.marks = [prepend_and_append(per_line, prepend_item=self.number_of_marks, append_item=self.number_of_marks) for per_line in self.marks]
        self.time_seq = [prepend_and_append(per_line, prepend_item=0, append_item=0) for per_line in self.time_seq]
        self.mask = [prepend_and_append(per_line, prepend_item=1, append_item=1) for per_line in self.mask]

        # Fix datatype
        self.time_seq = [np.array(seq, dtype=np.float32) for seq in self.time_seq]
        self.marks = [np.array(seq, dtype=np.int64) for seq in self.marks]
        self.mask = [np.array(seq, dtype=np.bool) for seq in self.mask]


    def __getitem__(self, index):
        '''
        Synthetic dataloader is very simple. It doesn't have any mark infomation at each timestamp,
        and only the time differences between two neighboring marks are available.
        '''
        if isinstance(index, slice):
            return [
                self[idx] for idx in range(index.start or 0, index.stop or len(self), index.step or 1)
            ]

        return self.time_seq[index], self.marks[index], self.mask[index]


    def __len__(self):
        return len(self.time_seq)


    def data_collator(self, data):
        '''
        The structure of data:
        [
            (time_seq, mark, score, mask, intensity if self.evaluate else it doesn't exist at all.)
        ], (mean, var)
        '''
        from torch.utils.data._utils.collate import default_collate
        data = default_collate(data)

        return data, (self.mean, self.std)


def read_data(path: str, file_name: str) -> dict:
    """Load the dataset.

    Args:
        path (str): the folder where the dataset locates.
        file_name (str): the name of the dataset file.

    Returns:
        dict: the loaded data.
    """
    return load_from_pkl(Path(path, file_name))


def generic_dataloader():
    '''
    Synthetic dataloader for all synthetic datasets.
    '''
    return [GenericDataset, read_data]
