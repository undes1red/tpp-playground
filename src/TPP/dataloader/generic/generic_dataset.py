import torch
import torch.utils as utils
import os
import numpy as np
from typing import Self, Dict, Any, Union, Iterable, List, Tuple, Callable

from src.toolbox.misc import load_from_pkl


def prepend(array: np.array, number: float) -> np.array:
    """insert a float number at the start of one numpy array.

    Args:
        per_line (np.array): the numpy array.
        number (float): the number inserted into the array.

    Returns:
        np.array: the result.
    """
    return np.concatenate([np.array([number]), array])


def append(per_line: np.array, number: float) -> np.array:
    """append a float number at the end of one numpy array.

    Args:
        per_line (np.array): the numpy array.
        number (float): the number inserted into the array.

    Returns:
        np.array: the result.
    """
    return np.concatenate([per_line, np.array([number])])


def diff(
    per_line: np.array, prepend: float = np._NoValue, append: float = np._NoValue
) -> np.array:
    """Calculate the diff of the input array

    Args:
        per_line (np.array): The input numpy array
        prepend (float, optional): Defaults to np._NoValue.
        append (float, optional): Defaults to np._NoValue.
        Values to prepend or append to a along axis prior to performing the difference.

    Returns:
        np.array: the result
    """
    return np.diff(per_line, prepend=prepend, append=append)


class generic_dataset(utils.data.Dataset):
    def __init__(
        self: Self,
        data: Dict[str, Any],
        device: torch.device,
        property_dict: Dict[str, Any],
        relative_time: bool = True,
        evaluate: bool = False,
        shift: bool = False,
        input_norm_data: bool = False,
    ) -> Self:
        """Create a generic dataset

        Args:
            self (Self): the dataset item,
            data (Dict[str, Any]): the raw data
            device (torch.device): the batched data will be moved to this device.
            property_dict (Dict[str, Any]): the property of the dataset.
            evaluate (bool, optional): enable or disable the evaluate mode. Defaults to False.
            shift (bool, optional): if enabled, we shift the time interval by a small number to avoid 0. Defaults to False.
            input_norm_data (bool, optional): output the mean and std of the time interval if true. Defaults to False.

        Returns:
            Self: the dataset item.
        """
        super(generic_dataset, self).__init__()
        self.device = device
        self.float_dtype = torch.get_default_dtype()
        self.evaluate = evaluate
        self.number_of_events = property_dict["num_events"]
        self.start_time = property_dict["t_0"]
        self.end_time = property_dict["T"]
        self.mean = property_dict["mean"] if input_norm_data else 0
        self.std = property_dict["std"] if input_norm_data else 1

        # Convert data from list to np.array.
        self.time_seq = data["time_seq"]
        self.score = data["score"]
        self.intensity = data["intensity"]
        self.event = data["event"]

        assert (
            len(self.time_seq)
            == len(self.score)
            == len(self.intensity)
            == len(self.event)
        ), "Dataset size mismatches!"
        self.dataset_size = len(self.time_seq)

        # Data preprocessing
        # we remove the end dummy event from the sequence when evaluate = True
        if relative_time:
            if self.evaluate:
                self.time_seq = [
                    np.diff(seq, prepend=self.start_time) for seq in self.time_seq
                ]
            else:
                # Use T
                # self.data.time_seq = self.data.time_seq.apply(diff, prepend = self.start_time, append = self.end_time)
                # Do not use T
                self.time_seq = [
                    np.diff(seq, prepend=self.start_time) for seq in self.time_seq
                ]
                self.time_seq = [append(seq, 0.1) for seq in self.time_seq]
                self.event = [append(seq, self.number_of_events) for seq in self.event]

            self.time_seq = [seq + (1e-30 if shift else 0) for seq in self.time_seq]
            self.time_seq = [prepend(seq, 0) for seq in self.time_seq]
            self.event = [prepend(seq, self.number_of_events) for seq in self.event]
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

            self.event = [prepend(seq, self.number_of_events) for seq in self.event]

        # Fix datatype
        self.time_seq = [np.array(seq, dtype=np.float32) for seq in self.time_seq]
        self.score = [np.array(seq, dtype=np.float32) for seq in self.score]
        self.intensity = [np.array(seq, dtype=np.float32) for seq in self.intensity]
        self.event = [np.array(seq, dtype=np.int64) for seq in self.event]

    def __getitem__(self: Self, index: Union[int, Iterable]) -> List:
        """Get the batched data.

        Args:
            self (Self): the dataset
            index (Union[int, Iterable]): the index of the selected data.

        Returns:
            List: the output.
        """
        """
        Synthetic dataloader is very simple. It doesn't have any event infomation at each timestamp,
        and only the time differences between two neighboring events are available.
        """
        if isinstance(index, slice):
            return [
                self[idx]
                for idx in range(
                    index.start or 0, index.stop or len(self), index.step or 1
                )
            ]
        else:
            if self.evaluate:
                return (
                    self.time_seq[index],
                    self.event[index],
                    self.score[index],
                    self.intensity[index],
                )
            else:
                return self.time_seq[index], self.event[index], self.score[index]

    def __len__(self: Self):
        """return the length of the dataset.

        Args:
            self (Self): the dataset item.

        Returns:
            int: the length of the dataset.
        """
        return self.dataset_size

    def data_collator(self: Self, data: List) -> Tuple[List, Tuple[float, float]]:
        """
        The structure of data:
        [
            (time_seq, event, score, mask, intensity if self.evaluate else it doesn't exist at all.)
        ], (mean, var)
        """
        max_length_of_this_batch = max([item[0].size for item in data])
        mask = []
        padded_data = []
        for item in data:
            pad_length = max_length_of_this_batch - item[0].size
            mask = np.array([1] * item[0].size + [0] * pad_length, dtype=np.bool)
            padded_time_seq = np.pad(
                item[0], (0, pad_length), mode="constant", constant_values=0
            )
            padded_event = np.pad(
                item[1],
                (0, pad_length),
                mode="constant",
                constant_values=self.number_of_events,
            )
            padded_score = np.pad(
                item[2], (0, pad_length), mode="constant", constant_values=0
            )
            padded_item = [padded_time_seq, padded_event, padded_score, mask]
            if self.evaluate:
                padded_intensity = np.pad(
                    item[3], (0, pad_length), mode="constant", constant_values=0
                )
                padded_item.append(padded_intensity)

            padded_data.append(tuple(padded_item))

        from torch.utils.data._utils.collate import default_collate

        padded_data = default_collate(padded_data)
        padded_data = [
            item.to(self.float_dtype) if torch.is_floating_point(item) else item
            for item in padded_data
        ]

        return padded_data, (self.mean, self.std)


def read_data(path: str, file_name: str) -> Dict:
    """Load the dataset.

    Args:
        path (str): the folder where the dataset locates.
        file_name (str): the name of the dataset file.

    Returns:
        Dict: the loaded data.
    """
    return load_from_pkl(os.path.join(path, file_name))


def generic_dataloader() -> Tuple[utils.data.Dataset, Callable]:
    return [generic_dataset, read_data]
