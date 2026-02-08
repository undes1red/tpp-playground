from einops import rearrange

import torch
from src.toolbox.algorithms import approximate_integration
from src.toolbox.misc import move_from_tensor_to_ndarray


def L1_distance_across_marks(input_data, time_next, has_flatten=False):  # noqa: N802
    """
    This function calculates the L^1 distance between two functions in scattered form.
    Itorchut:
    1. input_data:  function values
                    [seq_len, resolution, num_marks] or [seq_len * resolution, num_marks] if has_flatten is True.
    2. time_next:   int
                    [seq_len, resolution]
    3. has_flatten: bool
                    See the description of "itorchut".
    """
    resolution = time_next.shape[-1]
    if has_flatten:
        input_data = rearrange(input_data, "(s r) ne -> ne s r", r=resolution)
    # [num_marks, seq_len, resolution]
    else:
        input_data = rearrange(input_data, "s r ne -> ne s r")  # [num_marks, seq_len, resolution]
    intensity_1 = rearrange(input_data, "ne s r -> ne () s r")  # [num_marks, 1, seq_len, resolution]
    intensity_2 = rearrange(input_data, "ne s r -> () ne s r")  # [1, num_marks, seq_len, resolution]
    delta_intensity = torch.abs(intensity_1 - intensity_2)  # [num_marks, num_marks, seq_len, resolution]

    timestamp = move_from_tensor_to_ndarray(time_next)  # [seq_len, resolution]
    l1 = approximate_integration(delta_intensity, timestamp, dim=-1, only_integral=True).sum(axis=-1)
    # [num_marks, num_marks]
    # round off the value smaller than 1e-6
    l1[l1 < 1e-6] = 0

    return l1


def L1_distance_between_two_funcs(x, y, timestamp):  # noqa: N802
    """
    This function calculates the L^1 distance between two functions.
    Itorchut:
    1. x:          function values
                   [seq_len, resolution]
    2. y:          function values
                   [seq_len, resolution]
    3. timestamp:  timestamp
                   [seq_len, resolution]
    """

    function_interval = torch.abs(x - y)  # [seq_len, resolution]
    l1 = approximate_integration(function_interval, timestamp, dim=-1, only_integral=True).sum()
    # round off the value smaller than 1e-6
    if l1 < 1e-6:
        l1 = 0

    return l1


def l1_self(input_data, mask, timestamp):
    """
    This function calculates the L^1 distance between two functions in scattered form.
    Itorchut:
    1. input_data:  function values
                    [..., seq_len, resolution, num_marks] or [seq_len * resolution, num_marks] if has_flatten is True.
    2. timestamp:   int
                    [..., seq_len, resolution]
    3. has_flatten: bool
                    See the description of "itorchut".
    """
    input_data = rearrange(input_data, "... s r ne -> ... ne s r")  # [..., num_marks, seq_len, resolution]
    input_data_val_1 = rearrange(input_data, "... ne s r -> ... ne () s r")  # [..., num_marks, 1, seq_len, resolution]
    input_data_val_2 = rearrange(input_data, "... ne s r -> ... () ne s r")  # [..., 1, num_marks, seq_len, resolution]
    delta_intensity = torch.abs(input_data_val_1 - input_data_val_2)  # [..., num_marks, num_marks, seq_len, resolution]

    l1 = approximate_integration(delta_intensity, timestamp, dim=-1, only_integral=True)
    # [..., num_marks, num_marks, seq_len]
    l1 = (l1 * rearrange(mask, '... sl -> ... () () sl')).sum(dim=-1)
    # [..., num_marks, num_marks]
    # round off the value smaller than 1e-6
    l1[l1 < 1e-6] = 0

    return l1

def l1(input_data, target, mask, timestamp):
    """
    This function calculates the L^1 distance between two functions.
    Itorchut:
    1. input_data: function values
                   [..., seq_len, resolution]
    2. target:     function values
                   [..., seq_len, resolution]
    3. mask:       mask
                   [..., seq_len]
    4. timestamp:  timestamp
                   [..., seq_len, resolution]
    """

    function_interval = torch.abs(input_data - target)  # [..., seq_len, resolution]
    function_interval.masked_fill(~mask.unsqueeze(dim=-1), 0)  # [..., seq_len, resolution]
    l1_vals = approximate_integration(function_interval, timestamp, dim=-1, only_integral=True).sum(dim=-1)
    # [...]
    l1_vals[l1_vals < 1e-6] = 0

    return l1_vals

if __name__ == "__main__":
    resolution = 11
    x = torch.arange(0, resolution)
    func1 = 2 * x
    func2 = 3 * x
    func3 = 4 * x

    print(f"func1: {func1}")
    print(f"func2: {func2}")
    print(f"func3: {func3}")

    L1 = L1_distance_between_two_funcs(func1, func2, x)
    print(f"L1: {L1.item()}")
    print("L1 should be 50.")

    func = torch.stack([func1, func2, func3], axis=-1)
    l1_matrix = L1_distance_across_marks(func, x, has_flatten=True)
    print(l1_matrix)
    print("The output matrix should be [[0, 50, 100], [50, 0, 50], [100, 50, 0]]")

    func = torch.stack([func1, func2, func3], axis=-1)
    func = torch.expand_dims(func, axis=0)
    l1_matrix = L1_distance_across_marks(func, x)
    print(l1_matrix)
    print("The output matrix should be [[0, 50, 100], [50, 0, 50], [100, 50, 0]]")
