"""
Parameter Parser Module

This module provides functionality to parse parameter dictionaries and generate
command-line argument combinations for batch task execution.
"""

import math
import numpy as np
from typing import Dict, List, Any
from itertools import product

from src.toolbox.misc import merge_list_of_dicts, get_logger

logger = get_logger(__name__)


def parse_sequential_parameters(
    sequential_dict: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Parse sequential parameters using zip-like approach.

    Args:
        sequential_dict (dict): Dictionary with 'loop_vars' key containing parameter lists

    Returns:
        list: List of dictionaries, each containing one combination of parameters

    Raises:
        ValueError: If parameter lists have different lengths
    """
    # Return [{}] if the input sequential_dict is empty.
    if len(sequential_dict) == 0:
        return [{}]

    # Get nested sequential or combinatorial dict if exist.
    sub_sequential_dict = sequential_dict.get("sequential", {})
    sub_combinatorial_dict = sequential_dict.get("combinatorial", {})

    # Get the corresponding results of nested sequential or combinatorial dicts if exist.
    sub_sequential_result_list = parse_sequential_parameters(sub_sequential_dict)
    sub_combinatorial_result_list = parse_combinatorial_parameters(
        sub_combinatorial_dict
    )

    # The merge will work as expected when sub_sequential_result_list or sub_combinatorial_result_list is [{}], or empty.
    if sub_sequential_result_list == [{}]:
        sub_result_list = sub_combinatorial_result_list
    elif sub_combinatorial_result_list == [{}]:
        sub_result_list = sub_sequential_result_list
    else:
        # zip merge these two list of dicts.
        assert len(sub_combinatorial_result_list) == len(sub_sequential_result_list)
        sub_result_list = [
            merge_list_of_dicts(items)
            for items in zip(sub_sequential_result_list, sub_combinatorial_result_list)
        ]
    length_sub_result_list = len(sub_result_list) if sub_result_list != [{}] else None

    # nested sequential or combinatorial dict processed. Remove them.
    sequential_dict.pop("sequential", None)
    sequential_dict.pop("combinatorial", None)

    # Handle argument missing case
    if not sequential_dict:
        return sub_result_list

    # Get the lengths of all parameter lists
    lengths = [len(values) for values in sequential_dict.values()] + (
        [
            length_sub_result_list,
        ]
        if length_sub_result_list is not None
        else []
    )

    # Validate that all lists have the same length
    if len(set(lengths)) > 1:
        lengths = [str(item) for item in lengths]
        logger.exception(
            f"All parameter lists in sequential mode must have the same length. But we find the parameter list length can be {' or '.join(list(set(lengths)))}."
        )

    # Generate combinations
    combinations = []
    expected_length = lengths[0]

    for i in range(expected_length):
        combination = {key: values[i] for key, values in sequential_dict.items()}
        combinations.append(combination)

    if sub_result_list != [{}]:
        combinations = [
            merge_list_of_dicts(items) for items in zip(combinations, sub_result_list)
        ]

    return combinations


def parse_combinatorial_parameters(
    combinatorial_dict: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Parse combinatorial parameters generating all possible combinations.

    Args:
        combinatorial_dict (dict): Dictionary with 'loop_vars' key containing parameter lists

    Returns:
        list: List of dictionaries, each containing one combination of parameters
    """
    # Return [{}] if the input sequential_dict is empty.
    if len(combinatorial_dict) == 0:
        return [{}]

    # Get nested sequential or combinatorial dict if exist.
    sub_sequential_dict = combinatorial_dict.get("sequential", {})
    sub_combinatorial_dict = combinatorial_dict.get("combinatorial", {})

    # Get the corresponding results of nested sequential or combinatorial dicts if exist.
    sub_sequential_result_list = parse_sequential_parameters(sub_sequential_dict)
    sub_combinatorial_result_list = parse_combinatorial_parameters(
        sub_combinatorial_dict
    )

    # combinatorially merge these two list of dicts.
    sub_result_list = [
        merge_list_of_dicts(item)
        for item in product(sub_sequential_result_list, sub_combinatorial_result_list)
    ]

    # nested sequential or combinatorial dict processed. Remove them.
    combinatorial_dict.pop("sequential", None)
    combinatorial_dict.pop("combinatorial", None)

    # Handle empty case
    if not combinatorial_dict:
        return sub_result_list

    # value_matrices is useful when the value of some parameters relies on others.
    # For instance, the value of task_config relies on task_name and dataset_name.
    # We can place all possible task_config file names in a matrix then retrieve
    # the needed name according to task_name and dataset_name.
    param_value_matrices = combinatorial_dict.get("value_matrices", {})
    value_matrices = True
    if not param_value_matrices:
        value_matrices = False
    else:
        for key, value in param_value_matrices.items():
            param_value_matrices[key] = np.array(value)
    combinatorial_dict.pop("value_matrices", None)

    # Get parameter names and their possible values
    param_names = tuple(combinatorial_dict.keys())
    param_values = tuple(combinatorial_dict.values())
    param_values_length = tuple((len(values) for values in param_values))

    if value_matrices:
        for key, item in param_value_matrices.items():
            try:
                assert item.shape == param_values_length
            except AssertionError:
                logger.exception(
                    f'We expect the value matrix with key "{key}" has shape {param_values_length} but it has shape {item.shape}.'
                )

    # Calculate total number of combinations
    total_combinations = math.prod(param_values_length)

    # Handle case with no combinations
    if total_combinations == 0:
        return sub_result_list

    # Generate all combinations using iterative approach
    combinations = []

    # Initialize indices for each parameter
    indices = [0] * len(param_values)

    for _ in range(total_combinations):
        # Create combination from current indices
        combination = {
            param_names[i]: param_values[i][indices[i]] for i in range(len(param_names))
        }
        if value_matrices:
            combination.update(
                {
                    key: value[*indices].item()
                    for key, value in param_value_matrices.items()
                }
            )
        combinations.append(combination)

        # Increment indices (like counting)
        indices[0] += 1
        for i in range(len(indices)):
            if indices[i] >= len(param_values[i]):
                indices[i] = 0
                if i + 1 < len(indices):
                    indices[i + 1] += 1
            else:
                break

    # Merge combinations with the sub_result_list.
    combinations = [
        merge_list_of_dicts(item) for item in product(combinations, sub_result_list)
    ]

    return combinations


def combine_parameters(
    static_params: Dict[str, Any],
    sequential_combinations: List[Dict[str, Any]],
    combinatorial_combinations: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Combine all parsed parameters using combinatorial approach.

    Args:
        static_params (dict): Static parameters to be added to each combination
        sequential_combinations (list): List of sequential parameter dictionaries
        combinatorial_combinations (list): List of combinatorial parameter dictionaries

    Returns:
        list: List of complete parameter dictionaries
    """
    combined_results = []

    # Handle case where we have no sequential or combinatorial parameters
    if not sequential_combinations:
        sequential_combinations = [{}]
    if not combinatorial_combinations:
        combinatorial_combinations = [{}]

    combined_results = [
        merge_list_of_dicts(item)
        for item in product(
            [
                static_params,
            ],
            sequential_combinations,
            combinatorial_combinations,
        )
    ]

    return combined_results


def convert_to_command_line_args(param_dict: Dict[str, Any]) -> List[str]:
    """
    Convert a parameter dictionary to command-line arguments.

    Args:
        param_dict (dict): Dictionary of parameters

    Returns:
        list: List of command-line arguments
    """
    arguments = []
    for key, value in param_dict.items():
        if isinstance(value, bool):
            if value:
                arguments.append(f"--{key}")
        else:
            arguments.extend([f"--{key}", str(value)])
    return arguments


def parameter_parser(input_dict: Dict[str, Any]) -> List[List[str]]:
    """
    Main parameter parser function that orchestrates the parsing process.
    Please note that "sequential", "combinatorial", and "value_matrices"(combinatorial specific) are reserved keys.

    Args:
        input_dict (dict): Input dictionary with static, sequential, and combinatorial parameters
                          Expected structure:
                          {
                              "static": { ... },
                              "sequential": { ... },
                              "combinatorial": { ... , 'value_matrices': { ... }}
                          }

    Returns:
        list: List of dictionaries, each containing a complete set of parameters
    """
    # Extract different parameter types
    static_params = input_dict.get("static", {})
    sequential_params_dict = input_dict.get("sequential", {})
    combinatorial_params_dict = input_dict.get("combinatorial", {})

    # Parse each parameter type
    sequential_combinations = parse_sequential_parameters(sequential_params_dict)
    combinatorial_combinations = parse_combinatorial_parameters(
        combinatorial_params_dict
    )

    # Combine all parameters
    combined_params = combine_parameters(
        static_params, sequential_combinations, combinatorial_combinations
    )

    # Transfer dict parameters into command line arguments.
    combined_params = [convert_to_command_line_args(item) for item in combined_params]

    # Add headers
    # <interpreter> <file_name> <argparser> ...
    combined_params = [
        [input_dict["interpreter"], input_dict["file_name"], input_dict["argparser"]]
        + item
        for item in combined_params
    ]

    return combined_params


__all__ = [parameter_parser]


if __name__ == "__main__":
    # case 1
    dataset_name = ["a", "b", "c", "d"]
    dataset_config = [1, 2, 3, 4]
    train_on_syn_datasets = {
        "interpreter": "python",
        "file_name": "a.py",
        "argparser": "a",
        "worker": "start.py",
        "job_type": "train",
        "static": {
            "no_seed": True,
        },
        "sequential": {"dataset_name": dataset_name, "dataset_config": dataset_config},
    }

    gen_arguments = parameter_parser(train_on_syn_datasets)
    assert gen_arguments == [
        [
            "python",
            "a.py",
            "a",
            "--no_seed",
            "--dataset_name",
            "a",
            "--dataset_config",
            "1",
        ],
        [
            "python",
            "a.py",
            "a",
            "--no_seed",
            "--dataset_name",
            "b",
            "--dataset_config",
            "2",
        ],
        [
            "python",
            "a.py",
            "a",
            "--no_seed",
            "--dataset_name",
            "c",
            "--dataset_config",
            "3",
        ],
        [
            "python",
            "a.py",
            "a",
            "--no_seed",
            "--dataset_name",
            "d",
            "--dataset_config",
            "4",
        ],
    ]

    # case 2
    dataset_name = ["a", "b", "c", "d"]
    dataset_config = [1, 2, 3, 4]
    train_on_syn_datasets = {
        "interpreter": "python",
        "file_name": "a.py",
        "argparser": "a",
        "worker": "start.py",
        "job_type": "train",
        "static": {
            "no_seed": True,
        },
        "combinatorial": {
            "dataset_name": dataset_name,
            "dataset_config": dataset_config,
        },
    }

    gen_arguments = parameter_parser(train_on_syn_datasets)
    assert gen_arguments == [
        [
            "python",
            "a.py",
            "a",
            "--no_seed",
            "--dataset_name",
            "a",
            "--dataset_config",
            "1",
        ],
        [
            "python",
            "a.py",
            "a",
            "--no_seed",
            "--dataset_name",
            "b",
            "--dataset_config",
            "1",
        ],
        [
            "python",
            "a.py",
            "a",
            "--no_seed",
            "--dataset_name",
            "c",
            "--dataset_config",
            "1",
        ],
        [
            "python",
            "a.py",
            "a",
            "--no_seed",
            "--dataset_name",
            "d",
            "--dataset_config",
            "1",
        ],
        [
            "python",
            "a.py",
            "a",
            "--no_seed",
            "--dataset_name",
            "a",
            "--dataset_config",
            "2",
        ],
        [
            "python",
            "a.py",
            "a",
            "--no_seed",
            "--dataset_name",
            "b",
            "--dataset_config",
            "2",
        ],
        [
            "python",
            "a.py",
            "a",
            "--no_seed",
            "--dataset_name",
            "c",
            "--dataset_config",
            "2",
        ],
        [
            "python",
            "a.py",
            "a",
            "--no_seed",
            "--dataset_name",
            "d",
            "--dataset_config",
            "2",
        ],
        [
            "python",
            "a.py",
            "a",
            "--no_seed",
            "--dataset_name",
            "a",
            "--dataset_config",
            "3",
        ],
        [
            "python",
            "a.py",
            "a",
            "--no_seed",
            "--dataset_name",
            "b",
            "--dataset_config",
            "3",
        ],
        [
            "python",
            "a.py",
            "a",
            "--no_seed",
            "--dataset_name",
            "c",
            "--dataset_config",
            "3",
        ],
        [
            "python",
            "a.py",
            "a",
            "--no_seed",
            "--dataset_name",
            "d",
            "--dataset_config",
            "3",
        ],
        [
            "python",
            "a.py",
            "a",
            "--no_seed",
            "--dataset_name",
            "a",
            "--dataset_config",
            "4",
        ],
        [
            "python",
            "a.py",
            "a",
            "--no_seed",
            "--dataset_name",
            "b",
            "--dataset_config",
            "4",
        ],
        [
            "python",
            "a.py",
            "a",
            "--no_seed",
            "--dataset_name",
            "c",
            "--dataset_config",
            "4",
        ],
        [
            "python",
            "a.py",
            "a",
            "--no_seed",
            "--dataset_name",
            "d",
            "--dataset_config",
            "4",
        ],
    ]

    # case 3
    dataset_name = [
        "a",
        "b",
        "c",
    ]
    dataset_config = [1, 2, 3, 4]
    train_on_syn_datasets = {
        "interpreter": "python",
        "file_name": "a.py",
        "argparser": "a",
        "worker": "start.py",
        "job_type": "train",
        "static": {
            "no_seed": True,
        },
        "sequential": {"dataset_name": dataset_name, "dataset_config": dataset_config},
    }
    try:
        gen_arguments = parameter_parser(train_on_syn_datasets)
    except Exception:
        pass

    # case 4
    dataset_name = ["a", "b"]
    dataset_config = [1, 2, 3]
    value_matrices = {
        "first": [["a11", "a12", "a13"], ["a21", "a22", "a23"]],
        "second": [["b11", "b12", "b13"], ["b21", "b22", "b23"]],
    }
    train_on_syn_datasets = {
        "interpreter": "python",
        "file_name": "a.py",
        "argparser": "a",
        "worker": "start.py",
        "job_type": "train",
        "static": {
            "no_seed": True,
        },
        "combinatorial": {
            "dataset_name": dataset_name,
            "dataset_config": dataset_config,
            "value_matrices": value_matrices,
        },
    }
    gen_arguments = parameter_parser(train_on_syn_datasets)
    assert gen_arguments == [
        [
            "python",
            "a.py",
            "a",
            "--no_seed",
            "--dataset_name",
            "a",
            "--dataset_config",
            "1",
            "--first",
            "a11",
            "--second",
            "b11",
        ],
        [
            "python",
            "a.py",
            "a",
            "--no_seed",
            "--dataset_name",
            "b",
            "--dataset_config",
            "1",
            "--first",
            "a21",
            "--second",
            "b21",
        ],
        [
            "python",
            "a.py",
            "a",
            "--no_seed",
            "--dataset_name",
            "a",
            "--dataset_config",
            "2",
            "--first",
            "a12",
            "--second",
            "b12",
        ],
        [
            "python",
            "a.py",
            "a",
            "--no_seed",
            "--dataset_name",
            "b",
            "--dataset_config",
            "2",
            "--first",
            "a22",
            "--second",
            "b22",
        ],
        [
            "python",
            "a.py",
            "a",
            "--no_seed",
            "--dataset_name",
            "a",
            "--dataset_config",
            "3",
            "--first",
            "a13",
            "--second",
            "b13",
        ],
        [
            "python",
            "a.py",
            "a",
            "--no_seed",
            "--dataset_name",
            "b",
            "--dataset_config",
            "3",
            "--first",
            "a23",
            "--second",
            "b23",
        ],
    ]

    # case 5:
    dataset_name = ["a", "b"]
    dataset_config = [1, 2, 3]
    value_matrices = {
        "first": [["a11", "a12"], ["a21", "a22"]],
        "second": [["b11", "b12", "b13"], ["b21", "b22", "b23"]],
    }
    train_on_syn_datasets = {
        "interpreter": "python",
        "file_name": "a.py",
        "argparser": "a",
        "worker": "start.py",
        "job_type": "train",
        "static": {
            "no_seed": True,
        },
        "combinatorial": {
            "dataset_name": dataset_name,
            "dataset_config": dataset_config,
            "value_matrices": value_matrices,
        },
    }
    try:
        gen_arguments = parameter_parser(train_on_syn_datasets)
    except Exception as e:
        print(e)

    # case 6:
    dataset_name = ["a", "b", "c"]
    dataset_config = [1, 2, 3]
    first = ["alpha", "beta", "gamma"]
    train_on_syn_datasets = {
        "interpreter": "python",
        "file_name": "a.py",
        "argparser": "a",
        "worker": "start.py",
        "job_type": "train",
        "static": {
            "no_seed": True,
        },
        "sequential": {
            "dataset_name": dataset_name,
            "sequential": {
                "dataset_config": dataset_config,
                "sequential": {"first": first},
            },
        },
    }
    gen_arguments = parameter_parser(train_on_syn_datasets)
    assert gen_arguments == [
        [
            "python",
            "a.py",
            "a",
            "--no_seed",
            "--dataset_name",
            "a",
            "--dataset_config",
            "1",
            "--first",
            "alpha",
        ],
        [
            "python",
            "a.py",
            "a",
            "--no_seed",
            "--dataset_name",
            "b",
            "--dataset_config",
            "2",
            "--first",
            "beta",
        ],
        [
            "python",
            "a.py",
            "a",
            "--no_seed",
            "--dataset_name",
            "c",
            "--dataset_config",
            "3",
            "--first",
            "gamma",
        ],
    ]

    # case 7:
    dataset_name = ["a", "b", "c", "d"]
    dataset_config = [1, 2, 3, 4]
    first = ["alpha", "beta", "gamma", "theta"]
    t1 = ["a1", "a2"]
    t2 = ["b1", "b2"]
    m1 = [["c11", "c12"], ["c21", "c22"]]
    m2 = [["d11", "d12"], ["d21", "d22"]]
    train_on_syn_datasets = {
        "interpreter": "python",
        "file_name": "a.py",
        "argparser": "a",
        "worker": "start.py",
        "job_type": "train",
        "static": {
            "no_seed": True,
        },
        "sequential": {
            "dataset_name": dataset_name,
            "sequential": {
                "dataset_config": dataset_config,
                "sequential": {"first": first},
            },
            "combinatorial": {
                "combinatorial": {
                    "t1": t1,
                    "t2": t2,
                    "value_matrices": {"m1": m1, "m2": m2},
                }
            },
        },
        "combinatorial": {"t3": ["!", "@", "#"]},
    }
    gen_arguments = parameter_parser(train_on_syn_datasets)
    assert gen_arguments == [
        [
            "python",
            "a.py",
            "a",
            "--no_seed",
            "--dataset_name",
            "a",
            "--dataset_config",
            "1",
            "--first",
            "alpha",
            "--t1",
            "a1",
            "--t2",
            "b1",
            "--m1",
            "c11",
            "--m2",
            "d11",
            "--t3",
            "!",
        ],
        [
            "python",
            "a.py",
            "a",
            "--no_seed",
            "--dataset_name",
            "a",
            "--dataset_config",
            "1",
            "--first",
            "alpha",
            "--t1",
            "a1",
            "--t2",
            "b1",
            "--m1",
            "c11",
            "--m2",
            "d11",
            "--t3",
            "@",
        ],
        [
            "python",
            "a.py",
            "a",
            "--no_seed",
            "--dataset_name",
            "a",
            "--dataset_config",
            "1",
            "--first",
            "alpha",
            "--t1",
            "a1",
            "--t2",
            "b1",
            "--m1",
            "c11",
            "--m2",
            "d11",
            "--t3",
            "#",
        ],
        [
            "python",
            "a.py",
            "a",
            "--no_seed",
            "--dataset_name",
            "b",
            "--dataset_config",
            "2",
            "--first",
            "beta",
            "--t1",
            "a2",
            "--t2",
            "b1",
            "--m1",
            "c21",
            "--m2",
            "d21",
            "--t3",
            "!",
        ],
        [
            "python",
            "a.py",
            "a",
            "--no_seed",
            "--dataset_name",
            "b",
            "--dataset_config",
            "2",
            "--first",
            "beta",
            "--t1",
            "a2",
            "--t2",
            "b1",
            "--m1",
            "c21",
            "--m2",
            "d21",
            "--t3",
            "@",
        ],
        [
            "python",
            "a.py",
            "a",
            "--no_seed",
            "--dataset_name",
            "b",
            "--dataset_config",
            "2",
            "--first",
            "beta",
            "--t1",
            "a2",
            "--t2",
            "b1",
            "--m1",
            "c21",
            "--m2",
            "d21",
            "--t3",
            "#",
        ],
        [
            "python",
            "a.py",
            "a",
            "--no_seed",
            "--dataset_name",
            "c",
            "--dataset_config",
            "3",
            "--first",
            "gamma",
            "--t1",
            "a1",
            "--t2",
            "b2",
            "--m1",
            "c12",
            "--m2",
            "d12",
            "--t3",
            "!",
        ],
        [
            "python",
            "a.py",
            "a",
            "--no_seed",
            "--dataset_name",
            "c",
            "--dataset_config",
            "3",
            "--first",
            "gamma",
            "--t1",
            "a1",
            "--t2",
            "b2",
            "--m1",
            "c12",
            "--m2",
            "d12",
            "--t3",
            "@",
        ],
        [
            "python",
            "a.py",
            "a",
            "--no_seed",
            "--dataset_name",
            "c",
            "--dataset_config",
            "3",
            "--first",
            "gamma",
            "--t1",
            "a1",
            "--t2",
            "b2",
            "--m1",
            "c12",
            "--m2",
            "d12",
            "--t3",
            "#",
        ],
        [
            "python",
            "a.py",
            "a",
            "--no_seed",
            "--dataset_name",
            "d",
            "--dataset_config",
            "4",
            "--first",
            "theta",
            "--t1",
            "a2",
            "--t2",
            "b2",
            "--m1",
            "c22",
            "--m2",
            "d22",
            "--t3",
            "!",
        ],
        [
            "python",
            "a.py",
            "a",
            "--no_seed",
            "--dataset_name",
            "d",
            "--dataset_config",
            "4",
            "--first",
            "theta",
            "--t1",
            "a2",
            "--t2",
            "b2",
            "--m1",
            "c22",
            "--m2",
            "d22",
            "--t3",
            "@",
        ],
        [
            "python",
            "a.py",
            "a",
            "--no_seed",
            "--dataset_name",
            "d",
            "--dataset_config",
            "4",
            "--first",
            "theta",
            "--t1",
            "a2",
            "--t2",
            "b2",
            "--m1",
            "c22",
            "--m2",
            "d22",
            "--t3",
            "#",
        ],
    ]

    # case 8
    dataset_name = ["a", "b", "c", "d"]
    dataset_config = [1, 2, 3, 4]
    train_on_syn_datasets = {
        "interpreter": "python",
        "file_name": "a.py",
        "argparser": "a",
        "worker": "start.py",
        "job_type": "train",
        "static": {
            "no_seed": True,
        }
    }
    gen_arguments = parameter_parser(train_on_syn_datasets)
    assert gen_arguments == [['python', 'a.py', 'a', '--no_seed']]