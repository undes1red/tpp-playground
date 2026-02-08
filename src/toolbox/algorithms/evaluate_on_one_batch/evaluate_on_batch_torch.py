"""
Optimized batch evaluation.

OPTIMIZATION FEATURES:
---------------------
1. Unified parallelization over (metric, sequence) pairs using itertools.product
"""

from typing import TYPE_CHECKING

import torch
from einops import rearrange

from src.toolbox.metrics.torch.evaluate_func import evaluate_func as func

if TYPE_CHECKING:
    from collections.abc import Callable


def job_with_metric_and_seq(
    evaluate_func: Callable,
    batched_input_flat: torch.tensor,
    batched_target_flat: torch.tensor,
    mask_flat: torch.tensor,
    additional_inputs_flat: list[torch.tensor] | None,
    evaluate_kwargs: dict,
) -> torch.tensor:
    """Worker function for (metric, sequence) pair evaluation.

    Returns:
        torch.tensor: result
    """
    if additional_inputs_flat is not None:
        return evaluate_func(batched_input_flat, batched_target_flat, mask_flat, *additional_inputs_flat, **evaluate_kwargs)

    return evaluate_func(batched_input_flat, batched_target_flat, mask_flat, **evaluate_kwargs)

def evaluate_on_one_batch_torch(
    batched_input: torch.tensor | torch.Tensor,
    batched_target: torch.tensor | torch.Tensor,
    mask: torch.tensor | torch.Tensor,
    evaluate_func: list[Callable | str] | Callable | str,
    dim_input: int = -1,
    dim_target: int = -1,
    dim_mask: int = -1,
    additional_inputs: list[torch.tensor | torch.Tensor] | None = None,
    **evaluate_kwargs,
) -> dict[str, torch.tensor] | torch.tensor:
    """Common evaluation functions provided in scikit-learn can only evaluate one sequence per run, while machine learning
    models outputs waiting for evaluation are batches of padded sequences marked by a bool tensor usually called a mask.

    A naive approach is to flatten the 2d model outputs into a 1d sequence with only the prediction
    of true events using the mask then evaluate the 1d sequence. However, this approach makes the evaluation result dependent on
    the batch size, which is undesired.

    To address this issue, this function evaluates each sequence in the batch with mask in mind and returns the
    result of all sequences in a numpy array.

    The evaluate_func is expected to take in a sequence and return a single value or an array with a consistent shape so
    numpy can handle the final result.

    Args:
        batched_input (Union[torch.tensor, torch.Tensor]): the estimated target returned by a model
        batched_target (Union[torch.tensor, torch.Tensor]): the ground truth
        mask (Union[torch.tensor, torch.Tensor]): the mask tensor marking which is a true event and which is a padding one
        evaluate_func (Union[Callable, str, list]): the evaluation function(s). Can be a single metric or list of metrics.
        dim_input (int): where is the seq_len dim of batched_input?
        dim_target (int): where is the seq_len dim of dim_target?
        dim_mask (int): where is the seq_len dim of mask?
        additional_inputs (Union[list[Union[torch.tensor, torch.Tensor]], None]): additional batched inputs for evaluate_func beyond input and target.
    Returns:
        np.array or dict: the result. Returns dict when evaluate_func is a list, otherwise returns np.array.
    """
    # Handle single metric vs multiple metrics
    if isinstance(evaluate_func, list):
        # Multiple metrics: use unified Cartesian product approach
        metric_funcs = []
        metric_names = []
        for item in evaluate_func:
            if isinstance(item, str):
                metric_funcs.append(func(item))
                metric_names.append(item)
            else:
                metric_funcs.append(item)
                metric_names.append(item.__name__)

        return evaluate_on_batch_unified(
            batched_input,
            batched_target,
            mask,
            metric_funcs,
            metric_names,
            dim_input,
            dim_target,
            dim_mask,
            additional_inputs=additional_inputs,
            **evaluate_kwargs,
        )

    # Single metric case
    evaluate_func_resolved = func(evaluate_func) if isinstance(evaluate_func, str) else evaluate_func
    metric_name = evaluate_func if isinstance(evaluate_func, str) else evaluate_func.__name__

    results_dict = evaluate_on_batch_unified(
        batched_input,
        batched_target,
        mask,
        [evaluate_func_resolved],
        [metric_name],
        dim_input,
        dim_target,
        dim_mask,
        additional_inputs=additional_inputs,
        **evaluate_kwargs,
    )

    # Return single array for single metric
    return results_dict[metric_name]


def evaluate_on_batch_unified(
    batched_input: torch.tensor,
    batched_target: torch.tensor,
    mask: torch.tensor,
    metric_funcs: list[Callable],
    metric_names: list[str],
    dim_input: int = -1,
    dim_target: int = -1,
    dim_mask: int = -1,
    additional_inputs: list[torch.tensor] | None = None,
    **evaluate_kwargs,
) -> dict[str, torch.tensor]:
    """Unified evaluation function using Cartesian product of (metrics × sequences).

    This function creates a Cartesian product of all metrics and all sequences,
    then processes them sequentially.

    Args:
        batched_input: Input predictions
        batched_target: Ground truth targets
        mask: Boolean mask for valid entries
        metric_funcs: List of evaluation functions (already resolved)
        metric_names: List of metric names corresponding to metric_funcs
        dim_input: Dimension index for sequence length in input
        dim_target: Dimension index for sequence length in target
        dim_mask: Dimension index for sequence length in mask
        additional_inputs: Additional inputs for evaluation functions
        evaluate_kwargs: Additional keyword arguments for evaluation functions

    Returns:
        Dictionary mapping metric names to result arrays
    """
    # Move mask to bool if it is not
    if mask.dtype != torch.bool:
        mask = mask.to(torch.bool)

    batched_input_shape = batched_input.shape[:dim_input]
    batched_target_shape = batched_target.shape[:dim_target] if batched_target is not None else batched_input.shape[:dim_input]
    mask_shape = mask.shape[:dim_mask] if mask is not None else batched_input.shape[:dim_input]

    # The batch size should be the same
    if not batched_input_shape == batched_target_shape == mask_shape:
        raise ValueError("Bad input shape.")

    batch_size = batched_input_shape
    einop = f"{' '.join([f'a{index}' for index in range(len(batch_size))])} ... -> ({' '.join([f'a{index}' for index in range(len(batch_size))])}) ..."
    reversed_einop = f"({' '.join([f'a{index}' for index in range(len(batch_size))])}) ... -> {' '.join([f'a{index}' for index in range(len(batch_size))])} ..."

    batched_input_flat = rearrange(batched_input, einop)  # [(...), seq_len, ...]
    batched_target_flat = None
    if batched_target is not None:
        batched_target_flat = rearrange(batched_target, einop)  # [(...), seq_len, ...]
    mask_flat = None
    if mask is not None:
        mask_flat = rearrange(mask, einop)  # [(...), seq_len, ...]

    if additional_inputs is not None:
        additional_inputs_flat = [rearrange(inp, einop) for inp in additional_inputs]
    else:
        additional_inputs_flat = None

    # Calculate total number of sequences and metrics
    num_metrics = len(metric_funcs)

    # Sequential evaluation: loop over Cartesian product
    # Initialize results storage with proper structure
    results_by_metric = {metric_idx: [] for metric_idx in range(num_metrics)}

    # Create Cartesian product: (metric_idx, seq_idx) for all combinations
    for metric_idx in range(num_metrics):
        result = job_with_metric_and_seq(
            metric_funcs[metric_idx],
            batched_input_flat,
            batched_target_flat,
            mask_flat,
            additional_inputs_flat,
            evaluate_kwargs,
        )
        results_by_metric[metric_idx].append(result)

    # Sort each metric's results by sequence index and convert to arrays
    results_dict = {}
    for metric_idx, metric_name in enumerate(metric_names):
        result_list = torch.cat(results_by_metric[metric_idx])

        # Reshape back to original batch dimensions
        dim_prior_seq_len = {}
        for idx, item in enumerate(batched_input_shape):
            dim_prior_seq_len[f"a{idx}"] = item
        results_dict[metric_name] = rearrange(result_list, reversed_einop, **dim_prior_seq_len)

    return results_dict
