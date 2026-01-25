"""
Optimized batch evaluation with multiprocessing support.

MULTIPROCESSING PERFORMANCE NOTES:
----------------------------------
Multiprocessing adds overhead from process creation and inter-process communication.
It's beneficial when:
1. The evaluation function is computationally expensive ( e.g., complex metrics, OTD)
2. The batch size is large or there are multiple metrics (typically >= 8-16 total jobs)
3. Each sequence has sufficient data points to process

For lightweight metrics like accuracy_score on small sequences, the overhead may
outweigh the benefits. In such cases, sequential processing is faster.

UNIFIED MULTIPROCESSING APPROACH:
---------------------------------
When multiprocessing is enabled, the function parallelizes over the Cartesian product
of (metrics x sequences). This provides:
- Simple, unified approach for both single and multiple metrics
- Optimal load balancing across all available cores
- No nested parallelism issues (daemon processes limitation)

For example:
- 3 metrics x 64 sequences = 192 jobs distributed across workers
- 1 metric x 128 sequences = 128 jobs distributed across workers

OPTIMIZATION FEATURES:
---------------------
1. Automatic worker count based on CPU cores
2. Smart threshold (default: 8 total jobs minimum)
3. Chunked work distribution to reduce overhead
4. Proper pool cleanup using context managers
5. Worker count capped at number of jobs (no idle workers)
6. Unified parallelization over (metric, sequence) pairs using itertools.product
"""

import multiprocessing as mp
from collections.abc import Callable
from itertools import product

import numpy as np
import torch
from einops import rearrange

from src.toolbox.metrics.evaluate_func import evaluate_func as func
from src.toolbox.misc import move_from_tensor_to_ndarray


def job_with_metric_and_seq(
    metric_idx: int,
    seq_idx: int,
    evaluate_func: Callable,
    batched_input_flat: np.ndarray,
    batched_target_flat: np.ndarray,
    mask_flat: np.ndarray,
    additional_inputs_flat: list[np.ndarray] | None,
    evaluate_kwargs: dict,
) -> tuple[int, int, np.ndarray]:
    """Worker function for (metric, sequence) pair evaluation.

    Returns:
        tuple: (metric_index, sequence_index, result) for later reorganization
    """
    single_input = batched_input_flat[seq_idx]
    single_target = batched_target_flat[seq_idx]
    single_mask = mask_flat[seq_idx]

    single_input_masked = single_input[single_mask]
    single_target_masked = single_target[single_mask]

    if additional_inputs_flat is not None:
        additional_masked = [inp[seq_idx][single_mask] for inp in additional_inputs_flat]
        result = evaluate_func(single_input_masked, single_target_masked, *additional_masked, **evaluate_kwargs)
    else:
        result = evaluate_func(single_input_masked, single_target_masked, **evaluate_kwargs)

    return (metric_idx, seq_idx, result)


def evaluate_on_one_batch(
    batched_input: np.ndarray | torch.Tensor,
    batched_target: np.ndarray | torch.Tensor,
    mask: np.ndarray | torch.Tensor,
    evaluate_func: list[Callable | str] | Callable | str,
    dim_input: int = -1,
    dim_target: int = -1,
    dim_mask: int = -1,
    multiprocessing=False,
    additional_inputs: list[np.ndarray | torch.Tensor] | None = None,
    num_workers: int | None = None,
    mp_threshold: int = 8,
    **evaluate_kwargs,
) -> dict[str, np.ndarray] | np.ndarray:
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
        batched_input (Union[np.ndarray, torch.Tensor]): the estimated target returned by a model
        batched_target (Union[np.ndarray, torch.Tensor]): the ground truth
        mask (Union[np.ndarray, torch.Tensor]): the mask tensor marking which is a true event and which is a padding one
        evaluate_func (Union[Callable, str, list]): the evaluation function(s). Can be a single metric or list of metrics.
        dim_input (int): where is the seq_len dim of batched_input?
        dim_target (int): where is the seq_len dim of dim_target?
        dim_mask (int): where is the seq_len dim of mask?
        multiprocessing (bool): use multiprocessing to parallelize across (metric × sequence) pairs.
        additional_inputs (Union[list[Union[np.ndarray, torch.Tensor]], None]): additional batched inputs for evaluate_func beyond input and target.
        num_workers (int | None): Number of worker processes. If None, uses cpu_count(). Only used when multiprocessing=True.
        mp_threshold (int): Minimum number of total jobs (metrics × sequences) to enable multiprocessing. Default is 8.
    Returns:
        np.array or dict: the result. Returns dict when evaluate_func is a list, otherwise returns np.array.
    """
    if torch.is_tensor(batched_input):
        batched_input = move_from_tensor_to_ndarray(batched_input)

    if torch.is_tensor(batched_target):
        batched_target = move_from_tensor_to_ndarray(batched_target)

    if torch.is_tensor(mask):
        mask = move_from_tensor_to_ndarray(mask)

    if additional_inputs is not None:
        additional_inputs = [
            move_from_tensor_to_ndarray(inp) if torch.is_tensor(inp) else inp for inp in additional_inputs
        ]

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
            multiprocessing,
            additional_inputs=additional_inputs,
            num_workers=num_workers,
            mp_threshold=mp_threshold,
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
        multiprocessing,
        additional_inputs=additional_inputs,
        num_workers=num_workers,
        mp_threshold=mp_threshold,
        **evaluate_kwargs,
    )

    # Return single array for single metric
    return results_dict[metric_name]


def evaluate_on_batch_unified(
    batched_input: np.ndarray,
    batched_target: np.ndarray,
    mask: np.ndarray,
    metric_funcs: list[Callable],
    metric_names: list[str],
    dim_input: int = -1,
    dim_target: int = -1,
    dim_mask: int = -1,
    multiprocessing: bool = False,
    additional_inputs: list[np.ndarray] | None = None,
    num_workers: int | None = None,
    mp_threshold: int = 8,
    **evaluate_kwargs,
) -> dict[str, np.ndarray]:
    """Unified evaluation function using Cartesian product of (metrics × sequences).

    This function creates a Cartesian product of all metrics and all sequences,
    then processes them either sequentially or in parallel using multiprocessing.

    Args:
        batched_input: Input predictions
        batched_target: Ground truth targets
        mask: Boolean mask for valid entries
        metric_funcs: List of evaluation functions (already resolved)
        metric_names: List of metric names corresponding to metric_funcs
        dim_input: Dimension index for sequence length in input
        dim_target: Dimension index for sequence length in target
        dim_mask: Dimension index for sequence length in mask
        multiprocessing: Whether to use multiprocessing
        additional_inputs: Additional inputs for evaluation functions
        num_workers: Number of worker processes
        mp_threshold: Minimum jobs for enabling multiprocessing
        evaluate_kwargs: Additional keyword arguments for evaluation functions

    Returns:
        Dictionary mapping metric names to result arrays
    """
    # Move mask to bool if it is not
    if mask.dtype != np.bool:
        mask = mask.astype(np.bool)

    batched_input_shape = batched_input.shape
    batched_target_shape = batched_target.shape
    mask_shape = mask.shape

    # The batch size should be the same
    if not batched_input_shape[:dim_input] == batched_target_shape[:dim_target] == mask_shape[:dim_mask]:
        raise ValueError("Bad input shape.")

    batch_size = batched_input_shape[:dim_input]
    einop = f"{' '.join([f'a{index}' for index in range(len(batch_size))])} ... -> ({' '.join([f'a{index}' for index in range(len(batch_size))])}) ..."
    reversed_einop = f"({' '.join([f'a{index}' for index in range(len(batch_size))])}) ... -> {' '.join([f'a{index}' for index in range(len(batch_size))])} ..."

    batched_input_flat = rearrange(batched_input, einop)  # [(...), seq_len, ...]
    batched_target_flat = rearrange(batched_target, einop)  # [(...), seq_len, ...]
    mask_flat = rearrange(mask, einop)  # [(...), seq_len, ...]

    if additional_inputs is not None:
        additional_inputs_flat = [rearrange(inp, einop) for inp in additional_inputs]
    else:
        additional_inputs_flat = None

    # Calculate total number of sequences and metrics
    total_sequences = int(np.prod(batch_size))
    num_metrics = len(metric_funcs)
    total_jobs = total_sequences * num_metrics

    # Only use multiprocessing if beneficial (avoid overhead for small batches)
    use_mp = multiprocessing and total_jobs >= mp_threshold

    if use_mp:
        # Determine number of workers
        if num_workers is None:
            num_workers = mp.cpu_count()
        # Further optimize: don't use more workers than total jobs
        num_workers = min(num_workers, total_jobs)

        # Calculate optimal chunk size to reduce overhead
        chunksize = max(1, total_jobs // (num_workers * 4))

        # Create Cartesian product: (metric_idx, seq_idx) for all combinations
        job_args = [
            (
                metric_idx,
                seq_idx,
                metric_funcs[metric_idx],
                batched_input_flat,
                batched_target_flat,
                mask_flat,
                additional_inputs_flat,
                evaluate_kwargs,
            )
            for metric_idx, seq_idx in product(range(num_metrics), range(total_sequences))
        ]

        # Use context manager to ensure pool is properly closed
        # change processing creation method to spawn so cuda is happy.
        with mp.Pool(num_workers) as pool:
            results_with_indices = pool.starmap(job_with_metric_and_seq, job_args, chunksize=chunksize)

        # Organize results by metric
        results_by_metric = {metric_idx: [] for metric_idx in range(num_metrics)}
        for metric_idx, seq_idx, result in results_with_indices:
            results_by_metric[metric_idx].append((seq_idx, result))

        # Sort each metric's results by sequence index and convert to arrays
        results_dict = {}
        for metric_idx, metric_name in enumerate(metric_names):
            results_by_metric[metric_idx].sort(key=lambda x: x[0])
            result_list = np.array([result for _, result in results_by_metric[metric_idx]])

            # Reshape back to original batch dimensions
            dim_prior_seq_len = {}
            for idx, item in enumerate(batched_input_shape[:dim_input]):
                dim_prior_seq_len[f"a{idx}"] = item
            results_dict[metric_name] = rearrange(result_list, reversed_einop, **dim_prior_seq_len)
    else:
        # Sequential evaluation: loop over Cartesian product
        # Initialize results storage with proper structure
        results_by_metric = {metric_idx: [] for metric_idx in range(num_metrics)}

        # Create Cartesian product: (metric_idx, seq_idx) for all combinations
        for metric_idx, seq_idx in product(range(num_metrics), range(total_sequences)):
            _, _, result = job_with_metric_and_seq(
                metric_idx,
                seq_idx,
                metric_funcs[metric_idx],
                batched_input_flat,
                batched_target_flat,
                mask_flat,
                additional_inputs_flat,
                evaluate_kwargs,
            )
            results_by_metric[metric_idx].append((seq_idx, result))

        # Sort each metric's results by sequence index and convert to arrays
        results_dict = {}
        for metric_idx, metric_name in enumerate(metric_names):
            results_by_metric[metric_idx].sort(key=lambda x: x[0])
            result_list = np.array([result for _, result in results_by_metric[metric_idx]])

            # Reshape back to original batch dimensions
            dim_prior_seq_len = {}
            for idx, item in enumerate(batched_input_shape[:dim_input]):
                dim_prior_seq_len[f"a{idx}"] = item
            results_dict[metric_name] = rearrange(result_list, reversed_einop, **dim_prior_seq_len)

    return results_dict


if __name__ == "__main__":
    import time

    from sklearn.metrics import accuracy_score

    def acc1(input, target):
        return accuracy_score(y_pred=input, y_true=target)

    # case 1
    y_pred = np.array([0, 2, 1, 3])
    y_true = np.array([0, 1, 2, 3])
    mask = np.array([1, 1, 1, 1])
    result = evaluate_on_one_batch(
        y_pred,
        y_true,
        mask,
        [
            acc1,
        ],
    )
    print(f"case 1: {result}")

    # case 2
    y_pred = np.array([[0, 2, 1, 3], [0, 2, 1, 3]])
    y_true = np.array([[0, 1, 2, 3], [0, 1, 2, 3]])
    mask = np.array([[1, 1, 1, 1], [0, 1, 1, 1]])
    result = evaluate_on_one_batch(y_pred, y_true, mask, acc1)
    print(f"case 2: {result}")

    # case 3
    y_pred = np.array(
        [
            [[0, 2, 1, 3], [0, 2, 1, 3], [0, 2, 1, 3], [0, 2, 1, 3]],
            [[0, 2, 1, 3], [0, 2, 1, 3], [0, 2, 1, 3], [0, 2, 1, 3]],
        ]
    )
    y_true = np.array(
        [
            [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]],
            [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]],
        ]
    )
    mask = np.array(
        [
            [[1, 1, 1, 1], [0, 1, 1, 1], [1, 1, 1, 0], [0, 1, 1, 0]],
            [[1, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 0, 0, 1]],
        ]
    )
    result = evaluate_on_one_batch(y_pred, y_true, mask, acc1)
    print(f"case 3: {result}")

    # case 4
    def acc(input, target, a):
        print(a)
        return accuracy_score(y_pred=input, y_true=target)

    y_pred = np.array([[[0, 2, 1, 3], [0, 2, 1, 3], [0, 2, 1, 3], [0, 2, 1, 3]] for _ in range(32)])
    y_true = np.array([[[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]] for _ in range(32)])
    mask = np.array([[[1, 1, 1, 1], [0, 1, 1, 1], [1, 1, 1, 0], [0, 1, 1, 0]] for _ in range(32)])
    start_time = time.time()
    result_slow = evaluate_on_one_batch(y_pred, y_true, mask, [acc, "acc", "micro-f1"], multiprocessing=False, a=12)
    exec_time_slow = time.time() - start_time
    print(f"exec_time_slow: {exec_time_slow}s")

    start_time = time.time()
    result_fast = evaluate_on_one_batch(y_pred, y_true, mask, [acc, "acc", "micro-f1"], multiprocessing=True, a=12)
    exec_time_fast = time.time() - start_time
    # assert (result_slow == result_fast).all()

    # print(f'case 4: {result_fast}')
    print(f"exec_time_fast: {exec_time_fast}s")

    # case 5: test with additional inputs for L1 metric
    from src.toolbox.metrics.l1 import L1_distance_between_two_funcs

    def l1_wrapper(input, target, timestamp):
        return L1_distance_between_two_funcs(input, target, timestamp)

    # Simple test data: seq_len first, then resolution
    seq_len = 4
    resolution = 3
    batch_size = 2

    # batched_input: [batch_size, seq_len, resolution]
    batched_input = np.random.rand(batch_size, seq_len, resolution)
    batched_target = np.random.rand(batch_size, seq_len, resolution)
    mask = np.ones((batch_size, seq_len), dtype=bool)
    # timestamp: [batch_size, seq_len, resolution]
    timestamp = np.random.rand(batch_size, seq_len, resolution)

    result_l1 = evaluate_on_one_batch(
        batched_input,
        batched_target,
        mask,
        l1_wrapper,
        dim_input=1,
        dim_target=1,
        dim_mask=1,
        additional_inputs=[timestamp],
    )
    print(f"case 5 L1 result shape: {result_l1.shape}")
    print(f"case 5 L1 result: {result_l1}")

    # Test with different batch dimensions
    batched_input_3d = np.random.rand(2, 3, seq_len, resolution)
    batched_target_3d = np.random.rand(2, 3, seq_len, resolution)
    mask_3d = np.ones((2, 3, seq_len), dtype=bool)
    timestamp_3d = np.random.rand(2, 3, seq_len, resolution)

    result_l1_3d = evaluate_on_one_batch(
        batched_input_3d,
        batched_target_3d,
        mask_3d,
        l1_wrapper,
        dim_input=2,
        dim_target=2,
        dim_mask=2,
        additional_inputs=[timestamp_3d],
    )
    print(f"case 5 L1 3D result shape: {result_l1_3d.shape}")
    print(f"case 5 L1 3D result: {result_l1_3d}")

    # case 6: Performance comparison with large batches (lightweight metric)
    print("\n" + "=" * 60)
    print("Performance Benchmark: Lightweight Metric (accuracy)")
    print("Note: Multiprocessing may be slower due to overhead")
    print("=" * 60)

    for batch_size_test in [64, 128, 256]:
        y_pred_large = np.array(
            [[[0, 2, 1, 3], [0, 2, 1, 3], [0, 2, 1, 3], [0, 2, 1, 3]] for _ in range(batch_size_test)]
        )
        y_true_large = np.array(
            [[[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]] for _ in range(batch_size_test)]
        )
        mask_large = np.array(
            [[[1, 1, 1, 1], [0, 1, 1, 1], [1, 1, 1, 0], [0, 1, 1, 0]] for _ in range(batch_size_test)]
        )

        # Test without multiprocessing
        start_time = time.time()
        result_seq = evaluate_on_one_batch(y_pred_large, y_true_large, mask_large, "acc", multiprocessing=False)
        time_seq = time.time() - start_time

        # Test with multiprocessing
        start_time = time.time()
        result_mp = evaluate_on_one_batch(y_pred_large, y_true_large, mask_large, "acc", multiprocessing=True)
        time_mp = time.time() - start_time

        speedup = time_seq / time_mp if time_mp > 0 else 0
        print(f"\nBatch size: {batch_size_test}")
        print(f"  Sequential:      {time_seq:.4f}s")
        print(f"  Multiprocessing: {time_mp:.4f}s")
        print(f"  Speedup:         {speedup:.2f}x {'(slower)' if speedup < 1 else ''}")

        # Verify results match
        assert np.allclose(result_seq, result_mp), "Results don't match!"

    # case 7: Performance with computationally expensive metric
    print("\n" + "=" * 60)
    print("Performance Benchmark: Expensive Metric")
    print("Simulating expensive computation per sequence")
    print("=" * 60)

    def expensive_metric(input, target):
        """Simulate a computationally expensive metric."""
        # Perform some expensive operations
        for _ in range(1000):
            _ = np.linalg.norm(input - target)
            _ = np.sin(input) + np.cos(target)
        return accuracy_score(y_pred=input, y_true=target)

    for batch_size_test in [32, 64, 128]:
        y_pred_large = np.array(
            [[[0, 2, 1, 3], [0, 2, 1, 3], [0, 2, 1, 3], [0, 2, 1, 3]] for _ in range(batch_size_test)]
        )
        y_true_large = np.array(
            [[[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]] for _ in range(batch_size_test)]
        )
        mask_large = np.array(
            [[[1, 1, 1, 1], [0, 1, 1, 1], [1, 1, 1, 0], [0, 1, 1, 0]] for _ in range(batch_size_test)]
        )

        # Test without multiprocessing
        start_time = time.time()
        result_seq = evaluate_on_one_batch(
            y_pred_large, y_true_large, mask_large, expensive_metric, multiprocessing=False
        )
        time_seq = time.time() - start_time

        # Test with multiprocessing
        start_time = time.time()
        result_mp = evaluate_on_one_batch(
            y_pred_large, y_true_large, mask_large, expensive_metric, multiprocessing=True
        )
        time_mp = time.time() - start_time

        speedup = time_seq / time_mp if time_mp > 0 else 0
        print(f"\nBatch size: {batch_size_test}")
        print(f"  Sequential:      {time_seq:.4f}s")
        print(f"  Multiprocessing: {time_mp:.4f}s")
        print(f"  Speedup:         {speedup:.2f}x {'✓ FASTER!' if speedup > 1.1 else ''}")

        # Verify results match
        assert np.allclose(result_seq, result_mp), "Results don't match!"

    # case 8: Multiple metrics multiprocessing test
    print("\n" + "=" * 60)
    print("Performance Benchmark: Multiple Expensive Metrics")
    print("Testing unified multiprocessing over (metrics × sequences)")
    print("=" * 60)

    batch_size_test = 64
    y_pred_large = np.array([[[0, 2, 1, 3], [0, 2, 1, 3], [0, 2, 1, 3], [0, 2, 1, 3]] for _ in range(batch_size_test)])
    y_true_large = np.array([[[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]] for _ in range(batch_size_test)])
    mask_large = np.array([[[1, 1, 1, 1], [0, 1, 1, 1], [1, 1, 1, 0], [0, 1, 1, 0]] for _ in range(batch_size_test)])

    # Create multiple expensive metrics
    def expensive_metric_1(input, target):
        for _ in range(1000):
            _ = np.linalg.norm(input - target)
        return accuracy_score(y_pred=input, y_true=target)

    def expensive_metric_2(input, target):
        for _ in range(1000):
            _ = np.sin(input).sum() + np.cos(target).sum()
        return accuracy_score(y_pred=input, y_true=target)

    def expensive_metric_3(input, target):
        for _ in range(1000):
            _ = np.sqrt(np.abs(input - target) + 1).sum()
        return accuracy_score(y_pred=input, y_true=target)

    metric_list = [expensive_metric_1, expensive_metric_2, expensive_metric_3]

    # Test: Sequential vs MP
    print(
        f"\nBatch size: {batch_size_test}, Metrics: {len(metric_list)}, Total jobs: {batch_size_test * len(metric_list)}"
    )
    start_time = time.time()
    result_seq = evaluate_on_one_batch(y_pred_large, y_true_large, mask_large, metric_list, multiprocessing=False)
    time_seq = time.time() - start_time
    print(f"  Sequential:      {time_seq:.4f}s")

    start_time = time.time()
    result_mp = evaluate_on_one_batch(y_pred_large, y_true_large, mask_large, metric_list, multiprocessing=True)
    time_mp = time.time() - start_time
    speedup = time_seq / time_mp if time_mp > 0 else 0
    print(f"  Multiprocessing: {time_mp:.4f}s ({speedup:.2f}x) ✓")

    # Verify all results match
    for metric_name in result_seq.keys():
        assert np.allclose(result_seq[metric_name], result_mp[metric_name]), f"{metric_name} results don't match!"

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("\nSUMMARY:")
    print("- For lightweight metrics: Sequential is often faster")
    print("- For expensive metrics: Multiprocessing provides significant speedup")
    print("- Multiprocessing parallelizes over ALL (metric × sequence) pairs using itertools.product")
    print("- Works for both single and multiple metrics")
    print("- Use mp_threshold parameter to control when MP is enabled")
    print("=" * 60)
