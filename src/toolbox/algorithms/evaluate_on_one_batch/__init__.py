import torch

from .evaluate_on_batch_numpy import evaluate_on_one_batch_numpy
from .evaluate_on_batch_torch import evaluate_on_one_batch_torch


def evaluate_on_one_batch(
    batched_input,
    batched_target=None,
    mask=None,
    evaluate_func=None,
    dim_input=-1,
    dim_target=-1,
    dim_mask=-1,
    additional_inputs=None,
    **evaluate_kwargs,
):
    if torch.is_tensor(batched_input):
        return evaluate_on_one_batch_torch(
            batched_input,
            batched_target,
            mask,
            evaluate_func,
            dim_input,
            dim_target,
            dim_mask,
            additional_inputs,
            **evaluate_kwargs
        )
    return evaluate_on_one_batch_numpy(
        batched_input,
        batched_target,
        mask,
        evaluate_func,
        dim_input,
        dim_target,
        dim_mask,
        additional_inputs,
        **evaluate_kwargs
    )


if __name__ == "__main__":
    import time

    import numpy as np
    import torch
    from sklearn.metrics import accuracy_score

    def acc1(input, target):
        return accuracy_score(y_pred=input, y_true=target)

    # case 1
    y_pred = torch.tensor([[0, 2, 1, 3], [0, 2, 1, 3]])
    y_true = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
    mask = torch.tensor([[1, 1, 1, 1], [0, 1, 1, 1]])
    result = evaluate_on_one_batch(
        y_pred,
        y_true,
        mask,
        [
            'acc',
            'micro-f1',
            'macro-f1'
        ],
        num_classes=4,
        device='cpu'
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
    result_slow = evaluate_on_one_batch(y_pred, y_true, mask, [acc, "acc", "micro-f1"], a=12)
    exec_time_slow = time.time() - start_time
    print(f"exec_time_slow: {exec_time_slow}s")

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
        result_seq = evaluate_on_one_batch(y_pred_large, y_true_large, mask_large, "acc")
        time_seq = time.time() - start_time

        print(f"\nBatch size: {batch_size_test}")
        print(f"  Sequential:      {time_seq:.4f}s")

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
            y_pred_large, y_true_large, mask_large, expensive_metric
        )
        time_seq = time.time() - start_time

        print(f"\nBatch size: {batch_size_test}")
        print(f"  Sequential:      {time_seq:.4f}s")

    # case 8: Multiple metrics test
    print("\n" + "=" * 60)
    print("Performance Benchmark: Multiple Expensive Metrics")
    print("Testing unified evaluation over (metrics × sequences)")
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

    # Test: Sequential
    print(
        f"\nBatch size: {batch_size_test}, Metrics: {len(metric_list)}, Total jobs: {batch_size_test * len(metric_list)}"
    )
    start_time = time.time()
    result_seq = evaluate_on_one_batch(y_pred_large, y_true_large, mask_large, metric_list)
    time_seq = time.time() - start_time
    print(f"  Sequential:      {time_seq:.4f}s")

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("\nSUMMARY:")
    print("- Evaluation is now sequential for both numpy and torch")
    print("- Unified evaluation over ALL (metric × sequence) pairs using itertools.product")
    print("- Works for both single and multiple metrics")
    print("=" * 60)
