import numpy as np


def bisection_numpy(
    max_step,
    bisect_early_stop_threshold,
    resolution,
    bisect_func,
    threshold,
    *args,
    l_val=0.0001,
    r_val=1e6,
    **kwargs,
):
    """
    Bisection Method when the inputs are numpy arrays.
    """
    # Ensure threshold has a trailing dimension for broadcasting
    threshold = np.expand_dims(threshold, axis=-1)

    # Initialize left and right boundaries
    left = np.full_like(threshold, l_val)
    right = np.full_like(threshold, r_val)

    check_after_how_many_steps = max(1, max_step // 10)

    # Create a normalized linear space [0, 1]
    steps = np.linspace(0, 1, resolution)

    for idx in range(max_step):
        # Broadcast steps across the [batch, 1] range of (right - left)
        # probe_value shape: [..., resolution]
        probe_value = steps * (right - left) + left

        # val shape: [..., resolution]
        val = bisect_func(probe_value, threshold, *args, **kwargs)

        # Find the transition point where val goes from negative to positive
        negative_mask = val < 0

        # Count how many negatives exist per row to find the last negative index
        # We use clip (equivalent to clamp) to stay within array bounds
        index_negative = np.clip(negative_mask.sum(axis=-1) - 1, 0, resolution - 1)
        index_positive = np.clip(index_negative + 1, 0, resolution - 1)

        # Advanced indexing to replace torch.gather
        # np.take_along_axis is the direct equivalent to torch.gather
        left = np.take_along_axis(probe_value, np.expand_dims(index_negative, -1), axis=-1)
        right = np.take_along_axis(probe_value, np.expand_dims(index_positive, -1), axis=-1)

        # Early stopping check
        if (idx + 1) % check_after_how_many_steps == 0 and np.max(np.abs(right - left)) < bisect_early_stop_threshold:
            break

    return (left.squeeze(axis=-1) + right.squeeze(axis=-1)) / 2
