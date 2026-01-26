import torch


def bisection_torch(
    max_step,
    bisect_early_stop_threshold,
    bisect_func,
    threshold,
    *args,
    l_val=0.0001,
    r_val=1e6,
    **kwargs,
):
    """
    Bisection Method when the inputs are torch.tensors.
    """
    left = l_val * torch.ones_like(threshold)
    right = r_val * torch.ones_like(threshold)
    check_after_how_many_steps = max_step // 5

    for idx in range(max_step):
        center = (left + right) / 2
        val = bisect_func(center, threshold, *args, **kwargs)
        left = torch.where(val < 0, center, left)
        right = torch.where(val > 0, center, right)
        if (idx + 1) % check_after_how_many_steps == 0 and \
           torch.abs(right - left).max() < bisect_early_stop_threshold:
            return center

    return (left + right) / 2
