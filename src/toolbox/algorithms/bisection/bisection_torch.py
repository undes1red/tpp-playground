import torch


def bisection_torch(
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
    Bisection Method when the inputs are torch.tensors.
    """
    threshold = threshold.unsqueeze(dim=-1)
    left = l_val * torch.ones_like(threshold)
    right = r_val * torch.ones_like(threshold)
    check_after_how_many_steps = max_step // 10
    steps = torch.linspace(0, 1, resolution, device=threshold.device)

    for idx in range(max_step):
        probe_value = steps * (right - left) + left
        # [..., resolution]
        val = bisect_func(probe_value, threshold, *args, **kwargs)
        # [..., resolution]

        # find the biggest negative and smallest positive
        negative_mask = (val < 0)
        index_negative = (negative_mask.sum(dim=-1) - 1).clamp(min=0)
        index_positive = (index_negative + 1).clamp(max=resolution-1)
        left=probe_value.gather(-1, index_negative.unsqueeze(dim=-1))
        right=probe_value.gather(-1, index_positive.unsqueeze(dim=-1))

        if (idx + 1) % check_after_how_many_steps == 0 and \
           torch.abs(right - left).max() < bisect_early_stop_threshold:
            break

    return (left.squeeze(dim=-1) + right.squeeze(dim=-1)) / 2
