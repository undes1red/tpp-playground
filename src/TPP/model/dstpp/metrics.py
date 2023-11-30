import torch

# Time evaluation functions
def t_mae(pred, real):
    return (pred - real).abs()


def time_metric_func(metric_name, pred, real):
    assert len(pred.shape) == len(real.shape)
    return dict_time_metric_funcs[metric_name](pred = pred, real = real)


dict_time_metric_funcs = {
    'mae': t_mae
}



# Continuous mark evaluation functions
def m_euclid(pred, real):
    euclid_dis = (pred - real)**2                                              # [..., dim_marks]
    euclid_dis = torch.sqrt(euclid_dis.sum(dim = -1))                          # [...]
    return euclid_dis


def mark_metric_func(metric_name, pred, real):
    assert len(pred.shape) == len(real.shape)
    return dict_mark_metric_funcs[metric_name](pred = pred, real = real)


dict_mark_metric_funcs = {
    'euclid': m_euclid
}