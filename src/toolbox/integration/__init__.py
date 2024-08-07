import torch

from src.toolbox.integration.integration_torch import approximate_integration_torch
from src.toolbox.integration.integration_numpy import approximate_integration_numpy


def approximate_integration(expanded_func_value, expanded_x, dim, only_integral = False, func_val_x_having_same_shape = False):
    if torch.is_tensor(expanded_func_value) and torch.is_tensor(expanded_x):
        return approximate_integration_torch(expanded_func_value, expanded_x, dim, only_integral, func_val_x_having_same_shape)
    else:
        return approximate_integration_numpy(expanded_func_value, expanded_x, dim, only_integral, func_val_x_having_same_shape)
