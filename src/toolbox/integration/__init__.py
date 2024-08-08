import torch

from src.toolbox.integration.integration_torch import approximate_integration_torch
from src.toolbox.integration.integration_numpy import approximate_integration_numpy


def approximate_integration(expanded_func_value, expanded_x, dim, only_integral = False, func_val_x_having_same_shape = False):
    if torch.is_tensor(expanded_func_value) and torch.is_tensor(expanded_x):
        return approximate_integration_torch(expanded_func_value, expanded_x, dim, only_integral, func_val_x_having_same_shape)
    else:
        return approximate_integration_numpy(expanded_func_value, expanded_x, dim, only_integral, func_val_x_having_same_shape)


if __name__ == '__main__':
    import numpy as np
    import torch

    resolution = 11
    x = np.arange(0, resolution)
    func1 = 2 * x
    func2 = 3 * x
    func3 = 4 * x
    func4 = x**2
    func5 = 1/(1 + x)

    print(f'func1: {func1}')
    print(f'func2: {func2}')
    print(f'func3: {func3}')
    print(f'func4: {func4}')
    print(f'func5: {func5}')

    L1 = approximate_integration(func1, x, dim = -1, only_integral = True)
    print(f'L1: {L1}')
    print('L1 should be around 100.')

    L1 = approximate_integration(func1, x, dim = -1)
    print(f'L1: {L1}')
    print('L1 should be around [  0.   1.   4.   9.  16.  25.  36.  49.  64.  81. 100.].')

    func = np.stack([func1, func2, func3], axis = -1)
    l1_matrix = approximate_integration(func, x, dim = 0)
    print(l1_matrix)
    print('The matrix should be around [[  0.    0.    0. ]\
 [  1.    1.5   2. ]\
 [  4.    6.    8. ]\
 [  9.   13.5  18. ]\
 [ 16.   24.   32. ]\
 [ 25.   37.5  50. ]\
 [ 36.   54.   72. ]\
 [ 49.   73.5  98. ]\
 [ 64.   96.  128. ]\
 [ 81.  121.5 162. ]\
 [100.  150.  200. ]]')

    func = np.stack([func1, func2, func3, func4, func5], axis = -1)
    l1_matrix = approximate_integration(func, x, dim = 0, only_integral = True)
    print(l1_matrix)
    print('The matrix should be around [100.  150.  200., 333.33333, 2.397895272798371]')

    func = torch.stack([torch.from_numpy(func) for func in [func1, func2, func3]], axis = -1)
    l1_matrix = approximate_integration(func, x, dim = 0)
    print(l1_matrix)
    print('The matrix should be around [[  0.    0.    0. ]\
 [  1.    1.5   2. ]\
 [  4.    6.    8. ]\
 [  9.   13.5  18. ]\
 [ 16.   24.   32. ]\
 [ 25.   37.5  50. ]\
 [ 36.   54.   72. ]\
 [ 49.   73.5  98. ]\
 [ 64.   96.  128. ]\
 [ 81.  121.5 162. ]\
 [100.  150.  200. ]]')
    
    func = torch.stack([torch.from_numpy(func) for func in [func1, func2, func3, func4, func5]], axis = -1)
    l1_matrix = approximate_integration(func, x, dim = 0, only_integral = True)
    print(l1_matrix)
    print('The matrix should be around [100.  150.  200., 333.33333, 2.397895272798371].')