import torch
import numpy as np

def hawkes_1(time, intensity, resolution, device):
    '''
    Hawkes_1 process: \lambda(t) = \mu + a * b * exp(-b(t - t_l))
    In this case, \mu = 0.2, a = 0.8, b = 1, and all past events affect the intensity.

    Args:
    time      : [batch_size, seq_len + 1]
    intensity : [batch_size, seq_len]
              The value of true intensity function.
    resolution: int
    device: conduct all computations on cpu, gpu, or other devices
    '''
    # hyperparameters
    mu = 0.2
    a = 0.8
    b = 1.0

    batch_size = time.shape[0]
    intensity = torch.cat(
        (torch.zeros((batch_size, 1), device = device), intensity[:, :-1]), dim = -1
    )

    time_multiplier = torch.linspace(0, 1, resolution, device = device)
    expand_time = time_multiplier * time[:, 1:].unsqueeze(-1)                  # [batch_size, seq_len, resolution]
    true_intensity = intensity.unsqueeze(-1).repeat(1, 1, resolution) - mu + a * b
                                                                               # [batch_size, seq_len, resolution]
    intensity_multiplier_matrix = torch.exp(-b * expand_time)                  # [batch_size, seq_len, resolution]
    expand_true_intensity = true_intensity * intensity_multiplier_matrix + mu  # [batch_size, seq_len, resolution]
    expand_true_intensity = expand_true_intensity.reshape(batch_size, -1)      # [batch_size, seq_len * resolution]
    expand_true_intensity[:, 0:resolution] = mu
    return expand_true_intensity

def hawkes_1_integral(time, intensity, resolution, device):
    '''
    Hawkes_1 process: \Lambda(t) = \mu * (t - t_l) + a - a * exp(-b(t - t_l)). When t = t_l, \Lambda(t) = 0.
    Hyperparameters that are used here follow what they are in function hawkes_1.

    Args:
    time      : [batch_size, seq_len + 1]
    intensity : [batch_size, seq_len]
              The value of true intensity function.
    resolution: int
    device: conduct all computations on cpu, gpu, or other devices
    '''
    # hyperparameters
    mu = 0.2
    a = 0.8
    b = 1.0
    
    # Integral part 1
    batch_size = time.shape[0]
    time_multiplier = torch.linspace(0, 1, resolution, device = device)
    expand_time = time_multiplier * time[:, 1:].unsqueeze(-1)                  # [batch_size, seq_len, resolution]
    mu_integral = mu * expand_time                                             # [batch_size, seq_len, resolution]
    basic_exponential_integral = a - a * torch.exp(-b * expand_time)           # [batch_size, seq_len, resolution]

    # Integral part 2
    table = torch.diag_embed(time[:, 2:-1], offset = -2)                       # [batch_size, seq_len, seq_len]
    table = torch.cumsum(table, dim = -2)                                      # [batch_size, seq_len, seq_len]
    reversed_cumsum_of_table = torch.cumsum(table.flip(-1), dim = -1).flip(-1) # [batch_size, seq_len, seq_len]
    table_mask = (table != 0).int()                                            # [batch_size, seq_len, seq_len]
    reversed_cumsum_of_table *= table_mask                                     # [batch_size, seq_len, seq_len]
    historical_multiplier = torch.exp(-b * reversed_cumsum_of_table)           # [batch_size, seq_len, seq_len]
    historical_multiplier *= table_mask                                        # [batch_size, seq_len, seq_len]

    historical_integral = basic_exponential_integral.unsqueeze(-1) * historical_multiplier.unsqueeze(-2)
                                                                               # [batch_size, seq_len, resolution, seq_len]
    historical_integral = torch.sum(historical_integral, dim = -1)             # [batch_size, seq_len, resolution]

    # Get the integral
    expand_true_integral = mu_integral + basic_exponential_integral + historical_integral
                                                                               # [batch_size, seq_len, resolution]
    expand_true_integral = expand_true_integral.reshape(batch_size, -1)        # [batch_size, seq_len * resolution]
    expand_true_integral[:, 0:resolution] = mu_integral[:, 0, :]               # [batch_size, seq_len * resolution]

    return expand_true_integral


'''
Hawkes process whose intensity function has multiple kernels.
'''
def hawkes_2(time, intensity, resolution, device):
    '''
    Hawkes_2 process: \lambda(t) = \mu + a_1 * b_1 * exp(-b_1(t - t_l)) + a_2 * b_2 * exp(-b_2(t - t_l))
    In this case, \mu = 0.2, a_1 = 0.4, b_1 = 1, a_2 = 0.4, b_2 = 20.0, and all past events affect the intensity.

    It seems that we have no choice but to solve the intensity iteratively.

    Args:
    time      : [batch_size, seq_len + 1]
    intensity : [batch_size, seq_len]
              The value of true intensity function.
    resolution: int
    device: conduct all computations on cpu, gpu, or other devices
    '''
    # hyperparameters
    mu = 0.2
    a_1 = 0.4
    a_2 = 0.4
    b_1 = 1.0
    b_2 = 20.0

    batch_size, seq_len = time.shape
    seq_len -= 1
    p1d = (1, 0, 0, 0)

    expand_true_intensity = torch.ones((batch_size, seq_len, resolution), device = device) * mu
                                                                               # [batch_size, seq_len, resolution]
    expand_time = (time.unsqueeze(-1) / (resolution - 1)).repeat(1, 1, resolution - 1)
                                                                               # [batch_size, seq_len + 1, resolution - 1]
    expand_time = torch.nn.functional.pad(expand_time, p1d)                    # [batch_size, seq_len + 1, resolution]

    time_cumsum = torch.cumsum(expand_time.reshape(batch_size, -1), dim = -1)  # [batch_size, (seq_len + 1) * resolution]
    time_cumsum = time_cumsum.reshape(batch_size, seq_len + 1, resolution)     # [batch_size, (seq_len + 1), resolution]
    for seq_index in range(2, seq_len + 1):
        expand_batch_time = time_cumsum[:, seq_index:, :] - time_cumsum[:, seq_index, 0]
                                                                               # [batch_size, seq_len - seq_index + 1, resolution]
        expand_intensity_add = a_1 * b_1 * torch.exp(-b_1 * expand_batch_time) + a_2 * b_2 * torch.exp(-b_2 * expand_batch_time)
                                                                               # [batch_size, seq_len - seq_index + 1, resolution]
        p2d = (0, 0, seq_index - 1, 0)
        expand_true_intensity += torch.nn.functional.pad(expand_intensity_add, p2d)
                                                                               # [batch_size, seq_len, resolution]
    
    expand_true_intensity[:, 0, :] = mu
    expand_true_intensity = expand_true_intensity.reshape(batch_size, seq_len * resolution)
                                                                               # [batch_size, seq_len * resolution]
    return expand_true_intensity

def hawkes_2_integral(time, intensity, resolution, device):
    '''
    Hawkes_2 process: \Lambda(t) = \mu * (t - t_l) + a_1 - a_1 * exp(-b_1(t - t_l)) + a_2 - a_2 * exp(-b_2(t - t_l)).
    When t = t_l, \Lambda(t) = 0.
    Hyperparameters that are used here follow what they are in function hawkes_2.

    Args:
    time      : [batch_size, seq_len + 1]
    intensity : [batch_size, seq_len]
              The value of true intensity function.
    resolution: int
    device: conduct all computations on cpu, gpu, or other devices
    '''
    # hyperparameters
    mu = 0.2
    a_1 = 0.4
    a_2 = 0.4
    b_1 = 1.0
    b_2 = 20.0

    # Integral part 1
    batch_size = time.shape[0]
    time_multiplier = torch.linspace(0, 1, resolution, device = device)
    expand_time = time_multiplier * time[:, 1:].unsqueeze(-1)                  # [batch_size, seq_len, resolution]
    mu_integral = mu * expand_time                                             # [batch_size, seq_len, resolution]
    basic_exponential_integral_1 = a_1 - a_1 * torch.exp(-b_1 * expand_time)   # [batch_size, seq_len, resolution]
    basic_exponential_integral_2 = a_2 - a_2 * torch.exp(-b_2 * expand_time)   # [batch_size, seq_len, resolution]
    basic_exponential_integral = basic_exponential_integral_1 + basic_exponential_integral_2
                                                                               # [batch_size, seq_len, resolution]

    # Integral part 2
    table = torch.diag_embed(time[:, 2:-1], offset = -2)                       # [batch_size, seq_len, seq_len]
    table = torch.cumsum(table, dim = -2)                                      # [batch_size, seq_len, seq_len]
    reversed_cumsum_of_table = torch.cumsum(table.flip(-1), dim = -1).flip(-1) # [batch_size, seq_len, seq_len]
    table_mask = (table != 0).int()                                            # [batch_size, seq_len, seq_len]
    reversed_cumsum_of_table *= table_mask                                     # [batch_size, seq_len, seq_len]
    historical_multiplier_1 = torch.exp(-b_1 * reversed_cumsum_of_table)       # [batch_size, seq_len, seq_len]
    historical_multiplier_1 *= table_mask                                      # [batch_size, seq_len, seq_len]
    historical_multiplier_2 = torch.exp(-b_2 * reversed_cumsum_of_table)       # [batch_size, seq_len, seq_len]
    historical_multiplier_2 *= table_mask                                      # [batch_size, seq_len, seq_len]

    historical_integral_1 = basic_exponential_integral_1.unsqueeze(-1) * historical_multiplier_1.unsqueeze(-2)
                                                                               # [batch_size, seq_len, resolution, seq_len]
    historical_integral_2 = basic_exponential_integral_2.unsqueeze(-1) * historical_multiplier_2.unsqueeze(-2)
                                                                               # [batch_size, seq_len, resolution, seq_len]
    historical_integral = torch.sum(historical_integral_1, dim = -1) + torch.sum(historical_integral_2, dim = -1)
                                                                               # [batch_size, seq_len, resolution]

    # Get the integral
    expand_true_integral = mu_integral + basic_exponential_integral + historical_integral
                                                                               # [batch_size, seq_len, resolution]
    expand_true_integral = expand_true_integral.reshape(batch_size, -1)        # [batch_size, seq_len * resolution]
    expand_true_integral[:, 0:resolution] = mu_integral[:, 0, :]               # [batch_size, seq_len * resolution]

    return expand_true_integral

'''
Time-independent poisson process
'''
def poisson(time, intensity, resolution, device):
    '''
    Poisson process: \lambda(t) = 1
    The intensity function of poisson process is a constant.

    Args:
    time       : [batch_size, seq_len + 1]  (not used in this function)
    intensity  : [batch_size, seq_len]
               The value of true intensity function.
    resolution : int
    device: conduct all computations on cpu, gpu, or other devices
    '''
    # hyperparameters
    lam = 1

    batch_size, seq_len = intensity.shape
    return torch.ones((batch_size, seq_len * resolution), device = device) * lam
                                                                               # [batch_size, seq_len * resolution]

def poisson_integral(time, intensity, resolution, device):
    '''
    Poisson process: \lambda(t) = 1 and \Lambda(t) = t (\Lambda(t) is the integral of \lambda(t))

    Args:
    time       : [batch_size, seq_len + 1]  (not used in this function)
    intensity  : [batch_size, seq_len]
               The value of true intensity function.
    resolution : int
    device: conduct all computations on cpu, gpu, or other devices
    '''
    # hyperparameters
    lam = 1

    batch_size, _ = intensity.shape
    time_multiplier = torch.linspace(0, 1, resolution, device = device)
    expand_time = time_multiplier * time[:, 1:].unsqueeze(-1)                  # [batch_size, seq_len, resolution]

    return (expand_time * lam).reshape(batch_size, -1)                         # [batch_size, seq_len * resolution]


'''
Stationary renewal process, whose probability distribution instead of intensity function is defined.
'''
def stationary_renew(time, intensity, resolution, device):
    '''
    The stationary renewal process: \lambda(t) = -0.797885*exp(-0.5*(log(t))**2) / (-t + t * erf(0.707107 * log(t)))
    The intensity function only matches the explicitly given lognorm distribution used in the synthetic data generator. 
    Please check and modify this function if you want to use another hyperparameter set for the stationary renewal process during data generation.

    Timestamp 0 will be shifted to a very small value.

    Args:
    time       : [batch_size, seq_len + 1]
    intensity  : [batch_size, seq_len]
               The value of true intensity function.
    resolution : int
    device: conduct all computations on cpu, gpu, or other devices
    '''

    batch_size, _ = time.shape
    time_multiplier = torch.linspace(0, 1, resolution, device = device)
    expand_time = time_multiplier * time[:, 1:].unsqueeze(-1)                  # [batch_size, seq_len, resolution]
    expand_time[:, :, 0] += 1e-8
    expand_true_intensity = -0.797885*torch.exp(-0.5*(torch.log(expand_time))**2) / (-expand_time + expand_time * torch.erf(0.707107 * torch.log(expand_time)))
                                                                               # [batch_size, seq_len, resolution]
    expand_true_intensity = expand_true_intensity.reshape(batch_size, -1)      # [batch_size, seq_len * resolution]
    return expand_true_intensity

def stationary_renew_probability(time, intensity, resolution, device):
    '''
    We won't implement the integral of stationary renewal's intensity function.
    We will directly use the distribution, instead.

    Args:
    time       : [batch_size, seq_len + 1]
                 The original timestamp sequence.
    intensity  : [batch_size, seq_len]
                 The value of ture intensity function at all event points.
    resolution : int
    device: conduct all computations on cpu, gpu, or other devices
    '''
    # hyperparameter
    s = np.sqrt(1)
    mu = 0

    batch_size = time.shape[0]
    from scipy.stats import lognorm
    time_multiplier = torch.linspace(0, 1, resolution, device = device)        # [resolution]
    expand_time = time_multiplier * time[:, 1:].unsqueeze(-1)                  # [batch_size, seq_len, resolution]
    distribution_values = lognorm.pdf(expand_time.cpu().numpy(), s = s, scale = np.exp(mu))
                                                                               # [batch_size, seq_len, resolution]
    distribution_values = torch.from_numpy(distribution_values).reshape(batch_size, -1)
                                                                               # [batch_size, seq_len * resolution]
    return distribution_values


'''
Self-correct process, which the latest events would drastically decrease the probability of next events in a small time period.
'''
def self_correct(time, intensity, resolution, device):
    '''
    Self correct process has a iterative intensity function. \lambda(t) = exp(mu * tau - alpha * N)
    N is the number of happened events.

    Args:
    time       : [batch_size, seq_len + 1]
    intensity  : [batch_size, seq_len]
               The value of true intensity function when a event happens.
    resolution : int
    device: conduct all computations on cpu, gpu, or other devices
    '''
    # Hyperparameters
    mu = 1
    alpha = 1

    batch_size = time.shape[0]
    time_multiplier = torch.linspace(0, 1, resolution, device = device)
    shift_intensity = torch.cat((torch.ones(batch_size, 1, device = device), intensity[:, :-1]), dim = -1)
                                                                               # [batch_size, seq_len]
    start_intensity = shift_intensity / torch.exp(torch.tensor(alpha))         # [batch_size, seq_len]
    expand_time = time_multiplier * time[:, 1:].unsqueeze(-1)                  # [batch_size, seq_len, resolution]
    start_intensity = start_intensity.unsqueeze(-1).repeat(1, 1, resolution)   # [batch_size, seq_len, resolution]
    expand_intensity = start_intensity * torch.exp(mu * expand_time)           # [batch_size, seq_len, resolution]
    expand_intensity = expand_intensity.reshape(batch_size, -1)                # [batch_size, seq_len * resolution]

    return expand_intensity

def self_correct_integral(time, intensity, resolution, device):
    '''
    self correct process has intensity function: \lambda(t) = exp(mu * tau - alpha * N)
    N is the number of happened events. Omi et al. claim self correct process doesn't aggregate intensity functions of all
    historical events, but it does just like the Hawkes process.

    Args:
    time       : [batch_size, seq_len + 1]
    intensity  : [batch_size, seq_len]
               The value of true intensity function when a event happens.
    resolution : int
    device: conduct all computations on cpu, gpu, or other devices
    '''
    # hyperparameters
    mu = 1
    alpha = 1
    
    time_interval = time[:, 1:]                                                # [batch_size, seq_len]
    batch_size, seq_len = time_interval.shape
    N = torch.arange(0, seq_len, device = device)\
             .repeat(batch_size, 1)\
             .repeat_interleave(resolution, dim = -1)\
             .reshape(batch_size, seq_len, -1)                                 # [batch_size, seq_len, resolution]
    time_multiplier = torch.linspace(0, 1, resolution, device = device)
    expand_time = time_multiplier * time_interval.unsqueeze(-1)                # [batch_size, seq_len, resolution]
    historical_part = torch.exp(mu * time[:, :-1])                             # [batch_size, seq_len]
    historical_part = torch.cumprod(historical_part, dim = -1)                 # [batch_size, seq_len]
    historical_part = historical_part.repeat_interleave(resolution, dim = -1)  # [batch_size, seq_len * resolution]
    historical_part = historical_part.reshape(batch_size, seq_len, resolution) # [batch_size, seq_len, resolution]
    integral = (torch.exp(mu * expand_time - alpha * N) - torch.exp(-alpha * N))/mu * historical_part
                                                                               # [batch_size, seq_len, resolution]
    integral = integral.reshape(batch_size, -1)                                # [batch_size, seq_len * resolution]
    return integral

true_intensity_dict = {
    'hawkes_1': hawkes_1,
    'hawkes_2': hawkes_2,
    'poisson': poisson,
    'stationary_renewal': stationary_renew,
    'self_correct': self_correct,
}

true_probability_dict = {
    'hawkes_1': [hawkes_1, hawkes_1_integral],
    'hawkes_2': [hawkes_2, hawkes_2_integral],
    'poisson': [poisson, poisson_integral],
    'stationary_renewal': [stationary_renew_probability],
    'self_correct': [self_correct, self_correct_integral],
}

'''
custom metrics
'''
def L1_distance(x, y, timestamp, resolution):
    '''
    This function calculates the L^1 distance between two functions.
    Input:
    1. x:          function values
                   [seq_len * resolution, num_events]
    2. y:          function values
                   the number of points from [t_{i - 1}, t_i]
    3. time:       \Delta t
                   the number of event types
    '''

    function_interval = np.abs(x - y).reshape(-1, resolution)[:, :-1]          # [seq_len, resolution - 1]
    timestamp = timestamp.reshape(-1, resolution)                              # [seq_len, resolution]
    timestamp = np.diff(timestamp, axis = -1)                                  # [seq_len, resolution - 1]

    L1 = (function_interval * timestamp).sum()

    # round up the value smaller than 1e-6
    if L1 < 1e-6:
        L1 = 0

    return L1