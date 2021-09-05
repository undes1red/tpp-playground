import matplotlib.pyplot as plt
import seaborn as sns
import torch, os
import pandas as pd
from .utils import getLogger

logger = getLogger(name = __file__)

def expand_true_intensity(time, intensity, resolution, opt):
    return true_intensity_dict[opt.dataset_name](time, intensity, resolution)

def expand_model_intensity(model, data, resolution, opt):
    if opt.model_name in ['dwg', 'fullynn']:
        return model.function_prober(data, resolution)
    else:
        raise Exception('This model is incompatible with intensity prober!')

def draw_intensity(model, data, desc, plot_count, opt):
    '''
    Now you should investigate your own model implementations and modify this function.
    Because of the design of training_step() and evaluation_step(), intensity and integral outputs can not be handled automatically.

    data: [time_diff, score, target_intensity]
    '''
    # Intensity probe at all event timestamps.
    intensity_integral, intensity = model(data[0])
    print(intensity.squeeze())
    print(data[2].squeeze())
    mse = torch.nn.functional.mse_loss(intensity.squeeze(), data[2].squeeze())
    logger.info(f'The MSE loss between model intensity and target intensity at event points is {mse.item()}.')

    _, model_intensity, timestamp = expand_model_intensity(model, data[0], 100, opt)
    true_intensity = expand_true_intensity(data[0], data[2], 100, opt)

    # torch.tensor to numpy.array
    model_intensity = model_intensity.squeeze().detach().cpu().numpy()
    timestamp = timestamp.detach().cpu().numpy().cumsum()
    if true_intensity is not None:
        true_intensity = true_intensity.squeeze().detach().cpu().numpy()
    
    if true_intensity is not None:
        df = pd.DataFrame.from_dict(
            {'Time': timestamp, 'Predicted Intensity': model_intensity, 'Truth': true_intensity}
        )
    else:
        df = pd.DataFrame.from_dict(
            {'Time': timestamp, 'Predicted Intensity': model_intensity}
        )

    df_plot = pd.melt(df, 'Time')
    df_plot.columns = ['Time', '', 'Intensity']
    # Draw plot
    fig = plt.figure()
    sns.lineplot(x = 'Time', y = 'Intensity', hue = '', data = df_plot)
    plt.savefig(os.path.join(opt.store_dir, 'intensity_' + desc + '_' + str(plot_count) + '.png'), dpi = 1000)
    plt.close(fig = fig)

def draw_probability(model, data, desc, plot_count, opt):
    pass

def draw(model, data, desc, plot_count, type, opt):
    if type == 'intensity':
        draw_intensity(model, data, desc, plot_count, opt)
    elif type == 'probability':
        draw_probability(model, data, desc, plot_count, opt)
    else:
        raise Exception('Unknown plot type detected!')
    
# The true intensity function definition
def hawkes_1(time, intensity, resolution):
    '''
    Hawkes_1 process: \lambda(t) = \mu + a * b * exp(b(t - t_l))
    In this case, \mu = 0.2, a = 0.8, b = 1, and all of the past events are related to the intensity.

    Args:
    time      : [batch_size, seq_len]
    intensity : [batch_size, seq_len]
              The value of true intensity function.
    resolution: int
    '''
    # hyperparameters
    mu = 0.2
    a = 0.8
    b = 1.0

    batch_size = time.shape[0]
    intensity = torch.cat(
        (torch.zeros((batch_size, 1)), intensity[:, :-1]), dim = -1
    )

    time_multiplier = torch.linspace(0, 1, 100)
    expand_time = time_multiplier * time[:, 1:].unsqueeze(-1)                  # [batch_size, seq_len, resolution]
    true_intensity = intensity.unsqueeze(-1).repeat(1, 1, resolution) - mu + a * b
                                                                               # [batch_size, seq_len, resolution]
    intensity_multiplier_matrix = torch.exp(-b * expand_time)                  # [batch_size, seq_len, resolution]
    expand_true_intensity = true_intensity * intensity_multiplier_matrix + mu  # [batch_size, seq_len, resolution]
    expand_true_intensity = expand_true_intensity.reshape(batch_size, -1)      # [batch_size, seq_len * resolution]
    expand_true_intensity[:, 0:resolution] = mu
    return expand_true_intensity

def hawkes_2(time, intensity, resolution):
    '''
    Hawkes_2 process: \lambda(t) = \mu + a_1 * b_1 * exp(b_1(t - t_l)) + a_2 * b_2 * exp(b_2(t - t_l))
    In this case, \mu = 0.2, a_1 = 0.4, b_1 = 1, a_2 = 0.4, b_2 = 20.0, and all of the past events are related to the intensity.

    Args:
    time      : [batch_size, seq_len]
    intensity : [batch_size, seq_len]
              The value of true intensity function.
    resolution: int
    '''
    # hyperparameters
    mu = 0.2
    a_1 = 0.4
    a_2 = 0.4
    b_1 = 1.0
    b_2 = 20.0

    pass

def poisson(time, intensity, resolution):
    '''
    Poisson process: \lambda(t) = 1
    The intensity function of poisson process is a constant.

    Args:
    time       : [batch_size, seq_len]  (not used in this function)
    intensity  : [batch_size, seq_len]
               The value of true intensity function.
    resolution : int
    '''
    # hyperparameters
    lam = 1

    batch_size, seq_len = intensity.shape
    return torch.ones((batch_size, seq_len * resolution)) * lam                # [batch_size, seq_len * resolution]

def sta_renew(time, intensity, resolution):
    '''
    The stationary renewal process: \lambda(t) = -0.797885*exp(-0.5*(log(t))**2) / (-t + t * erf(0.707107 * log(t)))
    The intensity function only matches the explicitly-given lognorm distribution. please check and modify this function if you use
    another hyperparameters for stationary renewal process during data generation.

    Timestamp 0 will be shifted to a very small value.

    Args:
    time       : [batch_size, seq_len]
    intensity  : [batch_size, seq_len]
               The value of true intensity function.
    resolution : int
    '''

    batch_size, _ = time.shape
    time_multiplier = torch.linspace(0, 1, 100)
    expand_time = time_multiplier * time[:, 1:].unsqueeze(-1)                  # [batch_size, seq_len, resolution]
    expand_time[:, :, 0] += 1e-8
    expand_true_intensity = -0.797885*torch.exp(-0.5*(torch.log(expand_time))**2) / (-expand_time + expand_time * torch.erf(0.707107 * torch.log(expand_time)))
                                                                               # [batch_size, seq_len, resolution]
    expand_true_intensity = expand_true_intensity.reshape(batch_size, -1)      # [batch_size, seq_len * resolution]
    return expand_true_intensity


def self_correct(time, intensity, resolution):
    pass

true_intensity_dict = {
    'hawkes_1': hawkes_1,
    'hawkes_2': hawkes_2,
    'poisson': poisson,
    'stationary_renewal': sta_renew,
    'self_correct': self_correct
}