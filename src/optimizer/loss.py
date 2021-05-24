import torch
from tqdm import tqdm


def loss_f(intensity, intensity_integral):
    '''
    The definition of loss.
    '''
    intensity, intensity_integral = intensity.squeeze(), intensity_integral.squeeze()

    log_intensity = torch.log(intensity)
    log_p = log_intensity - intensity_integral

    loss = -log_p
    loss = torch.clamp(loss, max=10)
    loss = torch.sum(loss, dim=0)
    return loss

def train_step(model, minibatch, optimizer, device):
    ''' Epoch operation in training phase'''

    model.train()
    optimizer.zero_grad()
    intensity_integral, intensity = model(
            minibatch[0].to(device), minibatch[1].to(device))

    loss = loss_f(
        intensity=intensity, intensity_integral=intensity_integral
    )
    loss.backward()
    optimizer.step_and_update_lr()

    loss = loss.item()
    fact = minibatch[2].sum()

    return loss, fact


def evaluation_step(model, minibatch, device):
    ''' Epoch operation in evaluation phase '''

    model.eval()
    intensity_integral, intensity = model(
        minibatch[0].to(device), minibatch[1].to(device)
    )

    loss = loss_f(
        intensity=intensity, intensity_integral=intensity_integral
    )

    loss = loss.item()
    fact = minibatch[2].sum()

    return loss, fact