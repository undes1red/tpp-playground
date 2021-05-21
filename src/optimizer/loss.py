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

def train_epoch(model, training_data, optimizer, device):
    ''' Epoch operation in training phase'''

    model.train()
    fact, total_loss, training_set_length = 0, 0, len(training_data)

    desc = '  - (Training)   '
    for batch in tqdm(training_data, desc=desc, leave=False):
        # forward
        optimizer.zero_grad()
        intensity_integral, intensity = model(
            batch[0].to(device), batch[1].to(device))

        # backward and update parameters
        loss = loss_f(
            intensity=intensity, intensity_integral=intensity_integral
        )
        loss.backward()
        optimizer.step_and_update_lr()

        total_loss += loss.item()
        fact += batch[2].sum()

    loss_per_train = total_loss / training_set_length
    fact = fact / training_set_length
    return loss_per_train, fact


def eval_epoch(model, evaluation_data, device):
    ''' Epoch operation in evaluation phase '''

    model.eval()
    fact, total_loss, evaluation_set_length = 0, 0, len(evaluation_data)

    desc = '  - (Evaluation) '
    for batch in tqdm(evaluation_data, desc=desc, leave=False):
        # forward
        intensity_integral, intensity = model(
            batch[0].to(device), batch[1].to(device))
        loss = loss_f(
            intensity=intensity, intensity_integral=intensity_integral
        )

        total_loss += loss.item()
        fact += batch[2].sum()

    loss_per_eval = total_loss / evaluation_set_length
    fact = fact / evaluation_set_length
    return loss_per_eval, fact