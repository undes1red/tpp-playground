# The model training script

import argparse
import time
import os
from tqdm import tqdm

import pandas as pd
import torch
import torch.optim as optim

from src.utils import training_steps, getLogger

from torch.utils.data import DataLoader
from src.model.model import TemporalModel
from src.optimizer.optim import ScheduledOptim
from src.data.dataset import CustomDataset

logger = getLogger(__name__)

def cal_performance(intensity, intensity_integral):
    ''' 
        The most important thing: How to make intensity always positive? 
        Or, if not positive, give this model maximum penalty. Hope this can take some effects.
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
        loss = cal_performance(
            intensity=intensity, intensity_integral=intensity_integral
        )
        loss.backward()
        optimizer.step_and_update_lr()

        # note keeping
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
        loss = cal_performance(
            intensity=intensity, intensity_integral=intensity_integral
        )

        # note keeping
        total_loss += loss.item()
        fact += batch[2].sum()

    loss_per_eval = total_loss / evaluation_set_length
    fact = fact / evaluation_set_length
    return loss_per_eval, fact


def train(model, training_data, evaluation_data, test_data, optimizer, device, opt):
    ''' Start training '''

    log_train_file, log_eva_file, log_test_file = None, None, None

    if opt.log:
        log_train_file = os.path.join(opt.log, 'train.log')
        log_eva_file = os.path.join(opt.log, 'evaluate.log')
        log_test_file = os.path.join(opt.log, 'test.log')

        logger.info(f'Training performance will be written to file: {log_train_file} , {log_eva_file} and {log_test_file}')

        with open(log_train_file, 'w') as log_tf, open(log_eva_file, 'w') as log_vf, open(log_test_file, 'w') as log_ef:
            log_tf.write('epoch,loss,gap\n')
            log_vf.write('epoch,loss,gap\n')
            log_ef.write('epoch,loss,gap\n')

    def print_performances(header, loss, start_time, optimizer):
        logger.info('{header:12} loss_value: {loss: 8.5f} ppl: {ppl: 8.5f}, '
              'elapse: {elapse:3.3f} min, average lr: {lr:2.5f}'.format(
                  header=f"({header})", loss=loss, ppl=min(loss, 100),
                  elapse=(time.time() - start_time)/60, lr=optimizer.get_lr()))

    eva_losses = []
    warmup_epoches = opt.n_warmup_steps / len(training_data)

    for epoch_i in range(opt.epoch):
        logger.info('[ Epoch ' + str(epoch_i + 1) + ' ]')

        start = time.time()
        train_loss, fact_tr = train_epoch(model, training_data, optimizer, device)
        print_performances('Training', train_loss, start, optimizer)

        start = time.time()
        eva_loss, fact_ev = eval_epoch(model, evaluation_data, device)
        print_performances('Evaluation', eva_loss, start, optimizer)

        start = time.time()
        test_loss, fact_test = eval_epoch(model, test_data, device)
        print_performances('Test', test_loss, start, optimizer)

        if epoch_i > warmup_epoches:
            eva_losses += [eva_loss]

        checkpoint = {'epoch': epoch_i, 'settings': opt,
                      'model': model.state_dict()}

        if opt.save_model and epoch_i > warmup_epoches:
            if opt.save_mode == 'all':
                model_name = os.path.join(
                    opt.save_model, ('_tr_loss_{tr_loss:3.3f}' + '_ev_loss_{eva_loss:3.3f}' + '.chkpt').format(tr_loss=train_loss, eva_loss=eva_loss))
                torch.save(checkpoint, model_name)
            elif opt.save_mode == 'best':
                model_name = os.path.join(opt.save_model, 'checkpoint.chkpt')
                if eva_loss <= min(eva_losses):
                    torch.save(checkpoint, model_name)
                    logger.info('    - The checkpoint file has been updated.')

        if log_train_file and log_eva_file and log_test_file:
            with open(log_train_file, 'a') as log_tf, open(log_eva_file, 'a') as log_vf, open(log_test_file, 'a') as log_ef:
                log_tf.write('{epoch},{loss: 8.5f},{gap: 8.5f}\n'.format(
                    epoch=epoch_i, loss=train_loss,
                    gap=train_loss - fact_tr))
                log_vf.write('{epoch},{loss: 8.5f},{gap: 8.5f}\n'.format(
                    epoch=epoch_i, loss=eva_loss,
                    gap=eva_loss - fact_ev))
                log_ef.write('{epoch},{loss: 8.5f},{gap: 8.5f}\n'.format(
                    epoch=epoch_i, loss=test_loss,
                    gap=eva_loss - fact_test))


def main():
    ''' 
    Usage:
    See scripts files in directory scripts/ for examples. Check all arguments via 'python3 train.py --help'
    '''

    parser = argparse.ArgumentParser()
    # The Ultimate
    parser.add_argument('--seed', type=int, default=42,
                        help='The global random seed.')
    parser.add_argument('--cuda', action='store_true', 
                        help="Set it to true if you want to use GPU.")

    # Model save
    parser.add_argument('--data_path', default=None, help='Input dataset file path.')
    parser.add_argument('--log', default=None, help='Log file path.')
    parser.add_argument('--save_model', default=None, help='Saved checkpoint file path.')
    parser.add_argument('--save_mode', type=str, choices=['all', 'best'], default='best', help='Store all model checkpoints or only store the best one.')

    # Training procedure related hyperparameters
    parser.add_argument('--epoch', type=int, default=10, help='Epoch number')
    parser.add_argument('-b', '--batch_size', type=int, default=2048, help='Batch size')

    # Model-related hyperparameters
    parser.add_argument('--d_history', type=int, default=32)
    parser.add_argument('--d_intensity', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--rnn_layers', type=int, default=1)
    parser.add_argument('--mlp_layers', type=int, default=3)
    parser.add_argument('--n_warmup_steps', type=int, default=2000)

    # Optimizer-related hyperparameters
    parser.add_argument('--custom_op', action='store_true', help='Set it to true if you want to use your own optimizer or that from third-party packages.')
    parser.add_argument('--op_name', type=str, default='AdamW', help='The name of optimizer. All optimizer hyperparameters are set as default.')
    parser.add_argument('--lr_sched', action='store_true', help='Do you want to use learning rate scheduler? If scheduler is disabled, the warmup settings won\'t come into effect.')
    parser.add_argument('--lr', type=float, default=0.1, help='Input learning rate. The real learning rate could change due to the lr scheduler.')
    parser.add_argument('--num_cycles', type=float, default=0.5)
    parser.add_argument('--last_epoch', type=int, default=-1)

    opt = parser.parse_args()
    # optimizer and learning rate scheduler
    if opt.custom_op:
        import torch_optimizer as top
        if not hasattr(top, opt.op_name) and not hasattr(optim, opt.op_name):
            raise logger.exception(f'The given optimizer {opt.op_name} is not found in neither PyTorch nor pytorch_optimizer. Please check your optimizer settings and try again.')
    else:
        if not hasattr(optim, opt.op_name):
            raise logger.exception(f"The given optimizer {opt.op_name} is not found. Maybe it is a custom optimizer. Please set --custom_op and try again.")
    
    if torch.__version__ == '1.4.0':
        raise logger.exception('Due to pytorch issue #36313(https://github.com/pytorch/pytorch/issues/36313), several learning rate scheduler will fail to run. Please update PyTorch version to 1.5.0 or above.')

    # Reproducibility
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Debug
    torch.autograd.set_detect_anomaly(True)

    if not opt.log and not opt.save_model:
        logger.warning('No experiment result will be saved.')

    # Cuda
    opt.device = torch.device(
        'cuda' if opt.cuda and torch.cuda.is_available() else 'cpu')

    # Create there dirs if they don't exist.
    if not os.path.isdir(opt.log):
        os.makedirs(opt.log)
    if not os.path.isdir(opt.save_model):
        os.makedirs(opt.save_model)

    #========= Loading Dataset =========#

    if opt.data_path:
        training_data, evaluation_data, test_data, training_size = prepare_dataloaders(opt)
    else:
        raise logger.exception("Wrong input data path.")

    logger.info(opt)

    # Load model
    TPP = TemporalModel(
        d_history=opt.d_history,
        d_intensity=opt.d_intensity,
        dropout=opt.dropout,
        rnn_layers=opt.rnn_layers,
        mlp_layers=opt.mlp_layers
    ).to(opt.device)

    # Load optimizer
    optimizer = None
    if hasattr(optim, opt.op_name):
        optimizer = getattr(optim, opt.op_name)(TPP.parameters(), opt.lr)
    else:
        optimizer = top.get(opt.op_name)(TPP.parameters(), opt.lr)
    
    # Due to the complexity of learning rate scheduler, the scheduler is fixed. If you want to use another learning rate scheduler, plz modify it in src.optim.
    sched_optimizer = ScheduledOptim(optimizer, scheduler = opt.lr_sched, num_warmup_steps = opt.n_warmup_steps, 
                                     num_training_steps = training_steps(training_size, opt.epoch, opt.batch_size),
                                     num_cycles = opt.num_cycles, last_epoch = opt.last_epoch)

    train(TPP, training_data, evaluation_data,
          test_data, sched_optimizer, opt.device, opt)


def prepare_dataloaders(opt):
    batch_size = opt.batch_size
    file_names = os.listdir(os.path.expanduser(opt.data_path))
    # A strong assertion
    assert len(file_names) == 3

    data_raw = {}
    is_csv = file_names[0].split('.')[-1] == 'csv'
    try:
        if is_csv:
            for file_name in file_names:
                file, type = file_name.split('.')
                data_raw[file] = pd.read_csv(
                    os.path.join(opt.data_path, file + '.' + type))
        else:
            for file_name in file_names:
                file, type = file_name.split('.')
                data_raw[file] = pd.read_json(
                    os.path.join(opt.data_path, file + '.' + type))
    except:
        raise TypeError(
            f"Wrong datafile format. Please check your data file in {opt.data_path}")

    opt.max_token_seq_len = len(data_raw['train'].iloc[0].history)

    #========= Preparing Model =========#
    train = CustomDataset(data_raw['train'])
    evaluate = CustomDataset(data_raw['evaluate'])
    test = CustomDataset(data_raw['test'])
    size = len(train)

    train_iterator = DataLoader(train, shuffle = True, batch_size=batch_size, num_workers=4)
    evaluation_iterator = DataLoader(
        evaluate, batch_size=batch_size, num_workers=4)
    test_iterator = DataLoader(test, batch_size=batch_size, num_workers=4)

    return train_iterator, evaluation_iterator, test_iterator, size


if __name__ == '__main__':
    main()
