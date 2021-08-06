# This script is for FullyNN and dwg only.
import argparse, torch, os
import numpy as np
from tqdm import tqdm

from src.model import get_model
from src.data import prepare_dataloaders
from src.utils import read_json, suffix

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
root = os.path.abspath('../..')
print(root)

num_item = 5
max_end = 10
step = 0.01

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type = str, help='The name of the model.')
parser.add_argument('--dataset_name', type = str, help='The path of the dataset.')
parser.add_argument('--dataloader_name', type = str, help='The name of the needed dataloader.')
parser.add_argument('--batch_size', type = int, help='The size of each batch.')
parser.add_argument('--gpu', action='store_true', help='Do you need GPU optimization?')
parser.add_argument('--n_worker', type = int,default = 0, help='The number of worker.')
parser.add_argument('--model_setting', type = str, help='The path of model hyperparameter setting file.')
parser.add_argument('--lr', type = float, help='Learning rate')
parser.add_argument('--n_training_steps', type = int, help='Total training steps we should perform')
opt = parser.parse_args()

# Some post manipulations
model_class = get_model(opt.model_name)
opt.data_path = os.path.join(root, 'data/inputs', opt.dataset_name)
opt.device = 'cuda' if opt.gpu and torch.cuda.is_available() else 'cpu'
_, _, test_dataloader = prepare_dataloaders(opt, rank = 0, train = False, evaluate = False, test = True)
model_hyperparameters = read_json(opt.model_setting)
opt.__dict__.update(model_hyperparameters)
model_suffix = list(model_hyperparameters.keys())
model_ouput_dir_suffix = suffix(opt, 'model_name', 'lr', 'batch_size', 'n_training_steps', *model_suffix)
folder_suffix = "_".join(map(str, model_ouput_dir_suffix.values()))
model_output_dir = os.path.join(root, 'data/outputs', opt.dataset_name, 'output_' + folder_suffix)

model_raw = torch.load(os.path.join(model_output_dir, 'checkpoint.chkpt'))
opt_model = model_raw['settings']
model_state_dict = model_raw['model']

np.random.seed(opt_model.seed)
torch.manual_seed(opt_model.seed)

TPP = model_class(
    device = opt.device,
    **model_hyperparameters
)
TPP.eval()

missing, unexpected = TPP.load_state_dict(model_state_dict)
assert len(missing) == 0
assert len(unexpected) == 0

def bisect_target(model,history,taus):
    return model(history.to(opt.device), torch.from_numpy(taus).to(opt.device))[0].detach().cpu().numpy() - np.log(5)

def median_prediction(model,history,l,r):
    for _ in range(30):
        c = (l+r)/2
        v = bisect_target(model,history,c)
        l = np.where(v<0,c,l)
        r = np.where(v>=0,c,r)

    return (l+r)/2

def mean_absolute_error(model, history, target, *args):
    '''
    The input should be the history, its shape is [batch_size, history_length = 15]
    '''
    l=0.0001*np.ones((history.shape[0], 1), dtype = np.float32)
    r=6500.0*np.ones((history.shape[0], 1), dtype = np.float32)
    tau_pred = median_prediction(model,history,l,r) 
    return np.mean(np.abs(tau_pred-target.cpu().numpy()))


total_mae = 0
desc = '  - (MAE probing)   '
for batch in tqdm(test_dataloader, mininterval=2, desc=desc, leave=False):
    mae = mean_absolute_error(TPP, *batch)
    # note keeping
    total_mae += mae

print(total_mae / len(test_dataloader))