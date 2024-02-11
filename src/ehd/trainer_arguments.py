import os, argparse
from src.arguments import BasicArguments
from src.ehd.utils import suffix


class ehdTrainerArguments(BasicArguments):
    def __init__(self, parser, root_path):
        super().__init__(parser)

        self.root_path = root_path        
        # Input data
        self.parser.add_argument('--dataset_name', type=str, default=None, help='Name of the used dataset. All datasets should be placed in {root}/data/${main_procedure_name}.')
        self.parser.add_argument('--dataset_type', type=str, default='json', help='The format of the required dataset.')
        # self.parser.add_argument('--dataloader_name', default=None, help='Name of the used dataloader. All dataloaders are stored in *root*/src/TPP/dataloader.')
        # self.parser.add_argument('--dataloader_config', type=str, default=None, help='Relative path to the custom dataloader config file. This absolute file path is {root}/config/{model_name}/{dataloader_config}.')

        # Training procedure related hyperparameters
        self.parser.add_argument('--n_training_steps', type=int, default=10000, help='Training steps used for training the model.')
        self.parser.add_argument('--n_evaluation_steps', type=int, default=200, help='Evaluate the model on evaluation and test datasets per {n_evaluation_steps} steps.')
        self.parser.add_argument('--n_report_steps', type = int, default=200, help='Report the training metrics per {n_report_steps} steps.')
        self.parser.add_argument('--agg_update_step', type=int, default=1, help='The number of minibatches between two adjacent optimizer steps. The number of practical training steps is \
                                                                            agg_update_step * n_training_steps')
        self.parser.add_argument('--n_warmup_steps', type=int, default=2000, 
                            help='The number of warmup steps. We won\'t store any checkpoints during warmup.')

        # wandb support
        self.parser.add_argument('--wandb', action='store_true', help='Use wandb to record and visualize the training procedure.')

        # New
        # Used to generate path to the MTPP model checkpoints.
        self.parser.add_argument('--used_procedure', type = str, default = 'TPP', help='Which main procedure does this checkpoint belong to?')
        self.parser.add_argument('--used_model_name', default=None, help="The MTPP model name.")
        self.parser.add_argument('--used_lr', type=float, default=0.1, help='Used learning rate for training the model.')
        self.parser.add_argument('-utb', '--used_training_batch_size', type=int, default=2048, help='Used batch size of training data.')
        self.parser.add_argument('--used_n_training_steps', type=int, default=10000, help='Used training steps for training the model.')
        self.parser.add_argument('--used_dataloader_name', type=str, default = None, help='The name of dataloader used during training. Caution: we force the EHD model to use the same dataloader used to train the MTPP model.')
        self.parser.add_argument('--used_dataloader_config', type=str, default = None, help='The name of dataloader config file used during training. Caution: we force the EHD model to use the same dataloader config used to train the MTPP model.')
        self.parser.add_argument('--used_model_config', type=str, default = None, help='The name of model config file used during training.')

        # Model save and log management
        self.parser.add_argument('--save_mode', type=str, choices=['all', 'best'], default='best', help='Store all model checkpoints or only store the best one.')
        
        # Training procedure related hyperparameters
        self.parser.add_argument('-tb', '--training_batch_size', type=int, default=2048, help='Batch size of training data.')
        self.parser.add_argument('-eb', '--evaluation_batch_size', type=int, default=2048, help='Batch size of evaluation and test data.')
        self.parser.add_argument('--grad_clip', type=float, default=0.0, help='Clips gradient norm of an iterable of parameters. It only comes info effect when the argument \
                                                                          value is bigger than 0.')
        
        # Model-related hyperparameters
        self.parser.add_argument('--model_name', default=None, help="The model name.")
        self.parser.add_argument('--model_config', type=str, default=None,
                            help="Relative path to the custom model config file used for training. This absolute file path is {root}/config/{model_name}/{dataset_name}/{model_config}.")
        
        # Optimizer-related hyperparameters
        self.parser.add_argument('--optim_config', type=str, default=None,
                            help='The config file that contains optimizer and scheduler settings.')
        self.parser.add_argument('--custom_op', action='store_true', 
                            help='Set it to true if you want to use your own optimizer or that from third-party packages.')
        self.parser.add_argument('--op_name', type=str, default='AdamW', 
                            help='The name of optimizer. All optimizer hyperparameters are set as default.')
        self.parser.add_argument('--lr_sched', action='store_true', 
                            help='Do you want to use learning rate scheduler? If scheduler is disabled, the warmup settings won\'t come into effect.')
        self.parser.add_argument('--lr', type=float, default=0.1, 
                            help='Input learning rate. The real learning rate could change due to the lr scheduler.')
        self.parser.add_argument('--n_cycles', type=float, default=0.5)
        self.parser.add_argument('--last_epoch', type=int, default=-1)

        # self identification mark
        self.parser.add_argument('--procedure', type = str, default = 'ehd',
                            help=argparse.SUPPRESS)
        self.parser.add_argument('--displayed_procedure_name', type = str, default = 'Explainable History Distillation',
                            help=argparse.SUPPRESS)
        self.parser.add_argument('--task_category', type = str, default = 'Trainer',
                            help=argparse.SUPPRESS)
        self.parser.add_argument('--displayed_task_category', type = str, default = 'Model Training',
                            help=argparse.SUPPRESS)


'''
The following functions are preprocessing functions.
'''
def Trainer_postprocess(opt, root_path):
    '''
    Convert relative paths into absolute path.
    '''

    '''
    Gradient aggergation check
    '''
    if opt.agg_update_step > 1:
        opt.n_training_steps *= opt.agg_update_step
        opt.n_evaluation_steps *= opt.agg_update_step
        opt.n_report_steps *= opt.agg_update_step
        opt.n_warmup_steps *= opt.agg_update_step
    opt.base_dataset_name = cut_the_dataset_name(opt.dataset_name)

    # Used MTPP model related
    opt.abs_mtpp_model_config = os.path.join(root_path, 'config', opt.used_procedure, opt.used_model_name, opt.used_model_config)
    opt.used_model_config = os.path.basename(opt.abs_mtpp_model_config)
    opt.data_path = os.path.join(root_path, 'data', opt.procedure, opt.dataset_name)
    opt.log = os.path.join(root_path, 'log', opt.procedure, opt.dataset_name)
    opt.save_model = os.path.join(root_path, 'model', opt.procedure, opt.dataset_name)
    opt.abs_model_config = os.path.join(root_path, 'config', opt.procedure, opt.model_name, opt.model_config) if opt.model_config else None
    opt.model_config = os.path.basename(opt.abs_model_config) if opt.model_config else None
    opt.optim_config = os.path.join(root_path, 'config', opt.procedure, opt.optim_config)
    opt.abs_dataloader_config = os.path.join(root_path, 'config', opt.procedure, opt.model_name, opt.used_dataloader_config) if opt.used_dataloader_config else None
    opt.dataloader_config = os.path.basename(opt.abs_dataloader_config) if opt.used_dataloader_config else None
    opt.dataloader_name = opt.used_dataloader_name

    model_hyperparameters = suffix(opt, 'used_model_name', 'used_lr', 'used_training_batch_size', 'used_n_training_steps', 'dataloader_config', 'used_model_config')
    folder_suffix = 'model_' + model_hyperparameters
    opt.mtpp_checkpoint_dir = os.path.join(root_path, 'model', opt.used_procedure, opt.base_dataset_name, folder_suffix)

    return opt


def cut_the_dataset_name(name):
    '''
    New format of dataset name: [dataset_name]_[seq_len_x]_[seq_len_h]
    '''
    return name.rsplit('_', maxsplit = 2)[0]