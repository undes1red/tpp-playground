import os
from ..arguments import BasicArguments

class TPPArguments(BasicArguments):
    def __init__(self, root_path):
        super().__init__()

        self.root_path = root_path        
        # Input data
        self.parser.add_argument('--dataset_name', type=str, default=None, help='Feeding in dataset name. All datasets should be placed in root/data/input')
        self.parser.add_argument('--dataloader_name', default=None, help='Input dataloader class name.')
        self.parser.add_argument('--dataloader_config', type=str, default=None, help='The name of the dataloader config file. This file should be in directory config/$\{model_name\}.')
        self.parser.add_argument('--custom_collator', action='store_true',\
                help='If your datasets are special, and the default collator doesn\'t meet your requirements, you can write your own collate_fn() as a method in the dataset class and use it by toggling this argument to True.')

        # Model save and log management
        self.parser.add_argument('--save_mode', type=str, choices=['all', 'best'], default='best', help='Store all model checkpoints or only store the best one.')
        
        # Training procedure related hyperparameters
        self.parser.add_argument('-b', '--batch_size', type=int, default=2048, help='Batch size')
        self.parser.add_argument('--grad_clip', type=float, default=0.0, help='Clips gradient norm of an iterable of parameters. It only comes info effect when the argument \
                                                                          value is bigger than 0.')
        
        # Model-related hyperparameters
        self.parser.add_argument('--model_name', default=None, help="The model name.")
        self.parser.add_argument('--model_json', type=str, default=None,
                            help="The path of json file that contains model hyperparameters.")
        
        # Optimizer-related hyperparameters
        self.parser.add_argument('--optim_json', type=str, default=None,
                            help='The path of json file that contains optimizer and scheduler settings.')
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

    def get_args(self):
        return self.relativepath_to_absolutepath(self.parser.parse_args(), self.root_path)

    '''
    The following functions are preprocessing functions.
    '''
    def relativepath_to_absolutepath(self, opt, root_path):
        '''
        Convert relative paths into absolute path.
        '''
        opt.data_path = os.path.join(root_path, 'data', 'inputs', opt.dataset_name)
        opt.log = os.path.join(root_path, 'log', opt.dataset_name)
        opt.save_model = os.path.join(root_path, 'data', 'outputs', opt.dataset_name)
        opt.model_json = os.path.join(root_path, 'config', opt.model_name, opt.model_json)
        opt.optim_json = os.path.join(root_path, 'config', opt.optim_json)
        opt.abs_dataloader_config = os.path.join(root_path, 'config', opt.model_name, opt.dataloader_config) if opt.dataloader_config else None

        return opt