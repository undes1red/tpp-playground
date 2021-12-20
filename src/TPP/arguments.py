import argparse, os

class TPParguments:
    def __init__(self, root_path):
        self.root_path = root_path
        self.parser = argparse.ArgumentParser()
        # The Ultimate
        self.parser.add_argument('--no_seed', action='store_true',
                            help='Do not freeze random seed. Use this option if you want to explore your model\'s robustness.')
        self.parser.add_argument('--seed', type=int, default=42,
                            help='Set global random seed.')
        self.parser.add_argument('--cuda', action='store_true', 
                            help="Set it to true if you want to use GPU to accelerate model training.")
        self.parser.add_argument("--ngpus", type=int, default=1,
                            help="If you want to train your model on multiple GPUs, please set this parameter with integer bigger than 1.")
        self.parser.add_argument("--opt_level", type=str, default='O1',
                            help="The optimization level of mixed precision training. Only effective when --fp16 training is enabled.")
        
        # Input data
        self.parser.add_argument('--dataset_name', type=str, default=None, help='Feeding in dataset name. All datasets should be placed in root/data/input')
        self.parser.add_argument('--dataloader_name', default=None, help='Input dataloader class name.')
        self.parser.add_argument('--dataloader_config', type=str, default=None, help='The name of the dataloader config file. This file should be in directory config/$\{model_name\}.')
        self.parser.add_argument('--n_worker', default=8, type=int,
                  help='The number of dataloader workers. For most datasets, multiprocessing can speed up the training procedure. But you should set it to lower value, even 0 \
                      if you meet \'received 0 items of ancdata\' exception.')
        
        # Model save and log management
        self.parser.add_argument('--save_mode', type=str, choices=['all', 'best'], default='best', help='Store all model checkpoints or only store the best one.')
        self.parser.add_argument('--wandb', action='store_true', help='Use wandb to visualize the training result.')
        
        # Training procedure related hyperparameters
        self.parser.add_argument('--n_training_steps', type=int, default=10000, help='The number of training steps.')
        self.parser.add_argument('--n_evaluation_steps', type=int, default=200, help='The number of steps that follows a model evaluation.')
        self.parser.add_argument('--n_report_steps', type = int, default=200, help='After a given number of steps, report the current model training status.')
        self.parser.add_argument('-b', '--batch_size', type=int, default=2048, help='Batch size')
        self.parser.add_argument('--agg_update_step', type=int, default=1, help='The number of minibatches between two adjacent optimizer steps. The number of practical training steps is \
                                                                            agg_update_step * n_training_steps')
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
        self.parser.add_argument('--n_warmup_steps', type=int, default=2000, 
                            help='The number of warmup steps. Models during warmup won\'t be stored.')
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