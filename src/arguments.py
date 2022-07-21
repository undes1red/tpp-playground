import argparse
class BasicArguments:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        # The Ultimate
        self.parser.add_argument('--no_seed', action='store_true',
                            help='Do not freeze random seed. Use this option if you want to explore your model\'s robustness.')
        self.parser.add_argument('--seed', type=int, default=32,
                            help='Set global random seed.')
        self.parser.add_argument('--cuda', action='store_true', 
                            help="Set it to true if you want to use GPU to accelerate model training.")
        self.parser.add_argument("--ngpus", type=int, default=1,
                            help="If you want to train your model on multiple GPUs, please set this parameter with integer bigger than 1.")

        # The number of Dataloader worker
        self.parser.add_argument('--n_worker', default=0, type=int,
                  help='The number of dataloader workers. For most datasets, multiprocessing can speed up the training procedure. But you should set it to lower value, even 0 \
                      if you meet \'received 0 items of ancdata\' exception.')
                
        # Training procedure related hyperparameters
        self.parser.add_argument('--n_training_steps', type=int, default=10000, help='The number of training steps.')
        self.parser.add_argument('--n_evaluation_steps', type=int, default=200, help='The number of steps that follows a model evaluation.')
        self.parser.add_argument('--n_report_steps', type = int, default=200, help='After a given number of steps, report the current model training status.')
        self.parser.add_argument('--agg_update_step', type=int, default=1, help='The number of minibatches between two adjacent optimizer steps. The number of practical training steps is \
                                                                            agg_update_step * n_training_steps')
        self.parser.add_argument('--n_warmup_steps', type=int, default=2000, 
                            help='The number of warmup steps. Models during warmup won\'t be stored.')

        # wandb support
        self.parser.add_argument('--wandb', action='store_true', help='Use wandb to visualize the training result.')


    def get_args(self):
        return self.parser.parse_args()