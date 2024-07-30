class BasicArguments:
    def __init__(self, parser):
        self.parser = parser
        # The Ultimate
        self.parser.add_argument('--no_seed', action='store_true',
                            help='This argument tells our code to randomly select a seed. You can use this option to explore your model\'s robustness.')
        self.parser.add_argument('--seed', type=int, default=32,
                            help='Set global random seed.')
        self.parser.add_argument('--cuda', action='store_true', 
                            help="Set it to true if you want to use GPU to accelerate model training.")
        self.parser.add_argument('--cuda_device', type=int, default=0,
                            help="Select which CUDA device you want to use. Default number is 0. This argument does nothing if --cuda is not set.")
        self.parser.add_argument('--replace', action='store_true', 
                            help="True: Replace existing everything, such as logs, model checkpoints, and results with the new one.\n False: Do not replace.")
        self.parser.add_argument('--fpcounter', action='store_true', 
                            help="True: Enable the FlopCounterMode shipped by PyTorch to calculate how many FLOPS we spend on one task.\n False: disable FlopCounterMode.")

        # The number of Dataloader worker
        self.parser.add_argument('--n_worker', default=4, type=int,
                  help='The number of dataloader workers. For most datasets, multiprocessing might speed up the training procedure. But you should set it to lower value, even 0 \
                      if you meet \'received 0 items of ancdata\' exception.')
        self.parser.add_argument('--sleep', default=0, type=int,
                                 help='This task is delayed and will start in the amount of time you have set.')