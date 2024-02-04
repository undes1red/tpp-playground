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

        # The number of Dataloader worker
        self.parser.add_argument('--n_worker', default=4, type=int,
                  help='The number of dataloader workers. For most datasets, multiprocessing might speed up the training procedure. But you should set it to lower value, even 0 \
                      if you meet \'received 0 items of ancdata\' exception.')