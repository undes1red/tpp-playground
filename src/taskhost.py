import datetime, os, sys, torch, importlib, random
import numpy as np

import torch.distributed as dist
import torch.multiprocessing as mp

from src.taskhost_utils import getLogger

'''
The TaskHost executes tasks using pytorch.multiprocessing. Credits to the neural_stpp created by RTQ Chen from Facebook.
'''
logger = getLogger('TaskHost')

class TaskHost:
    def __init__(self, parser, root_path):
        '''
        Spawn a TaskHost.
        Args:
        * parser:    argparse.ArgumentParser
                     The argument parser. This parser consists subparsers that load different arguments based on the main and sub procedure.
        * root_path: str
                     Where the start.py locates.
        '''
        self.opt = parser.parse_args()
        self.root_path = root_path

        procedure = importlib.import_module('src.' + self.opt.procedure)
        self.opt = getattr(procedure, f'{self.opt.task_category}_postprocess')(self.opt, self.root_path)


    def start(self):
        '''
        All source files related to the specific procedure should locate in src, and the folder name
        should match the given name of the main procedure.
        
        Caveats:
        1. The arguments loader should be named as "main procedure name + sub-procedure name + Arguments"(no whitespace). E.x.: TPP_plotter's has main procedure
           name 'TPP' and sub-proceudre name 'Plotter', so its argument parser name should be 'TPPPlotterArguments'. The argument should inherit the BasicArguments
           in src.arguments.
        2. The name of the entry function should be work().
        '''
        logger.debug(f'Root path: {self.root_path}.')
        logger.info(f'Main procedure name: {self.opt.procedure}. Sub-procedure name: {self.opt.task_category}.')

        '''
        Reproducibility.
        '''
        if self.opt.no_seed:
            import time
            logger.warning(f'Reproducibility only presents when a random seed is present. If you want reproducible results, please ABORT this run ASAP and manually assign a random seed using argument \'--seed\'')
            logger.warning(f'No explicit random seed detected, the framework will spontaneously select a number as the random seed.')
            random.seed(int(time.time()) % 65535)
            self.opt.seed = random.randint(0, 65535)
            logger.info(f'The model prefers {self.opt.seed} this time.')
        else:
            logger.info(f'You request, we follow. We will use number {self.opt.seed} as the random seed.')
        
        '''
        Show and check PyTorch version.
        '''
        logger.info(f'PyTorch Version: {torch.__version__}.')
        # Avoid pytorch issue #36313
        if torch.__version__ == '1.4.0':
            raise logger.exception('Due to the pytorch issue #36313(https://github.com/pytorch/pytorch/issues/36313),\
            several learning rate schedulers including LambdaLR used by this architecture fail to run. Please update PyTorch to 1.5.0 or above.')

        '''
        Check cuda availability. We will force using CPU if cuda is unavailable even the user script wants to use cuda.
        '''
        if self.opt.cuda and not torch.cuda.is_available():
            logger.warning('You expect cuda acceleration but cuda is unavailable in this machine. Please check your cuda configuration and make sure that you have installed pytorch with cuda support.')
            logger.warning('We use cpu now.')
            self.opt.cuda = False
        else:
            logger.warning('We use cuda to speed up model training!')
            logger.info('Found {} CUDA devices.'.format(torch.cuda.device_count()))
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                logger.info('{} \t Memory: {:.2f}GiB'.format(props.name, props.total_memory / (1024**3)))

        # Prepare for multithreading
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(int(np.random.randint(30000, 65535)))

        '''
        Please check https://pytorch.org/docs/stable/notes/randomness.html?highlight=reproducibility for further information about
        reproducibility
        '''
        # set up random seed for various packages
        random.seed(self.opt.seed)
        torch.manual_seed(self.opt.seed)
        np.random.seed(self.opt.seed)
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        # For debug usage
        # torch.autograd.set_detect_anomaly(True)

        try:
            mp.set_start_method("forkserver")
            mp.spawn(self.main, nprocs=self.opt.ngpus, join=True)
        except Exception:
            import traceback
            logger.error(traceback.format_exc())
            sys.exit(1)
    

    def main(self, rank):
        '''
        Multiprocessing training controller.

        Args:
        * rank: int
                Generated and used by mp.spawn().
        '''
        dist.init_process_group("nccl" if self.opt.cuda else 'gloo', rank=rank, world_size=self.opt.ngpus, timeout=datetime.timedelta(minutes=30))

        procedure = importlib.import_module('src.' + self.opt.procedure)

        '''
        The name of the worker should:
        1. be named "main procedure + sub procedure name". E.x.: TPP_plotter's has main procedure name 'TPP' and sub-proceudre name 'Plotter', 
        so its procedure class name should be 'TPPPlotter'. This class does not inherit any class.
        2. present in src/${procedure}/__init__.py.
        '''
        self.worker = getattr(procedure, self.opt.procedure + self.opt.task_category)()

        '''
        Report device properties.
        '''
        self.opt.device = torch.device(f'cuda:{rank:d}' if self.opt.cuda else 'cpu')
    
        try:
            self.worker.work(rank = rank, opt = self.opt)
        except:
            import traceback
            logger.error(traceback.format_exc())
            raise
    
        dist.destroy_process_group()