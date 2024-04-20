import sys, torch, importlib, random, re, operator
import numpy as np

from src.taskhost_utils import get_logger, version_check

'''
The TaskHost executes tasks using pytorch.multiprocessing. Credits to the neural_stpp created by RTQ Chen from Facebook.
'''
logger = get_logger('TaskHost')


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

        self.procedure = importlib.import_module('src.' + self.opt.procedure)
        self.opt = getattr(self.procedure, f'{self.opt.task_category}_postprocess')(self.opt, self.root_path)
        self.pytorch_warning_dict = getattr(self.procedure, 'pytorch_version_warnings')
    

    def pytorch_warning(self, version):
        for key, warning_message in self.pytorch_warning_dict.items():
            if version_check(version, key):
                warning, action = warning_message
                if action == 'continue':
                    logger.warning(warning)
                    logger.warning('Continue training.')
                else:
                    logger.exception(warning)
    

    def reproducibility(self):
        '''
        Please check https://pytorch.org/docs/stable/notes/randomness.html?highlight=reproducibility for further information about
        reproducibility
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


        # set up random seed for various packages
        random.seed(self.opt.seed)
        torch.manual_seed(self.opt.seed)
        np.random.seed(self.opt.seed)
        torch.backends.cudnn.benchmark = False
        # Please read documentations and check if you have used any operations which don't have a deterministic implementation before
        # set it to True. 
        torch.use_deterministic_algorithms(False)
        # torch.use_deterministic_algorithms(True)
        # For gradient debug usage.
        # torch.autograd.set_detect_anomaly(True)
    
    
    def cuda(self):
        '''
        Check cuda availability. We will force using CPU if cuda is unavailable even the user script wants to use cuda.
        '''
        if self.opt.cuda and not torch.cuda.is_available():
            logger.warning('You expect cuda acceleration but cuda is unavailable in this machine. Please check your cuda configuration and make sure that you have installed pytorch with cuda support.')
            logger.warning('We use cpu now.')
            self.opt.cuda = False
        elif self.opt.cuda and torch.cuda.is_available():
            logger.warning('We use cuda to speed up model training!')
            logger.info('Found {} CUDA devices.'.format(torch.cuda.device_count()))
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                logger.info('{} \t Memory: {:.2f}GiB'.format(props.name, props.total_memory / (1024**3)))
        else:
            logger.warning('We use cpu.')


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
        logger.info(f'Main procedure name: {self.opt.displayed_procedure_name}. Sub-procedure name: {self.opt.displayed_task_category}.')
        self.reproducibility()
        
        '''
        Show and check PyTorch version.
        '''
        logger.info(f'PyTorch Version: {torch.__version__}.')
        self.pytorch_warning(torch.__version__)
        self.cuda()

        '''
        start the task.
        '''
        self.main()


    def main(self):
        '''
        Multiprocessing training controller.
        '''

        '''
        The name of the worker should:
        1. be named "main procedure + sub procedure name". E.x.: TPP_plotter's has main procedure name 'TPP' and sub-proceudre name 'Plotter', 
        so its procedure class name should be 'TPPPlotter'. This class does not inherit any class.
        2. present in src/${procedure}/__init__.py.
        '''
        self.worker = getattr(self.procedure, self.opt.procedure + self.opt.task_category)()

        '''
        Report device properties.
        '''
        self.opt.device = torch.device(f'cuda' if self.opt.cuda else 'cpu')
    
        self.worker.work(opt = self.opt)

        sys.exit(0)