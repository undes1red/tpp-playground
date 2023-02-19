import datetime, os, sys, torch, importlib, random
import numpy as np

import torch.distributed as dist
import torch.multiprocessing as mp

from src.traininghost_utils import getLogger

'''
The TrainingHost executes model training using pytorch multiprocessing backbones, referring to neural_stpp created by Facebook.
'''
logger = getLogger('__TrainingHost__')

class TrainingHost:
    def __init__(self, parser, root_path):
        self.opt = parser.parse_args()
        self.root_path = root_path

        procedure = importlib.import_module('src.' + self.opt.procedure)
        self.opt = getattr(procedure, 'postprocess')(self.opt, self.root_path)

    def start(self):
        '''
        All source files related to the specific procedure should locate in src, and the folder name
        should match the given procedure name.
        
        Caveats:
        1. The arguments loader should be named as "procedure_name + arguments"(no whitespace).
        2. The name of the entry function should be 'train'.
        '''
        logger.info(f'Root path: {self.root_path}')
        logger.info(f'Procedure name: {self.opt.procedure}')

        procedure = importlib.import_module('src.' + self.opt.procedure)
        self.trainer = getattr(procedure, self.opt.procedure + 'Trainer')()

        '''
        Reproducibility.
        '''
        if self.opt.no_seed:
            import time
            logger.warning(f'Reproducibility only presents when a random seed is explicitly given. If you really request reproducible results. Please ABORT this run ASAP and manually assign a random seed using argument \'--seed\'')
            logger.warning(f'No explicit random seed is detected, so the framework will spontaneously select a number as the random seed based on the UNIX timestamp.')
            random.seed(int(time.time()) % 65535)
            self.opt.seed = random.randint(0, 65535)
            logger.info(f'The model loves {self.opt.seed} this time.')
        else:
            logger.info(f'You request that we should use number {self.opt.seed} as the random seed.')

        '''
        Please check https://pytorch.org/docs/stable/notes/randomness.html?highlight=reproducibility for furhter information about
        reproducibility
        '''
        random.seed(self.opt.seed)
        torch.manual_seed(self.opt.seed)
        np.random.seed(self.opt.seed)
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        # For debug usage
        # torch.autograd.set_detect_anomaly(True)

        # Prepare for multithreading
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(int(np.random.randint(10000, 20000)))

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
        '''
        dist.init_process_group("nccl" if self.opt.cuda else 'gloo', rank=rank, world_size=self.opt.ngpus, timeout=datetime.timedelta(minutes=30))

        '''
        Gradient aggergation check
        '''
        if self.opt.agg_update_step > 1 and rank == 0:
            logger.warning(f'Gradient aggregation is detected! The number of all training steps is multiplied by {self.opt.agg_update_step}!')
            self.opt.n_training_steps *= self.opt.agg_update_step
            self.opt.n_evaluation_steps *= self.opt.agg_update_step
            self.opt.n_report_steps *= self.opt.agg_update_step
            self.opt.n_warmup_steps *= self.opt.agg_update_step
    
        '''
        Avoid pytorch issue #36313
        '''
        if torch.__version__ == '1.4.0' and rank == 0:
            raise logger.exception('Due to the pytorch issue #36313(https://github.com/pytorch/pytorch/issues/36313),\
            several learning rate schedulers including LambdaLR used by this architecture fail to run. Please update PyTorch to 1.5.0 or above.')

        '''
        Host tries to check if model and log are saved and gives some hints if you don't store any models or logs.(most time you should store them)
        '''
        if not self.opt.log and not self.opt.save_model and rank == 0:
            logger.warning('No experiment result will be saved. If it is not intended, please check your training script.')
    
        '''
        Report device status
        '''
        self.opt.device = torch.device(
            f'cuda:{rank:d}' if self.opt.cuda and torch.cuda.is_available() else 'cpu')
    
        if rank == 0:
            if self.opt.device.type == 'cuda':
                logger.info('Found {} CUDA devices.'.format(torch.cuda.device_count()))
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    logger.info('{} \t Memory: {:.2f}GiB'.format(props.name, props.total_memory / (1024**3)))
            else:
                logger.info('WARNING: Using device {}'.format(self.opt.device))
    
        try:
            self.trainer.train(rank = rank, opt = self.opt)
        except:
            import traceback
            logger.error(traceback.format_exc())
            raise
    
        dist.destroy_process_group()