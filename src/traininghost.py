import datetime, os, sys, torch, importlib, random
import numpy as np

import torch.distributed as dist
import torch.multiprocessing as mp

from .traininghost_utils import getLogger

'''
The TrainingHost executes model training using pytorch multiprocessing backbones, referring to neural_stpp created by Facebook.
'''
logger = getLogger('__TrainingHost__')

class TrainingHost:
    def __init__(self, root_path, procedure_name):
        self.root_path = root_path
        self.procedure_name = procedure_name
        logger.info(f'Root path: {self.root_path}')
        logger.info(f'Procedure name: {self.procedure_name}')

    def start(self):
        '''
        All source files related to the specific procedure should locate in src, and the folder name
        should match the given procedure name.
        
        Caveats:
        1. The arguments loader should be named as "procedure_name + arguments"(no whitespace).
        2. The name of the entry function should be 'train'.
        '''
        procedure = importlib.import_module('src.' + self.procedure_name)
        argument_class_name = self.procedure_name + 'Arguments'
        opt = getattr(procedure, argument_class_name)(self.root_path).get_args()
        self.trainer = getattr(procedure, self.procedure_name + 'Trainer')()

        '''
        Reproducibility
        '''
        if opt.no_seed:
            import time
            logger.warning(f'No explicit random seed is available. Now, the model will choose a number as the random seed by itself.')
            random.seed(int(time.time()) % 65535)
            opt.seed = random.randint(0, 65535)
            logger.info(f'It seems that your model loves {opt.seed} this time.')
        else:
            logger.info(f'You require that we should use number {opt.seed} as the random seed this time.')

        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(int(np.random.randint(10000, 20000)))

        random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        np.random.seed(opt.seed)
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

        try:
            mp.set_start_method("forkserver")
            mp.spawn(self.main, args = (opt.ngpus, opt), nprocs=opt.ngpus, join=True)
        except Exception:
            import traceback
            logger.error(traceback.format_exc())
            sys.exit(1)
    
    def main(self, rank, ngpus, opt):
        '''
        Multiprocessing training controller.
        '''
        dist.init_process_group("nccl" if opt.cuda else 'gloo', rank=rank, world_size=ngpus, timeout=datetime.timedelta(minutes=30))

        '''
        Gradient aggergation check
        '''
        if opt.agg_update_step > 1 and rank == 0:
            logger.warning(f'Gradient aggregation is detected! The number of all training steps is multiplied by {opt.agg_update_step}!')
            opt.n_training_steps *= opt.agg_update_step
            opt.n_evaluation_steps *= opt.agg_update_step
            opt.n_report_steps *= opt.agg_update_step
            opt.n_warmup_steps *= opt.agg_update_step
    
        '''
        Avoid pytorch issue #36313
        '''
        if torch.__version__ == '1.4.0' and rank == 0:
            raise logger.exception('Due to the pytorch issue #36313(https://github.com/pytorch/pytorch/issues/36313),\
            several learning rate schedulers including LambdaLR used by this architecture fail to run. Please update PyTorch to 1.5.0 or above.')

        '''
        Host tries to check if model and log are saved and gives some hints if you don't store any models or logs.(most time you should store them)
        '''
        if not opt.log and not opt.save_model and rank == 0:
            logger.warning('No experiment result will be saved.')
    
        '''
        Report device status
        '''
        opt.device = torch.device(
            f'cuda:{rank:d}' if opt.cuda and torch.cuda.is_available() else 'cpu')
    
        if rank == 0:
            if opt.device.type == 'cuda':
                logger.info('Found {} CUDA devices.'.format(torch.cuda.device_count()))
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    logger.info('{} \t Memory: {:.2f}GiB'.format(props.name, props.total_memory / (1024**3)))
            else:
                logger.info('WARNING: Using device {}'.format(opt.device))
    
        '''
        Create log and model-saving dirs if they are not present.
        '''
        if not os.path.isdir(opt.log):
            os.makedirs(opt.log)
        if not os.path.isdir(opt.save_model):
            os.makedirs(opt.save_model)
    
        try:
            self.trainer.train(rank = rank, opt = opt)
        except:
            import traceback
            logger.error(traceback.format_exc())
            raise
    
        dist.destroy_process_group()