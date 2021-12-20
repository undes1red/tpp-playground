import datetime, logging, os, sys, torch
import numpy as np

import torch.distributed as dist
import torch.multiprocessing as mp
import src.TPP as TPP


'''
All the following two functions constructs the pytorch multiprocessing backbones, referring to neural_stpp created by Facebook.
'''
class TrainingHost:
    def __init__(self, root_path):
        self.logger = getLogger('__TrainingHost__')
        self.root_path = root_path
    
    def start(self):
        opt = TPP.TPParguments(self.root_path).get_args()

        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(int(np.random.randint(10000, 20000)))
        try:
            mp.set_start_method("forkserver")
            mp.spawn(self.main, args = (opt.ngpus, opt), nprocs=opt.ngpus, join=True)
        except Exception:
            import traceback
            self.logger.error(traceback.format_exc())
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
            self.logger.warning(f'Gradient aggregation is detected! The number of practical training steps is multiplied by {opt.agg_update_step}!')
            opt.n_training_steps *= opt.agg_update_step
            opt.n_evaluation_steps *= opt.agg_update_step
            opt.n_report_steps *= opt.agg_update_step
            opt.n_warmup_steps *= opt.agg_update_step
    
        '''
        Host tries to avoid pytorch issue #36313
        '''
        if torch.__version__ == '1.4.0' and rank == 0:
            raise self.logger.exception('Due to the pytorch issue #36313(https://github.com/pytorch/pytorch/issues/36313), several learning rate schedulers including LambdaLR fail to run. Please update PyTorch to 1.5.0 or above.')
        
        '''
        Host tries to check if model and log are saved and gives some hints if you don't store any models or logs.(most time you should store them)
        '''
        if not opt.log and not opt.save_model and rank == 0:
            self.logger.warning('No experiment result will be saved.')
    
        '''
        Report device status
        '''
        opt.device = torch.device(
            f'cuda:{rank:d}' if opt.cuda and torch.cuda.is_available() else 'cpu')
    
        if rank == 0:
            if opt.device.type == 'cuda':
                self.logger.info('Found {} CUDA devices.'.format(torch.cuda.device_count()))
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    self.logger.info('{} \t Memory: {:.2f}GB'.format(props.name, props.total_memory / (1024**3)))
            else:
                self.logger.info('WARNING: Using device {}'.format(opt.device))
    
        '''
        Create there dirs if they don't exist.
        '''
        if not os.path.isdir(opt.log):
            os.makedirs(opt.log)
        if not os.path.isdir(opt.save_model):
            os.makedirs(opt.save_model)
    
        try:
            TPP.train(rank = rank, logger = self.logger, opt = opt)
        except:
            import traceback
            self.logger.error(traceback.format_exc())
            raise
    
        dist.destroy_process_group()


'''
Logger settings are everywhere!
'''
def getEventLogger(name, root):
    logger = logging.getLogger(name)
    if root:
        logger.parent = None
        logger.root = logger

    logger.setLevel(logging.DEBUG)
    if (logger.hasHandlers()):
        logger.handlers.clear()
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(asctime)s [%(filename)s:%(lineno)d]: %(message)s', datefmt = '%Y-%m-%d %H:%M:%S')
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)

    return logger

def getFileLogger(name, file, root):
    logger = logging.getLogger(name)
    if root:
        logger.parent = None
        logger.root = logger

    logger.setLevel(logging.DEBUG)
    if (logger.hasHandlers()):
        logger.handlers.clear()
    # create console handler and set level to debug
    ch = logging.FileHandler(file, mode = 'w')
    ch.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(message)s')
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)

    return logger

def getLogger(name = None, file = None, root = True):
    '''
    Get normal loggers or file loggers.

    Args:
    name: The name of a generated logger
    file: print all logs into the file if set.
    '''
    if file:
        return getFileLogger(name, file, root)
    else:
        return getEventLogger(name, root)

logger_ = getLogger(__name__)

'''
File logger handler.
'''
class FileLogger(object):
    def __init__(self, print_format, **kwargs):
        self.loggers = dict()
        for name, path in kwargs.items():
            self.loggers[name] = getLogger(name, path)
        self.print_format = print_format
        self.print_item = self.print_format.keys()
        self.format_string = ''
        for key in self.print_item:
            self.format_string += '{' + key + self.print_format[key] + '}, '

        # Initial info
        for logger in self.loggers.values():
            logger.info(', '.join(self.print_item))

    def print(self, logger_name, **kwargs):
        logger = self.loggers[logger_name]
        logger.info(self.format_string.format_map(kwargs))