import os
import argparse

from src.evaluator_arguments import BasicEvaluatorArguments
from src.TPP.utils import suffix


class EvaluatorArguments(BasicEvaluatorArguments):
    def __init__(self, parser, root_path):
        super().__init__(parser)

        self.root_path = root_path

        # identification mark
        self.parser.add_argument('--procedure', type = str, default = 'TPP',
                            help=argparse.SUPPRESS)
        self.parser.add_argument('--displayed_procedure_name', type = str, default = 'Temporal Point Process',
                            help=argparse.SUPPRESS)
        self.parser.add_argument('--required_worker', type = str, default = 'Evaluator',
                            help=argparse.SUPPRESS)
        self.parser.add_argument('--displayed_task_category', type = str, default = 'Model Evaluation',
                            help=argparse.SUPPRESS)
        

'''
The following functions are preprocessing functions.
'''
def Evaluator_postprocess(opt, root_path):
    '''
    Convert relative paths into absolute path.
    '''

    # Gradient aggergation check
    if opt.agg_update_step > 1:
        opt.n_training_steps *= opt.agg_update_step

    opt.training_batch_size = 1
    opt.evaluation_batch_size = 1
        
    opt.data_path = os.path.join(root_path, 'data', opt.procedure, opt.dataset_name)
    
    opt.abs_dataloader_config = os.path.join(root_path, 'config', opt.procedure, opt.model_name, opt.dataloader_config) if opt.dataloader_config else None
    opt.abs_procedure_config = os.path.join(root_path, 'config', opt.procedure, opt.procedure_config) if opt.procedure_config else None
    
    opt.procedure_config = os.path.basename(opt.abs_procedure_config) if opt.procedure_config else None
    opt.dataloader_config = os.path.basename(opt.abs_dataloader_config) if opt.dataloader_config else None
    
    opt.abs_model_config = os.path.join(root_path, 'config', opt.procedure, opt.model_name, opt.model_config) if opt.model_config else None
    opt.model_config = os.path.basename(opt.abs_model_config) if opt.model_config else None
    
    opt.abs_task_config = os.path.join(root_path, 'config', opt.procedure, opt.model_name, opt.task_config) if opt.task_config else None
    opt.task_config = os.path.basename(opt.abs_task_config) if opt.abs_task_config else None
    
    if opt.combine_used_and_current_dataloader_config:
        opt.abs_used_dataloader_config = os.path.join(root_path, 'config', opt.procedure, opt.model_name, opt.training_dataset_name if opt.training_dataset_name is not None else opt.dataset_name, opt.used_dataloader_config) if opt.used_dataloader_config else None
        opt.used_dataloader_config = os.path.basename(opt.used_dataloader_config) if opt.used_dataloader_config else None

    opt.checkpoint_of_this_procedure = os.path.join(root_path, 'model', opt.procedure)
    opt.results_of_this_procedure = os.path.join(root_path, 'results', opt.procedure)
    opt.model_identifier = suffix(opt, 'model_name', 'lr', 'used_batch_size', 'n_training_steps', 'used_procedure_config', 'used_dataloader_config', 'model_config')
    opt.task_identifier = suffix(opt, 'procedure_config', 'dataloader_config', 'task_config')

    return opt