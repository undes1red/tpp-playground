import os, argparse
from src.trainer_arguments import BasicTrainerArguments
from src.NTPP.utils import suffix


class TrainerArguments(BasicTrainerArguments):
    def __init__(self, parser, root_path):
        super().__init__(parser)
        self.root_path = root_path        

        # self identification mark
        self.parser.add_argument('--procedure', type = str, default = 'NTPP',
                            help=argparse.SUPPRESS)
        self.parser.add_argument('--displayed_procedure_name', type = str, default = 'Noted Temporal Point Process',
                            help=argparse.SUPPRESS)
        self.parser.add_argument('--required_worker', type = str, default = 'Trainer',
                            help=argparse.SUPPRESS)
        self.parser.add_argument('--displayed_task_category', type = str, default = 'Model Training',
                            help=argparse.SUPPRESS)


'''
The following functions are preprocessing functions.
'''
def Trainer_postprocess(opt, root_path):
    '''
    Convert relative paths into absolute path.
    '''

    '''
    Gradient aggergation check
    '''
    if opt.agg_update_step > 1:
        opt.n_training_steps *= opt.agg_update_step
        opt.n_evaluation_steps *= opt.agg_update_step
        opt.n_report_steps *= opt.agg_update_step
        opt.n_warmup_steps *= opt.agg_update_step

    opt.abs_model_config = os.path.join(root_path, 'config', opt.procedure, opt.model_name, opt.model_config) if opt.model_config else None
    opt.model_config = os.path.basename(opt.abs_model_config) if opt.model_config else None
    opt.optim_config = os.path.join(root_path, 'config', opt.procedure, opt.optim_config)
    opt.abs_dataloader_config = os.path.join(root_path, 'config', opt.procedure, opt.model_name, opt.dataloader_config) if opt.dataloader_config else None
    opt.dataloader_config = os.path.basename(opt.abs_dataloader_config) if opt.dataloader_config else None
    opt.abs_procedure_config = os.path.join(root_path, 'config', opt.procedure, opt.procedure_config) if opt.procedure_config else None
    opt.procedure_config = os.path.basename(opt.abs_procedure_config) if opt.procedure_config else None
    opt.data_path = os.path.join(root_path, 'data', opt.procedure, opt.dataset_name)
    opt.model_identifier = suffix(opt, 'model_name', 'lr', 'training_batch_size', 'n_training_steps', 'procedure_config', 'dataloader_config', 'model_config')

    return opt