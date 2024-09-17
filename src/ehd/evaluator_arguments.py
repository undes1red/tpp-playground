import os, argparse
from src.evaluator_arguments import BasicEvaluatorArguments
from src.ehd.utils import suffix


class EvaluatorArguments(BasicEvaluatorArguments):
    def __init__(self, parser, root_path):
        super().__init__(parser)

        self.root_path = root_path  

        # New
        # Used to generate path to the MTPP model checkpoints.
        # self.parser.add_argument('--used_procedure', type = str, default = 'TPP', help='Which main procedure does this checkpoint belong to?')
        # self.parser.add_argument('--used_model_name', default=None, help="The MTPP model name.")
        # self.parser.add_argument('--used_model_config', type=str, default = None, help='The name of model config file used during training.')
        # self.parser.add_argument('--log_time', action='store_true')

        # plotter specific
        parser.add_argument('--figure_count', type = int, help='We will select {figure_count} records from training set(if set),\
                                                  test set(if set), and evaluation set(if set), respectively. So there will be \
                                                  {enabled_dataset} * figure_count plots when the plotter finish running.')
        parser.add_argument('--plot_type', type=str, choices=['removed_events',], default = 'intensity', help='Temporal point process only.')
        parser.add_argument('--resolution', type=int, default=100, help='How many interpolating points may each time interval have?')
        parser.add_argument('--sample_amount', type=int, default=500, help='The number of samples per dim of a high-dimensional space.')

        # self identification mark
        self.parser.add_argument('--procedure', type = str, default = 'ehd',
                            help=argparse.SUPPRESS)
        self.parser.add_argument('--displayed_procedure_name', type = str, default = 'Explainable History Distillation',
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

    '''
    Gradient aggergation check
    '''
    if opt.agg_update_step > 1:
        opt.n_training_steps *= opt.agg_update_step
    opt.base_dataset_name = cut_the_dataset_name(opt.dataset_name)

    opt.training_batch_size = 1
    opt.evaluation_batch_size = 1
    # opt.abs_mtpp_model_config = os.path.join(root_path, 'config', opt.used_procedure, opt.used_model_name, opt.used_model_config)
    opt.data_path = os.path.join(root_path, 'data', opt.procedure, opt.dataset_name)
    opt.abs_dataloader_config = os.path.join(root_path, 'config', opt.procedure, opt.model_name, opt.dataloader_config) if opt.dataloader_config else None
    opt.dataloader_config = os.path.basename(opt.abs_dataloader_config) if opt.dataloader_config else None
    opt.abs_model_config = os.path.join(root_path, 'config', opt.procedure, opt.model_name, opt.model_config) if opt.model_config else None
    opt.model_config = os.path.basename(opt.abs_model_config) if opt.model_config else None

    # locate where checkpoints are stored.
    model_hyperparameters = suffix(opt, 'model_name', 'lr', 'used_batch_size', 'n_training_steps', 'used_dataloader_config', 'model_config')
    folder_suffix = 'model_' + model_hyperparameters
    opt.checkpoint_folder = os.path.join(root_path, 'model', opt.procedure, opt.dataset_name, folder_suffix)

    # where figures, records are stored.
    opt.store_dir = os.path.join(root_path, 'results', opt.procedure, opt.dataset_name, 'results_' + model_hyperparameters)

    return opt


def cut_the_dataset_name(name):
    '''
    New format of dataset name: [dataset_name]_[seq_len_x]_[seq_len_h]
    '''
    return name.rsplit('_', maxsplit = 2)[0]