import os, argparse
from src.evaluator_arguments import BasicEvaluatorArguments
from src.TPP.utils import suffix


class EvaluatorArguments(BasicEvaluatorArguments):
    def __init__(self, parser, root_path):
        super().__init__(parser)

        self.root_path = root_path        

        # plotter specific
        parser.add_argument('--figure_count', type = int, help='We will select {figure_count} records from training set(if set),\
                                                  test set(if set), and evaluation set(if set), respectively. So there will be\
                                                  {enabled_dataset} * figure_count plots when the plotter finish running.')
        parser.add_argument('--resolution', type=int, default=100, help='How many interpolating points may each time interval have?')
        parser.add_argument('--sample_amount', type=int, default=500, help='The number of samples per dim of a high-dimensional space.')
        parser.add_argument('--mask_rate', type=float, default = 0.0, help='')
        
        # Specfically for the HYPRO dataset preparation.
        parser.add_argument('--number_of_events_hypro', type=int, default = 1, help = 'The number of events that hypro should predict based on the history.')
        parser.add_argument('--number_of_negative_samples', type=int, default = 1, help = 'The number of negative samples that each positive sequence has.')


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

    '''
    Gradient aggergation check
    '''
    if opt.agg_update_step > 1:
        opt.n_training_steps *= opt.agg_update_step

    opt.training_batch_size = 1
    opt.evaluation_batch_size = 1
        
    opt.data_path = os.path.join(root_path, 'data', opt.procedure, opt.dataset_name)
    opt.abs_dataloader_config = os.path.join(root_path, 'config', opt.procedure, opt.model_name, opt.dataloader_config) if opt.dataloader_config else None
    opt.dataloader_config = os.path.basename(opt.abs_dataloader_config) if opt.dataloader_config else None
    opt.abs_model_config = os.path.join(root_path, 'config', opt.procedure, opt.model_name, opt.model_config) if opt.model_config else None
    opt.model_config = os.path.basename(opt.abs_model_config) if opt.model_config else None
    if opt.combine_used_and_current_dataloader_config:
        opt.abs_used_dataloader_config = os.path.join(root_path, 'config', opt.procedure, opt.model_name, opt.training_dataset_name if opt.training_dataset_name is not None else opt.dataset_name, opt.used_dataloader_config) if opt.used_dataloader_config else None
        opt.used_dataloader_config = os.path.basename(opt.used_dataloader_config) if opt.used_dataloader_config else None

    opt.checkpoint_of_this_procedure = os.path.join(root_path, 'model', opt.procedure)
    opt.results_of_this_procedure = os.path.join(root_path, 'results', opt.procedure)
    opt.model_identifier = suffix(opt, 'model_name', 'lr', 'used_batch_size', 'n_training_steps', 'used_dataloader_config', 'model_config')

    return opt