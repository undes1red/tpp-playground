import os

# Hope we can get rid of absolute path in training scripts.
root_path = os.path.dirname(os.path.abspath(__file__))

main_procedure_translator = {
    # Temporal Point Process
    'TPP_train': 'TPP',
    'TPP_evaluate': 'TPP',

    # Noted Temporal Point Process
    'NTPP_train': 'NTPP',
    'NTPP_evaluate': 'NTPP',

    # Long-horizon Temporal Point Process
    'LH_train': 'LH',
    'LH_evaluate': 'LH',

    # Missing Data Imputation with Temporal Point Process
    'MDI_train': 'MDI',
    'MDI_evaluate': 'MDI',

    # Outlier Detection with Temporal Point Process
    'OD_train': 'OD',
    'OD_evaluate': 'OD',

    # Explainable History Distillation.
    'ehd_train': 'ehd',
    'ehd_evaluate': 'ehd',

    'Transformers': 'Transformers',
    'fakenews': 'fakenews'
}

sub_procedure_translator = {
    # Temporal Point Process
    'TPP_train': 'Trainer',
    'TPP_evaluate': 'Evaluator',

    # Noted Temporal Point Process
    'NTPP_train': 'Trainer',
    'NTPP_evaluate': 'Evaluator',

    # Long-horizon Temporal Point Process
    'LH_train': 'Trainer',
    'LH_evaluate': 'Evaluator',

    # Missing Data Imputation with Temporal Point Process
    'MDI_train': 'Trainer',
    'MDI_evaluate': 'Evaluator',

    # Outlier Detection with Temporal Point Process
    'OD_train': 'Trainer',
    'OD_evaluate': 'Evaluator',

    # Explainable History Distillation.
    'ehd_train': 'Trainer',
    'ehd_evaluate': 'Evaluator',

    'Transformers': 'Trainer',
    'fakenews': 'Trainer'
}


def environment_var_settings():
    '''
    Set up custom environment variables.
    '''
    env_dict = {}
    
    if os.path.exists(os.path.join(root_path, 'config', 'matplotlibrc')):
        env_dict['MATPLOTLIBRC'] = os.path.join(root_path, 'config', 'matplotlibrc')
    
    # set up PYTORCH_CUDA_ALLOC_CONF to mitigate GPU memory fragmentation.
    env_dict['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    env_dict['TOKENIZERS_PARALLELISM'] = 'true'
    
    os.environ.update(env_dict)


if __name__ == '__main__':
    '''
    We should configure all environment variables here before ```from src import TaskHost``` imports everything.
    '''
    environment_var_settings()
    
    '''
    Process starts here.
    Do NOT move these import codes to the beginning of this file.
    '''
    import argparse, importlib
    from src import TaskHost
    
    '''
    Enumerate subparsers from procedure_names
    we need main_procedure_translator and sub_procedure_translator to translate procedure names into correct argument classes.
    '''
    parser = argparse.ArgumentParser()
    procedure_names = [
        # Temporal point process
        'TPP_train',
        'TPP_evaluate',

        # Noted Temporal point process
        'NTPP_train',
        'NTPP_evaluate',

        # Explainable History Distillation.
        'ehd_train',
        'ehd_evaluate',

        # Long-horizon Temporal point process
        'LH_train',
        'LH_evaluate',

        # Missing Data Imputation with Temporal Point Process
        'MDI_train',
        'MDI_evaluate',
        
        # Outlier Detection with Temporal Point Process
        'OD_train',
        'OD_evaluate'
    ]

    subparsers = parser.add_subparsers(help = 'Define the procedure name.')
    for procedure_name in procedure_names:
        '''
        Fetch the argument class and attach them to the main parser.
        '''
        main_procedure = main_procedure_translator[procedure_name]
        sub_procedure_argument_prefix = sub_procedure_translator[procedure_name]

        tmp_parser_hook = subparsers.add_parser(procedure_name, help = f'We use {procedure_name}.')
        procedure = importlib.import_module('src.' + main_procedure)
        argument_class_name = sub_procedure_argument_prefix + 'Arguments'
        getattr(procedure, argument_class_name)(tmp_parser_hook, root_path)

    '''
    Call TaskHost to start the task.
    '''
    agent = TaskHost(parser = parser, root_path = root_path)
    agent.start()