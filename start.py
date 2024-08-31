import os

# Hope we can get rid of absolute path in training scripts.
root_path = os.path.dirname(os.path.abspath(__file__))

main_procedure_translator = {
    # Temporal point process
    'TPP_train': 'TPP',
    'TPP_evaluate': 'TPP',

    # Long-horizon Temporal point process
    # 'TPP_patch_train': 'TPP_patch',
    # 'TPP_patch_evaluate': 'TPP_patch',

    # Explainable History Distillation.
    'ehd_train': 'ehd',
    'ehd_evaluate': 'ehd',

    'Transformers': 'Transformers',
    'fakenews': 'fakenews'
}

sub_procedure_translator = {
    # Temporal point process
    'TPP_train': 'Trainer',
    'TPP_evaluate': 'Evaluator',

    # Long-horizon Temporal point process
    # 'TPP_patch_train': 'Trainer',
    # 'TPP_patch_evaluate': 'Evaluator',

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
    if os.path.exists(os.path.join(root_path, 'config', 'matplotlibrc')):
        os.environ['MATPLOTLIBRC'] = os.path.join(root_path, 'config')


if __name__ == '__main__':
    # We should configure all environment variables here before importing everything via ```from src import TaskHost```.
    environment_var_settings()
    
    # Start the process.
    import argparse, importlib
    from src import TaskHost
    
    # Enumerate subparsers from procedure_names
    # we need main_procedure_translator and sub_procedure_translator to translate
    # procedure names into the correct procedure and argument class.
    # The argument class of each process will be attached to the main parser and can be selected by procedure_name.
    parser = argparse.ArgumentParser()
    procedure_names = [
        # Temporal point process
        'TPP_train',
        'TPP_evaluate',

        # Explainable History Distillation.
        # 'ehd_train',
        # 'ehd_evaluate',

        # Long-horizon Temporal point process
        # 'TPP_patch_train',
        # 'TPP_patch_evaluate',
    ]

    subparsers = parser.add_subparsers(help = 'Define the procedure name.')
    for procedure_name in procedure_names:
        '''
        Get the argument list
        '''
        main_procedure = main_procedure_translator[procedure_name]
        sub_procedure_argument_prefix = sub_procedure_translator[procedure_name]

        tmp_parser_hook = subparsers.add_parser(procedure_name, help = f'We use {procedure_name}.')
        procedure = importlib.import_module('src.' + main_procedure)
        argument_class_name = sub_procedure_argument_prefix + 'Arguments'
        getattr(procedure, argument_class_name)(tmp_parser_hook, root_path)

    agent = TaskHost(parser = parser, root_path = root_path)
    agent.start()