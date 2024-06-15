from src import TaskHost
import os, argparse, importlib

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

if __name__ == '__main__':
    # Train
    parser = argparse.ArgumentParser()

    # Enumerate subparsers from procedure_names
    # we need main_procedure_translator and sub_procedure_translator to translate
    # procedure names into the correct procedure and argument class.
    procedure_names = [
        # Temporal point process
        'TPP_train',
        'TPP_evaluate',

        # Explainable History Distillation.
        'ehd_train',
        'ehd_evaluate',

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
        sub_procedure_argument_prefix = main_procedure + sub_procedure_translator[procedure_name]

        tmp_parser_hook = subparsers.add_parser(procedure_name, help = f'We use {procedure_name}.')
        procedure = importlib.import_module('src.' + main_procedure)
        argument_class_name = sub_procedure_argument_prefix + 'Arguments'
        getattr(procedure, argument_class_name)(tmp_parser_hook, root_path)

    agent = TaskHost(parser = parser, root_path = root_path)
    agent.start()