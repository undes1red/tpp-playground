from src import TaskHost
import os, argparse, importlib

# Hope we can get rid of absolute path in training scripts.
root_path = os.path.dirname(os.path.abspath(__file__))

main_procedure_translator = {
    # Temporal point process
    'TPP_train': 'TPP',
    'TPP_plot': 'TPP',

    # Outlier-directed missing data imputation
    'missing_train': 'missing',
    'missing_evaluate': 'missing',

    'Transformers': 'Transformers',
    'fakenews': 'fakenews'
}

sub_procedure_translator = {
    # Temporal point process
    'TPP_train': 'Trainer',
    'TPP_plot': 'Plotter',

    # Outlier-directed missing data imputation
    'missing_train': 'Trainer',
    'missing_evaluate': 'Plotter',

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
        'TPP_plot',

        # Outlier-directed missing data imputation
        'missing_train',
        'missing_evaluate'

        # 'Transformers',
        # 'fakenews'
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