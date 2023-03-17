from src import TrainingHost
import os, argparse, importlib

# Hope we can get rid of absolute path in training scripts.
root_path = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    # Train
    parser = argparse.ArgumentParser()

    # Enumerate subparsers
    procedure_names = ['TPP', 'Transformers', 'fakenews']

    subparsers = parser.add_subparsers(help = 'Define the procedure name.')
    for procedure_name in procedure_names:
        '''
        Get the argument list
        '''

        tmp_parser_hook = subparsers.add_parser(procedure_name, help = f'We use {procedure_name}.')
        procedure = importlib.import_module('src.' + procedure_name)
        argument_class_name = procedure_name + 'Arguments'
        getattr(procedure, argument_class_name)(tmp_parser_hook, root_path)

    agent = TrainingHost(parser = parser, root_path = root_path)
    agent.start()