import argparse

class TransformersTrainerArguments:
    def __init__(self, parser, root_path):
        self.parser = parser
        # identification mark
        self.parser.add_argument('--procedure', type = str, default = 'Transformers',
                            help='Used as an identifier. DO NOT USE IT.')

    def get_args(self):
        pass

def postprocess(opt, root_path):
    pass