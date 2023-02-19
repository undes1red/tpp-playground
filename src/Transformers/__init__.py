'''
We should have two classes:
1. TransformerArguments: it contains all required arguments to kick off a training procedure.
2. TransformerTrainer: this module should do all the work, such as training, evaluating, and models saving.
'''

from .arguments import TransformersArguments, postprocess