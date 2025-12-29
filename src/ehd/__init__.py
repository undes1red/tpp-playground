"""
TaskHost needs this file to start required tasks. Please, do not modify the content of this file.
"""

from src.ehd.dataloader import get_dataloader
from src.ehd.evaluate_arguments import Evaluator_postprocess, EvaluatorArguments
from src.ehd.evaluate_functions import desc_funcs as get_evaluation_funcs
from src.ehd.model import get_model
from src.ehd.train_arguments import Trainer_postprocess, TrainerArguments
from src.ehd.utils import easy_model_load

__all__ = [
    "get_dataloader",
    "Evaluator_postprocess",
    "EvaluatorArguments",
    "get_evaluation_funcs",
    "get_model",
    "Trainer_postprocess",
    "TrainerArguments",
    "easy_model_load",
]


pytorch_version_warnings = {
    '==1.4.0': [
'''
It is known that several learning rate schedulers including LambdaLR, which we use in optim.py, fail to run. Please update PyTorch to 1.5.0 or above.
Detailed information is available at https://github.com/pytorch/pytorch/issues/36313
'''.replace("\n", ""), 'stop'],
    '==2.0.1': [
'''
The gumbel_sample() of PyTorch 2.0.0 might produce NaN when running on CPU because the sample of the exponential distribution provided by MKL can be zero, which is unexpected to PyTorch.
Please check https://github.com/pytorch/pytorch/issues/101620 for more information.
This has been fixed in PyTorch 2.1.0.
'''.replace("\n", ""), 'continue'],
}
