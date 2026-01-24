"""
TaskHost needs this file to start required tasks. Please, do not modify the content of this file.
"""

from .dataloader import get_dataloader
from .evaluate_arguments import EvaluatorArguments
from .evaluate_functions import desc_funcs as get_evaluation_funcs
from .model import get_model
from .train_arguments import TrainerArguments
from .utils import easy_model_load

__all__ = [
    "get_dataloader",
    "EvaluatorArguments",
    "get_evaluation_funcs",
    "get_model",
    "TrainerArguments",
    "easy_model_load",
]

pytorch_version_warnings = {
    ">=2.0.0": (
        "torch.compile() doesn't support double backwards, so you shouldn't compile models in the fullynn and ifn family. Please track this issue at https://github.com/pytorch/pytorch/issues/91469.",
        "continue",
    ),
    "==1.4.0": (
        "It is known that several learning rate schedulers shipped by PyTorch 1.4.0 are buggy and fail to run. Please update PyTorch to 1.5.0 or above. Detailed information is available at https://github.com/pytorch/pytorch/issues/36313",
        "stop",
    ),
}
