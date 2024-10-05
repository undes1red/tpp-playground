'''
TPPTrainerArgument and TPPPlotterArguments are argument parser for model training and model evaluation
You need Trainer_postprocess() and Plotter_postprocess() for postprocessing the arguments.

TPPTrainer and TPPPlotter handles the main process of training and evaluation.

TaskHost needs this file to start required tasks. Please, do not modify the content of this file.
'''

from src.ehd.trainer_arguments import TrainerArguments, Trainer_postprocess
from src.ehd.evaluator_arguments import EvaluatorArguments, Evaluator_postprocess


# Model and dataloader entry.
from src.ehd.model import get_model
from src.ehd.dataloader import get_dataloader

# Evaluation.
# from src.TPP.tpp_evaluation_functions import desc_funcs as get_evaluation_funcs


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