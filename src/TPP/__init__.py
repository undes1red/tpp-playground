'''
TPPTrainerArgument and TPPPlotterArguments are argument parser for model training and model evaluation
You need Trainer_postprocess() and Plotter_postprocess() for postprocessing the arguments.

TPPTrainer and TPPPlotter handles the main process of training and evaluation.

TaskHost needs this file to start required tasks. Please, do not modify the content of this file.
'''

# Argument parser
from src.TPP.train_arguments import TrainerArguments, Trainer_postprocess
from src.TPP.evaluate_arguments import EvaluatorArguments, Evaluator_postprocess

# Model and dataloader entry.
from src.TPP.model import get_model
from src.TPP.dataloader import get_dataloader

# Evaluation.
from src.TPP.evaluation_functions import desc_funcs as get_evaluation_funcs

# Function load_model() will use this function for stable and easy corss-process model loading.
from src.TPP.utils import easy_model_load

pytorch_version_warnings = {
    '==1.4.0': [
'''
It is known that several learning rate schedulers shipped by PyTorch 1.4.0 are buggy and fail to run. Please update PyTorch to 1.5.0 or above.
Detailed information is available at https://github.com/pytorch/pytorch/issues/36313
'''.replace("\n", ""), 'stop'],
}