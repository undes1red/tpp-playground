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
from src.TPP.tpp_evaluation_functions import desc_funcs as get_evaluation_funcs

# Function load_model() will use this function for stable and easy corss-process model loading.
from src.TPP.utils import easy_model_load

pytorch_version_warnings = {
    '>=2.1.0': [
'''
We have noticed the evaluation procedure of IFIB, FullyNN, and FENN is around 1-2x slower on PyTorch 2.x.y(x > 0) than PyTorch 2.0.1 and previous releases.
The benchmark results suggest that there might be a performance regression in nn.Linear() when its inputs are very tall tensors, i.e. tensors having four, five, or even higher dimensions.
This means possibly all MTPP models in this codebase are affected given processing high dimensional tensors are ubiquitous in our code.
One can track this regression at https://github.com/pytorch/pytorch/issues/124838.
Because of this, we suggest to use PyTorch 2.0.1 and previous. You can still train these models on the later releases because we have not observed any model performance degradation, but you may not be able to reproduce the evaluation speed we have reported in the paper.
'''.replace("\n", ""), 'continue'],

    '==1.4.0': [
'''
It is known that several learning rate schedulers shipped by PyTorch 1.4.0 are buggy and fail to run. Please update PyTorch to 1.5.0 or above.
Detailed information is available at https://github.com/pytorch/pytorch/issues/36313
'''.replace("\n", ""), 'stop'],
}