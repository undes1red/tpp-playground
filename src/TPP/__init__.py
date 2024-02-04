'''
TPPTrainerArgument and TPPPlotterArguments are argument parser for model training and model evaluation
You need Trainer_postprocess() and Plotter_postprocess() for postprocessing the arguments.

TPPTrainer and TPPPlotter handles the main process of training and evaluation.

TaskHost needs this file to start required tasks. Please, do not modify the content of this file.
'''

from src.TPP.trainer_arguments import TPPTrainerArguments, Trainer_postprocess
from src.TPP.plotter_arguments import TPPPlotterArguments, Plotter_postprocess

from src.TPP.trainer import TPPTrainer
from src.TPP.plotter import TPPPlotter


pytorch_version_warnings = {
    '2.1.2': [
'''
We have noticed the evaluation procedure of IFIB is over 1x slower on PyTorch 2.1.2 than PyTorch 2.0.1, from around 3 mins to nearly 7 mins on synthetic datasets.
Other PyTorch releases belonging to the 2.1.y family are not tested, so we can not tell if this problem persists in PyTorch 2.1.0 and 2.1.1.
We are not clear about the real cause of this regression. 
However, we suspect it might be the aten:fill_kernel introduced in PyTorch 2.1.0 as other people have reported this function causes a performance regression at https://github.com/pytorch/pytorch/issues/117081 targeting all PyTorch 2.1.y releases.
If aten:fill_kernel is the one to blame, then this issue should also persist on PyTorch 2.1.0 and 2.1.1 + cu118.
Because of this, we suggest to use PyTorch 2.0.1 + cu117. You can still train these models on the current setting, but you might not be able to reproduce the evaluation speed we have reported in the paper.
'''.replace("\n", ""), 'continue'],

    '1.4.0': [
'''
It is known that several learning rate schedulers including LambdaLR, which we use in optim.py, fail to run. Please update PyTorch to 1.5.0 or above.
Detailed information is available at https://github.com/pytorch/pytorch/issues/36313
'''.replace("\n", ""), 'stop'],
}