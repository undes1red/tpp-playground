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