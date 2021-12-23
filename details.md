# Some information about this architecture

## train.py

```train.py``` should kick off the entire training procedure by creating an ```TrainingHost``` object and calling the ```start``` method. That's what this script should do.

## src.traininghost.py

This file implements the class ```TrainingHost``` based on the ```torch.multiprocessing``` backbone. Most time it should work despite any specific models.  
Instead of finding a specific model directly, ```TrainingHost``` only creates a ```trainer``` object using the following argument ```procedure_name``` and requires it to manage the rest processes. The ```trainer``` comprises two parts:  

1. ```procedure_name``` + ```Trainer```: The primary manager conducts all the training, evaluation, and saving tasks. Each group of models has different features, so one should implement specific ```Trainers``` for a model group to maximize the model efficiency.
2. ```procedure_name``` + ```Argument```: Every model needs hyperparameters to start training. Some common arguments are defined by ```TrainingHost```, while the ```Trainer``` still needs to complete the hyperparameter list by inheriting the basic argument class and adding more model-specific arguments.

Everyone can add more useful functions into ```traininghost_utils.py```.