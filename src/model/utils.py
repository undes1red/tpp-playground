from abc import ABCMeta, abstractmethod
import torch.nn as nn


class BasicModule(nn.Module, metaclass = ABCMeta):
    '''
    The parent of all model classes.
    '''
    @abstractmethod
    def forward(self, *args):
        raise NotImplementedError()

    @staticmethod
    def train_step(model, minibatch, device):
        raise NotImplementedError()

    @staticmethod
    def evaluation_step(model, minibatch, device):
        raise NotImplementedError()

    @staticmethod
    def postprocess(input):
        pass