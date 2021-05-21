from .submodel import FullyNN
import torch
import torch.nn as nn


def check_tensor(x):
    assert (x < 0).cpu().numpy().any() == False


class FullyNNModel(nn.Module):
    def __init__(self, d_history,
                 d_intensity,
                 dropout,
                 rnn_layers,
                 mlp_layers):
        super(FullyNNModel, self).__init__()
        self.model = FullyNN(d_history, d_intensity,
                                dropout, rnn_layers, mlp_layers)

    def forward(self, input_time, input_result):
        input_result.requires_grad = True

        integral = self.model(input_time, input_result)

        intensity = torch.autograd.grad(
            outputs=integral,
            inputs=input_result,
            grad_outputs=torch.ones_like(integral),
            create_graph=True,
        )[0]
        check_tensor(intensity)

        input_result.requires_grad = False

        assert intensity.shape == input_result.shape

        return integral, intensity
