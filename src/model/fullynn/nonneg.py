import torch
import torch.nn.functional as F
from torch import nn

# From Babylon's neuralTPPs: https://github.com/babylonhealth/neuralTPPs


class NonNegLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, eps=0.):
        super(NonNegLinear, self).__init__(in_features, out_features, bias)
        self.eps = eps
        self.positivify_weights()

    def positivify_weights(self):
        mask = (self.weight < 0).float() * - 1
        mask = mask + (self.weight >= 0).float()
        self.weight.data = self.weight.data * mask

    def forward(self, inputs):
        weight = self.weight > 0
        weight = self.weight * weight.float()
        self.weight.data = torch.clamp(weight, min=self.eps)
        return F.linear(inputs, self.weight, self.bias)
