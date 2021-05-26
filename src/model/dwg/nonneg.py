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


class SigmoidLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(SigmoidLinear, self).__init__(in_features, out_features, bias)
        self.positivify_weights()

    def positivify_weights(self):
        mask = (self.weight < 0).float() * - 1
        mask = mask + (self.weight >= 0).float()
        self.weight.data = self.weight.data * mask

    def forward(self, inputs):
        weight = F.sigmoid(self.weight)
        return F.linear(inputs, weight, self.bias)


class SoftPlusLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True,
                 beta=1., threshold=20):
        super(SoftPlusLinear, self).__init__(in_features, out_features, bias)
        self.beta = beta
        self.threshold = threshold
        self.positivify_weights()

    def positivify_weights(self):
        mask = (self.weight < 0).float() * - 1
        mask = mask + (self.weight >= 0).float()
        self.weight.data = self.weight.data * mask

    def forward(self, inputs):
        weight = F.softplus(
            self.weight, beta=self.beta, threshold=self.threshold)
        return F.linear(inputs, weight, self.bias)

class ClampLinear(nn.Linear):
    '''
    Alleviate the negative gradient issue.
    Normal datasets do not require this trick except the intensity is too steep(always come with negative loss).
    '''
    def __init__(self, in_features, out_features, clamp_min = -1, bias=True):
        super(ClampLinear, self).__init__(in_features, out_features, bias)
        self.clamp_min = clamp_min

        if self.clamp_min:
            self.clamp_weights()

    def clamp_weights(self):
        self.weight.data = torch.clamp(self.weight.data, min=self.clamp_min)

    def forward(self, inputs):
        if self.clamp_min:
            self.weight.data = torch.clamp(self.weight.data, min=self.clamp_min)
        return F.linear(inputs, self.weight, self.bias)