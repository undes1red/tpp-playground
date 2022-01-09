import torch
import torch.nn.functional as F
from torch import nn

# From Babylon's neuralTPPs: https://github.com/babylonhealth/neuralTPPs


class NonNegLinear(nn.Linear):
    def __init__(self, in_features, out_features, device, bias=True, eps=0.):
        super(NonNegLinear, self).__init__(in_features, out_features, bias, device = device)
        self.eps = eps
        self.device = device
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
    def __init__(self, in_features, out_features, device, bias=True):
        super(SigmoidLinear, self).__init__(in_features, out_features, bias, device = device)
        self.device = device
        self.positivify_weights()

    def positivify_weights(self):
        mask = (self.weight < 0).float() * - 1
        mask = mask + (self.weight >= 0).float()
        self.weight.data = self.weight.data * mask

    def forward(self, inputs):
        weight = F.sigmoid(self.weight)
        return F.linear(inputs, weight, self.bias)


class SoftPlusLinear(nn.Linear):
    def __init__(self, in_features, out_features, device, bias=True,
                 beta=1., threshold=20):
        super(SoftPlusLinear, self).__init__(in_features, out_features, bias, device = device)
        self.device = device
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
    def __init__(self, in_features, out_features, device, clamp_min = None, bias=True, clamp_max = None):
        super(ClampLinear, self).__init__(in_features, out_features, bias, device = device)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.device = device

        if self.clamp_min or self.clamp_max:
            self.clamp_weights()

    def clamp_weights(self):
        self.weight.data = torch.clamp(self.weight.data, min=self.clamp_min, max = self.clamp_max)

    def forward(self, inputs):
        if self.clamp_min or self.clamp_max:
            self.weight.data = torch.clamp(self.weight.data, min=self.clamp_min, max = self.clamp_max)
        return F.linear(inputs, self.weight, self.bias)

class NonNegNormLinear(nn.Linear):
    '''
    Alleviate the negative gradient issue.
    Normal datasets do not require this trick except the intensity is too steep(always come with negative loss).
    '''
    def __init__(self, in_features, out_features, norm_min, device, bias = None):
        super(NonNegNormLinear, self).__init__(in_features, out_features, bias, device = device)
        self.norm_min = norm_min
        self.device = device

        if self.norm_min:
            self.clamp_weights()

    def regularization(self, x, norm_min):
        return x if torch.sum(x) >= norm_min else x + (norm_min - torch.sum(x))/(x.numel())

    def clamp_weights(self):
        self.weight.data = self.regularization(self.weight.data, norm_min = self.norm_min)

    def forward(self, inputs):
        if self.norm_min:
            self.weight.data = self.regularization(self.weight.data, norm_min=self.norm_min)
        return F.linear(inputs, self.weight, self.bias)