import torch
import torch.nn.functional as F
from torch import nn

# From Babylon's neuralTPPs: https://github.com/babylonhealth/neuralTPPs

class NonNegLinear(nn.Linear):
    def __init__(self, in_features, out_features, device, bias=True, eps=0., embedding_like = False):
        super(NonNegLinear, self).__init__(1 if embedding_like else in_features, out_features, bias, device = device)
        self.eps = eps
        self.device = device
        self.positivify_weights()
        self.embedding_like = embedding_like

    def positivify_weights(self):
        mask = (self.weight < 0).float() * - 1
        mask = mask + (self.weight >= 0).float()
        self.weight.data = self.weight.data * mask

    def forward(self, inputs):
        weight = self.weight > 0
        weight = self.weight * weight.float()
        self.weight.data = torch.clamp(weight, min=self.eps)
        if self.embedding_like:
            return F.linear(inputs.unsqueeze(dim = -1), self.weight, self.bias)# [..., original_tensor_last_dimention, out_features]
        else:
            return F.linear(inputs, self.weight, self.bias)                    # [..., out_features]