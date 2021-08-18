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
    def __init__(self, in_features, out_features, clamp_min = None, bias=True):
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


class Polynomial(nn.Module):
    def __init__(self, dimension, device, polynomial_start, polynomial_end):
        '''
        Size should be odd.
        '''
        super(Polynomial, self).__init__()
        self.demension = dimension
        self.polynomial_start = polynomial_start
        self.polynomial_end = polynomial_end
        self.size = polynomial_end - polynomial_start + 1
        self.device = device

        # The weight
        self.polynomial_weight = nn.Parameter(
                                    nn.init.normal_(torch.zeros((dimension, self.size, 1), dtype = torch.float32, device = device)), \
                                    requires_grad = True
                                )
        
        shift = torch.arange(polynomial_start, polynomial_end + 1, device = device).reshape(-1, 1)
        # Zeros in weight_correction matrix should be replaced by 1 and corresponding values in expand_matrix_t are replaced with nature log.
        # self.weight_correction: [self.size, self.size]
        self.weight_correction = torch.arange(polynomial_start, polynomial_end + 1, device = device).repeat(self.size, 1) + shift + 1
        self.weight_mask = self.weight_correction == 0
        self.weight_correction[self.weight_mask] = 1

    def forward(self, x):
        # x is a number.
        # x: [batch_size, 1]
        batch_size, _ = x.shape

        # [batch_size, self.size, self.size]
        time_expand = x.repeat(1, self.size * self.size).reshape(-1, self.size, self.size)
        expand_poly_t = torch.pow(time_expand, self.weight_correction)
        expand_matrix_t = expand_poly_t.masked_fill(self.weight_mask, 0)
        # [dimension, self.size, self.size]
        generated_matrix = torch.bmm(self.polynomial_weight, self.polynomial_weight.transpose(1, 2))
        generated_matrix *= 1 / (self.weight_correction.unsqueeze(0))
        # [batch_size, dimension, self.size, self.size]
        polynomial_result = expand_matrix_t.unsqueeze(1) * generated_matrix.unsqueeze(0)
        # [batch_size, dimension, 1]
        sum_result = polynomial_result.reshape(batch_size, self.demension, -1).sum(dim = -1, keepdim = True)

        return sum_result