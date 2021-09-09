import torch
from torch.autograd import Function
from scipy.special import expi

class EI(Function):
    '''
    Special function: Exponential Integral function: expi(x) = \integral_{-x}^{\inft}{\frac{e^{-t}}{t}}
    '''
    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        data = input.cpu().detach().numpy()
        result = expi(data)
        return input.new(result)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.new(grad_output * torch.exp(ctx.input)/ctx.input)

if __name__ == '__main__':
    from torch.autograd import gradcheck

    # gradcheck takes a tuple of tensors as input, check if your gradient
    # evaluated with these tensors are close enough to numerical
    # approximations and returns True if they all verify this condition.
    input = (torch.randn(3,dtype=torch.double,requires_grad=True))
    test = gradcheck(EI.apply, input, eps=1e-6, atol=1e-4)
    print(test)