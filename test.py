import torch
from torch.utils.flop_counter import FlopCounterMode

device = 'cpu'

a = torch.randn(3, 4, 5, device = device)
model = torch.nn.Linear(5, 7, device = device)

a.requires_grad = True
with FlopCounterMode(display = False) as counter:
    b = model(a)
gradient = torch.autograd.grad(
    outputs = b,
    inputs = a,
    grad_outputs = torch.ones_like(b),
    create_graph = True
)
a.requires_grad = False

print(gradient)