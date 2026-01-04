import torch
import numpy as np

x = torch.tensor([[1.0],[2.0],[3.0]], requires_grad=True)
y = torch.tensor([10.0,20.0], requires_grad=True)
z = x * y
loss = z.sum()
loss.backward()

print(x.grad, y.grad)