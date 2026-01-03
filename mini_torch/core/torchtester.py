import torch
from tensor import Tensor

x = torch.tensor([3.0, 2.0], requires_grad=True)
y = torch.tensor([4.0, 1.0], requires_grad=True)

def numerical_grad(f, x, eps=1e-6):
    return (f(x + eps) - f(x - eps)) / (2 * eps)

x = Tensor(3.0, requires_grad=True)
y = x * x * x
y.backward()

num = numerical_grad(lambda v: (Tensor(v) * Tensor(v) * Tensor(v)).data, 3.0)

print("autograd:", x.grad)
print("numerical:", num)