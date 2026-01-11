import numpy as np

from mini_torch.core.tensor import Tensor
from mini_torch.nn.module import Module

class SGD(Module):
    def __init__(self, parameters, lr, grad_clipping=False):
        super().__init__()
        self.parameters = parameters
        self.lr = lr
        self.grad_clipping = grad_clipping

    def zero_grad(self):
        for p in self.parameters:
            p.grad = np.zeros_like(p.data)

    def step(self):
        for p in self.parameters:
            if p.grad is None:
                continue

            if self.grad_clipping:
                np.clip(p.grad, -1.0, 1.0, out=p.grad)
                
            p.data -= self.lr * p.grad