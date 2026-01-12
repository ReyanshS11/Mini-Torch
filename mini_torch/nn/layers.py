import numpy as np
import torch

from .init import _kaiming_init
from mini_torch.core.tensor import Tensor
from mini_torch.core.ops import zeros_like, max
from .module import Module

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, a = 5):
        super().__init__()
        self.a = a

        self.weight = Tensor(_kaiming_init((out_features, in_features), out_features, self.a, 11), requires_grad=True)

        if bias:
            self.bias = Tensor(_kaiming_init((out_features), out_features, self.a, 11), requires_grad=True)
        else:
            self.bias = None

    def __call__(self, x: Tensor) -> Tensor:
        x = x @ self.weight.T()
        if self.bias is not None:
            x = x + self.bias
        return x
    
class ReLU(Module):
    def __init__(self):
        super().__init__()
    
    def __call__(self, x: Tensor) -> Tensor:
        return max(zeros_like(x), x)