import numpy as np
from mini_torch.core.tensor import Tensor
import torch

class Linear:
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = Tensor(np.random.randn(out_features, in_features), requires_grad=True)

        if bias:
            self.bias = Tensor(np.random.randn(out_features), requires_grad=True)
        else:
            self.bias = None

    def __call__(self, x: Tensor) -> Tensor:
        x = x@self.weight.T()
        if self.bias is not None:
            x = x + self.bias
        return x