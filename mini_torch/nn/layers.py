import numpy as np
import torch

from .init import _kaiming_init
from mini_torch.core.tensor import Tensor
from mini_torch.core.ops import cat, zeros_like, max, stack
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
    
class Conv1d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, a=5):
        super().__init__()
        self.a = a

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.weight = Tensor(_kaiming_init((self.out_channels, self.in_channels, self.kernel_size), self.out_channels, self.a, 11))
        self.bias = Tensor(_kaiming_init((self.out_channels), self.out_channels, self.a, 11), requires_grad=True) if bias else None

    def __call__(self, x: Tensor) -> Tensor:
        if len(x.shape()) == 3:
            B, C, W = x.shape()
        elif len(x.shape()) == 2:
            container = []
            container.append(x)
            x = stack(container)

            B, C, W = x.shape()
        else:
            raise RuntimeError(
                f"shape {x.shape()} not valid for Conv1d"
            )

        C_out = self.out_channels
        C_in = self.in_channels
        K = self.kernel_size

        S = self.stride
        P = self.padding
        D = self.dilation

        W_out = (W + 2*self.padding - self.dilation*(K-1) - 1) // self.stride + 1
        
        rows = []
        for b in range(B):
            for i in range(W_out):
                row = []

                start = i * S - P

                for c in range(C_in):
                    for u in range(K):
                        x_index = start + D * u
                        
                        if 0 <= x_index and x_index < W:
                            row.append(x[b, c, x_index])
                        else:
                            row.append(Tensor(0.0))

                rows.append(stack(row))

        X_col = stack(rows)

        W_flat = self.weight.reshape((C_out, C_in * K))
        out_flat = X_col @ W_flat.T()

        out = out_flat.reshape((B, W_out, C_out)).T((0, 2, 1))

        if self.bias is not None:
            out = out + self.bias

        return out