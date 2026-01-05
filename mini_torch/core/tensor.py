import numpy as np

import autograd
import ops

class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.asarray(data)
        self.shape = self.data.shape
        self.size = self.data.size

        self.requires_grad = requires_grad
        self.grad = None

        self._prev = set()
        self._backward = lambda: None

    def numerical_grad(self, f, x, eps=1e-6):
        return autograd.numerical_grad(f, x, eps)
    
    def unbroadcast(self, grad, shape):
        return autograd.unbroadcast(grad, shape)

    def __add__(self, other) -> Tensor:
        return ops.__add__(self, other)
    
    def __sub__(self, other) -> Tensor:
        return ops.__sub__(self, other)
    
    def __mul__(self, other) -> Tensor:
        return ops.__mul__(self, other)
    
    def __truediv__(self, other) -> Tensor:
        return ops.__truediv__(self, other)
    
    def __neg__(self) -> Tensor:
        return ops.__neg__(self)
    
    def __pow__(self, power) -> Tensor:
        return ops.__pow__(self, power)
    
    def sum(self) -> Tensor:
        return ops.sum(self)
    
    def mean(self) -> Tensor:
        return ops.mean(self)
    
    def reshape(self, shape) -> Tensor:
        return ops.reshape(self, shape)

    def backward(self) -> None:
        autograd.backward(self)

if __name__ == "__main__":
    x = Tensor(
        [
            [[1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]],

            [[7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0]]
        ],
        requires_grad=True
    )

    y = x.reshape([3, 2, 2])

    loss = y.sum()
    loss.backward()

    print(y.data, "\n\n", y.grad)