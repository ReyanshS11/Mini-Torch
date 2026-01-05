from __future__ import annotations
import numpy as np
from .tensor import Tensor

def numerical_grad(f, x, eps=1e-6):
    return (f(x + eps) - f(x - eps)) / (2 * eps)

def unbroadcast(grad, shape):
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
            
    for i, s in enumerate(shape):
        if s == 1:
            grad  = grad.sum(axis=i, keepdims=True)

    return grad

def backward(tns: Tensor) -> None:
    if tns.data.ndim != 0:
        raise RuntimeError(
            "grad can be implicitly created only for scalar outputs"
        )
    
    topo = []
    visited = set()

    def build(v: Tensor):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build(child)
            topo.append(v)

    build(tns)

    if tns.grad is None:
        tns.grad = np.ones_like(tns.data) # type: ignore

    for t in reversed(topo):
        t._backward()