from mini_torch.core.tensor import Tensor
from mini_torch.nn.layers import Linear
from mini_torch.nn.loss import MSELoss
from mini_torch.optim.sgd import SGD
from mini_torch.data.dataset import Dataset
from mini_torch.data.dataloader import DataLoader

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

np.random.seed(0)

def numerical_grad(f, x, eps=1e-6):
    grad = np.zeros_like(x.data)
    it = np.nditer(x.data, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index
        old = x.data[idx]

        x.data[idx] = old + eps
        p1 = f().data

        x.data[idx] = old - eps
        p2 = f().data

        x.data[idx] = old

        grad[idx] = (p1 - p2) / (2 * eps)
        it.iternext()

    return grad

x = Tensor(np.random.randn(4, 3), requires_grad=False)
y = Tensor(np.random.randn(4, 1), requires_grad=False)

W = Tensor(np.random.randn(1, 3), requires_grad=True)
b = Tensor(np.random.randn(1), requires_grad=True)

def forward():
    out = x @ W.T() + b
    loss = ((out - y) * (out - y)).mean()
    return loss

loss = forward()
loss.backward()

print("Your grad W:", W.grad)
print("Your grad b:", b.grad)

print("Num grad W:", numerical_grad(forward, W))
print("Num grad b:", numerical_grad(forward, b))