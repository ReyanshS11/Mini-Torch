from mini_torch.core.tensor import Tensor
from mini_torch.core.ops import max
from mini_torch.nn.layers import Linear
from mini_torch.nn.loss import MSELoss
from mini_torch.optim.sgd import SGD
from mini_torch.data.dataset import Dataset
from mini_torch.data.dataloader import DataLoader

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

x = Tensor([[1, 2, 3, 4],
     [1, 2, 3, 4]], requires_grad=True)

y = Tensor([[2, 1, 4, 5],
     [2, 1, 4, 3]], requires_grad=True)

z = max(x, y)

loss = z.sum()
loss.backward()

print(z.data, x.grad, y.grad)