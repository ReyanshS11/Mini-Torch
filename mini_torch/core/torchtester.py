import torch
import torch.nn as nn
import numpy as np

x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
linear = nn.Linear(4, 8)

print(linear(x))