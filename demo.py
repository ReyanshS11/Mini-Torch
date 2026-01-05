from mini_torch.core.tensor import Tensor
from mini_torch.nn.layers import Linear

import torch
import torch.nn as nn
import numpy as np

if __name__ == "__main__":
    x = Tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    custom_linear = Linear(4, 8)

    print(custom_linear(x).data)

    x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    linear = nn.Linear(4, 8)
    
    weight = torch.nn.Parameter(torch.tensor(custom_linear.weight.data.tolist()), requires_grad=True)
    bias = torch.nn.Parameter(torch.tensor(custom_linear.bias.data.tolist()), requires_grad=True)
    
    linear.weight = weight
    linear.bias = bias

    print(linear(x))