from mini_torch.core.tensor import Tensor
from mini_torch.nn.layers import Linear

import torch
import torch.nn as nn
import numpy as np

if __name__ == "__main__":
    linear = Linear(1, 4)

    print(linear.parameters())