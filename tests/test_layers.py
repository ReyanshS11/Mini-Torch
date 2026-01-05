import unittest
import numpy as np

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mini_torch.core.tensor import Tensor
from mini_torch.nn.layers import Linear

import torch
import torch.nn as nn

class TestLayers(unittest.TestCase):
    def test_linear(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        custom_linear = Linear(4, 8)
        custom_linear_out = custom_linear(x).data

        torch_x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        linear = nn.Linear(4, 8)
        
        weight = torch.nn.Parameter(torch.tensor(custom_linear.weight.data.tolist()), requires_grad=True)
        bias = torch.nn.Parameter(torch.tensor(custom_linear.bias.data.tolist()), requires_grad=True)
        
        linear.weight = weight
        linear.bias = bias

        torch_linear_out = linear(torch_x).detach().numpy()

        np.testing.assert_array_almost_equal(custom_linear_out, torch_linear_out)

if __name__ == "__main__":
    unittest.main()