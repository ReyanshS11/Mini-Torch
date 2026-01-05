import numpy as np
import torch

from mini_torch.core.tensor import Tensor
from .module import Module

class MSELoss(Module):
    def __call__(self, y_hat: Tensor, y: Tensor) -> Tensor:
        y_hats = y_hat.data.tolist()
        ys = y.data.tolist()

        differences = []
        for y, y_hat in zip(ys, y_hats):
            differences.append(y - y_hat)
        
        squared_differences = [i**2 for i in differences]
        mse = np.mean(squared_differences)

        return Tensor(mse, requires_grad=True)