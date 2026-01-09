import numpy as np

from mini_torch.core.tensor import Tensor
from .dataset import Dataset

class DataLoader:
    def __init__(self, dataset: Dataset, batch_size: int, shuffle=False) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.length = dataset.__len__()
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        
        if self.shuffle:
            np.random.shuffle(indices)

        for i in range(0, self.length, self.batch_size):
            batch_idx = indices[i:i+self.batch_size]
            batch = [self.dataset.__getitem__(idx) for idx in batch_idx]

            Xs, ys = zip(*batch)
            X = Tensor(np.stack(Xs), requires_grad=True)
            y = Tensor(np.stack(Xs), requires_grad=True)

            yield X, y