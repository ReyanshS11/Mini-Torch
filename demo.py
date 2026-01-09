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

data = pd.read_csv("Housing.csv")
data = data.replace({"yes": 1, "no": 0, "furnished": 2, "semi-furnished": 1, "unfurnished": 0})
data = (data - data.mean()) / (data.std() + 1e-8)

train = data[:500]
test = data[500:]

full_ds = [[list(row[1:]), row[0]] for row in data.itertuples(index=False)]

class TestData(Dataset):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_tns = self.data[idx][0]
        label_tns = self.data[idx][1]

        return input_tns, label_tns

if __name__ == "__main__":
    dataset = TestData(full_ds)
    dataset = DataLoader(dataset, 4)

    pytorch_model = nn.Linear(12, 1)

    model = Linear(12, 1)
    model.weight = Tensor(pytorch_model.weight.tolist(), requires_grad=True)

    optim = SGD(model.parameters(), lr=1e-5)
    loss_fn = MSELoss()

    for epoch in range(10):
        total_loss = []

        for batch in dataset:
            x = batch[0]
            y = batch[1]

            optim.zero_grad()

            out = model(x)
            loss = loss_fn(out, y)

            loss.backward()
            optim.step()

            total_loss.append(loss.data)

        print(f"{epoch}: {np.mean(total_loss)}")