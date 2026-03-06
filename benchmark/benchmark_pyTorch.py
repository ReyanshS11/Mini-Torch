import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

# -----------------------------
# Load data
# -----------------------------
data = pd.read_csv("Housing.csv")
data = data.replace({
    "yes": 1,
    "no": 0,
    "furnished": 2,
    "semi-furnished": 1,
    "unfurnished": 0
})

data = (data - data.mean()) / (data.std() + 1e-8)

data_np = data.to_numpy(dtype=np.float32)

X = data_np[:, 1:]
y = data_np[:, 0:1]


# -----------------------------
# Dataset
# -----------------------------
class TestData(Dataset):

    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


dataset = TestData(X, y)
loader = DataLoader(dataset, batch_size=4, shuffle=True)


# -----------------------------
# Benchmark settings
# -----------------------------
epoch_list = list(range(10, 251, 10))
trials = 5

times = []
losses = []


# -----------------------------
# Benchmark loop
# -----------------------------
for epochs in epoch_list:

    trial_times = []
    trial_losses = []

    for t in range(trials):

        model = nn.Linear(12, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        start = time.perf_counter()

        for epoch in range(epochs):

            total_loss = []

            for x, y in loader:

                optimizer.zero_grad()

                out = model(x)
                loss = loss_fn(out, y)

                loss.backward()
                optimizer.step()

                total_loss.append(loss.item())

        end = time.perf_counter()

        trial_times.append(end - start)
        trial_losses.append(np.mean(total_loss))

    times.append(np.mean(trial_times))
    losses.append(np.mean(trial_losses))

    print(f"Epochs {epochs}")


# -----------------------------
# Plot time
# -----------------------------
plt.figure()
plt.plot(epoch_list, times, marker="o")
plt.xlabel("Epochs")
plt.ylabel("Training Time (seconds)")
plt.title("PyTorch: Epochs vs Training Time")
plt.grid(True)
plt.show()


# -----------------------------
# Plot loss
# -----------------------------
plt.figure()
plt.plot(epoch_list, losses, marker="o")
plt.xlabel("Epochs")
plt.ylabel("Final Loss")
plt.title("PyTorch: Epochs vs Final Loss")
plt.grid(True)
plt.show()