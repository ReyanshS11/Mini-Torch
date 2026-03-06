from mini_torch.nn.layers import Linear
from mini_torch.nn.loss import MSELoss
from mini_torch.optim.sgd import SGD
from mini_torch.data.dataset import Dataset
from mini_torch.data.dataloader import DataLoader

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

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


dataset = TestData(X, y)
loader = DataLoader(dataset, 4)


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

        model = Linear(12, 1)
        optimizer = SGD(model.parameters(), lr=1e-3)
        loss_fn = MSELoss()

        start = time.perf_counter()

        for epoch in range(epochs):

            total_loss = []

            for batch in loader:

                x = batch[0]
                y = batch[1]

                optimizer.zero_grad()

                out = model(x)
                loss = loss_fn(out, y)

                loss.backward()
                optimizer.step()

                total_loss.append(loss.data)

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
plt.title("MiniTorch: Epochs vs Training Time")
plt.grid(True)
plt.show()


# -----------------------------
# Plot loss
# -----------------------------
plt.figure()
plt.plot(epoch_list, losses, marker="o")
plt.xlabel("Epochs")
plt.ylabel("Final Loss")
plt.title("MiniTorch: Epochs vs Final Loss")
plt.grid(True)
plt.show()