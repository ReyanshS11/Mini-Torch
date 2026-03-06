import matplotlib.pyplot as plt

# Epochs used in your benchmark
epochs = list(range(10, 251, 10))

# Paste the times printed from each program here
mini_times = [
0.4,0.7,1.0,1.5,2.0,2.3,2.4,2.8,5.0,6.3,
4.8,4.2,4.5,4.9,9.8,10.9,8.2,6.6,6.5,6.7,
7.1,7.7,8.2,8.5,8.6
]

torch_times = [
1.0,1.9,5.1,3.8,4.1,4.6,4.0,3.7,4.2,4.6,
5.1,5.7,6.3,6.5,7.0,8.7,11.8,12.6,13.8,13.1,
9.7,10.2,10.7,11.1,21.0
]

plt.figure()

plt.plot(epochs, mini_times, marker="o", label="MiniTorch")
plt.plot(epochs, torch_times, marker="o", label="PyTorch")

plt.xlabel("Epochs")
plt.ylabel("Training Time (seconds)")
plt.title("Training Time Comparison: MiniTorch vs PyTorch")
plt.legend()
plt.grid(True)

plt.show()