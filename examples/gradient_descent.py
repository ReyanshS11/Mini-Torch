import numpy as np
import matplotlib.pyplot as plt
from scipy.differentiate import derivative
import sympy as smp

def y_function(x):
    return 0.1*x**2 + np.cos(x)

def numerical_derivative(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)

x = np.arange(-5, 5, 0.1)
y = y_function(x)

current_pos = (1.5, y_function(1.5))

lr = 0.01

for _ in range(1000):
    grad = numerical_derivative(y_function, current_pos[0])
    new_x = current_pos[0] - lr * grad
    new_y = y_function(new_x)

    current_pos = (new_x, new_y)

    plt.plot(x, y)
    plt.scatter(current_pos[0], current_pos[1], color="red")
    plt.pause(0.001)
    plt.clf()