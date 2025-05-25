import numpy as np
import matplotlib.pyplot as plt

# Activation functions
def sigmoid(x): return 1 / (1 + np.exp(-x))
def tanh(x): return np.tanh(x)
def relu(x): return np.maximum(0, x)
def leaky(x): return np.where(x > 0, x, 0.01 * x)
def softmax(x): 
    e = np.exp(x - np.max(x))
    return e / e.sum()

# Input
x = np.linspace(-5, 5, 100)

# Plot all activation functions
plt.figure(figsize=(12, 6))

plt.subplot(2, 3, 1)
plt.plot(x, sigmoid(x)); plt.title("Sigmoid"); plt.grid(True)

plt.subplot(2, 3, 2)
plt.plot(x, tanh(x)); plt.title("Tanh"); plt.grid(True)

plt.subplot(2, 3, 3)
plt.plot(x, relu(x)); plt.title("ReLU"); plt.grid(True)

plt.subplot(2, 3, 4)
plt.plot(x, leaky(x)); plt.title("Leaky ReLU"); plt.grid(True)

# Softmax (bar plot)
x_soft = np.array([1, 2, 3, 4, 5])
y_soft = softmax(x_soft)
plt.subplot(2, 3, 5)
plt.bar(range(len(y_soft)), y_soft)
plt.title("Softmax"); plt.grid(True)

plt.tight_layout()
plt.show()
