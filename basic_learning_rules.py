import numpy as np

# Inputs and targets
x = np.array([
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1],
    [1, 0, 0]
])
t = np.array([1, 1, 0, 0])

# Parameters
ep = 10
lr = 0.1

# Step activation function
step = lambda z: np.where(z >= 0, 1, 0)

# Hebbian Learning
w_heb = np.sum([i * j for i, j in zip(x, t)], axis=0)

# Perceptron Learning
w_per = np.zeros(x.shape[1])
for _ in range(ep):
    for i, j in zip(x, t):
        y = step(np.dot(i, w_per))
        w_per += lr * (j - y) * i

# Delta Learning (Widrow-Hoff)
w_del = np.zeros(x.shape[1])
for _ in range(ep):
    for i, j in zip(x, t):
        y = np.dot(i, w_del)
        w_del += lr * (j - y) * i

# Correlation Learning
w_cor = np.dot(t, x)

# OutStar Learning
w_out = np.random.rand(len(t))
for _ in range(ep):
    w_out += lr * (t - w_out)

# Show results
print("Hebbian Weights:   ", w_heb)
print("Perceptron Weights:", w_per)
print("Delta Weights:     ", w_del)
print("Correlation Weights:", w_cor)
print("OutStar Weights:   ", w_out)
