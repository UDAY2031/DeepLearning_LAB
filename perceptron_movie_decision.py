import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

# Input (3 features): [Hero, Exam, Weather]
x = np.array([
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 0],
    [0, 0, 1],
    [1, 1, 1]
])

# Output: 1 = Go, 0 = Don't go
y = np.array([1, 1, 1, 0, 0])

# Train model
model = Perceptron(max_iter=1000, eta0=1.0)
model.fit(x, y)

# Predict
yp = model.predict(x)

# Show results
print("Input\t\tPred\tTrue\tResult")
print("-" * 40)
for i in range(len(x)):
    res = "OK" if yp[i] == y[i] else "X"
    print(f"{x[i]}\t{yp[i]}\t{y[i]}\t{res}")

# Accuracy
print("\nAccuracy: {:.0f}%".format(accuracy_score(y, yp) * 100))
