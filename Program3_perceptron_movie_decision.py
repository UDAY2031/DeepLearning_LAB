import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

# Input features (5 samples, 3 features each)
x = np.array([
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 0],
    [0, 0, 1],
    [1, 1, 1]
])

# Target labels
y = np.array([1, 1, 1, 0, 0])

# Manually defined initial weights and bias
weights = np.array([0.2, 0.4, 0.2])  # weight vector for each feature
bias = -0.5                         # bias term

# Create a Perceptron model with maximum 1000 iterations
model = Perceptron(max_iter=1000)

# Fit the model on input data (required before setting weights manually)
model.fit(x, y)

# Overwrite the model's learned weights and bias with custom values
model.coef_ = np.array([weights])    # shape must be (1, n_features)
model.intercept_ = np.array([bias]) # shape must be (1,)

# Predict output using the manually set weights and bias
yp = model.predict(x)

# Display the result in a readable table format
print("input\t\tpred\tactual\tresult")
for i in range(5):
    res = 'ok' if yp[i] == y[i] else 'no'  # check if prediction matches actual label
    print(f"{x[i]}\t\t{yp[i]}\t{y[i]}\t{res}")

print(f"Accuracy : {accuracy_score(yp,y)*100}")
