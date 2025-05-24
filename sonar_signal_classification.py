import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load the dataset (CSV must have column headers)
df = pd.read_csv('Sonar.csv')

# Separate features (X) and target labels (y)
x = df.iloc[:, :-1].values.astype('float32')  # All columns except the last
y = pd.get_dummies(df.iloc[:, -1], drop_first=True).values.astype('float32')  # Convert labels to binary (e.g., R -> 1, M -> 0)

# Split the data into training and testing sets (75% train, 25% test)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

# Define the neural network architecture
model = keras.Sequential([
    keras.layers.Dense(60, activation='relu'),       # First hidden layer with 60 neurons
    keras.layers.Dropout(0.5),                       # Dropout to prevent overfitting
    keras.layers.Dense(30, activation='relu'),       # Second hidden layer
    keras.layers.Dropout(0.5),
    keras.layers.Dense(15, activation='relu'),       # Third hidden layer
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation='sigmoid')      # Output layer with sigmoid for binary classification
])

# Compile the model
model.compile(
    optimizer='adam', 
    loss='binary_crossentropy', 
    metrics=['accuracy']
)

# Train the model silently (verbose=0)
model.fit(x_train, y_train, epochs=100, batch_size=8, verbose=0)

# Predict on the test set
y_pred = model.predict(x_test)
y_pred = np.round(y_pred)  # Convert probabilities to binary (0 or 1)

# Evaluate predictions using classification metrics
print(classification_report(y_test, y_pred))
