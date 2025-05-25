# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Dropout

# Load the MNIST dataset (handwritten digits 0–9)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the pixel values (scale between 0 and 1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# Build the neural network model
model = Sequential([
    # Flatten the 28x28 input images into a 1D vector
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    BatchNormalization(),     # Normalize activations to improve training
    Dropout(0.2),              # Prevent overfitting by randomly dropping 20% of neurons
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(10, activation='softmax')
])

# Compile the model with optimizer, loss function and evaluation metric
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model for 3 epochs with batch size 32 and validate on test set
history = model.fit(
    x_train, y_train,
    epochs=3,
    batch_size=32,
    validation_data=(x_test, y_test)
)

# Evaluate the model on training data
train_loss, train_accuracy = model.evaluate(x_train, y_train)

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(x_test, y_test)

# Print the results
print("Training Loss:", train_loss)
print("Training Accuracy:", train_accuracy)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
