# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to the range [0, 1]
x_train, x_test = x_train / 255.0, x_test / 255.0

# One-hot encode the labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Define the class names for CIFAR-10
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# Build a simple softmax classifier model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(32, 32, 3)),     # Flatten the input image
    tf.keras.layers.Dense(10, activation='softmax')       # Output layer with 10 classes
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train,
          epochs=20,
          batch_size=64,
          validation_data=(x_test, y_test))

# Pick one test image to make a prediction
img = np.expand_dims(x_test[30], axis=0)  # Add batch dimension for prediction

# Display the selected image
plt.imshow(x_test[30])
plt.axis('off')
plt.show()

# Predict the class of the selected image
pred = model.predict(img)
label = np.argmax(pred)          # Get the class index with highest probability
conf = np.max(pred) * 100        # Confidence percentage

# Print the predicted class and confidence
print(f"Predicted Class: {classes[label]}")
print(f"Confidence: {conf:.2f}%")
