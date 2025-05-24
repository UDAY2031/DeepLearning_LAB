import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# 1. Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. Preprocess data - normalize and add channel dimension
x_train = x_train[..., None] / 255.0  # Shape: (60000, 28, 28, 1)
x_test = x_test[..., None] / 255.0    # Shape: (10000, 28, 28, 1)

# 3. Build CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')  # 10 classes (digits 0–9)
])

# 4. Compile and train the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=64, verbose=0)

# 5. Evaluate on test data
_, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy: {acc:.2f}")

# 6. Predict first 10 test images
predictions = model.predict(x_test[:10])
predicted_labels = tf.argmax(predictions, axis=1)

# 7. Display first 5 test images with predicted labels
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(x_test[i].squeeze(), cmap='gray')
    plt.title(f"Pred: {predicted_labels[i].numpy()}")
    plt.axis('off')

plt.tight_layout()
plt.show()
