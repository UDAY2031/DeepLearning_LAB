import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, GRU, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb

# Load IMDB dataset (top 10,000 words), pad to length 200
num_words, maxlen = 10000, 200
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)
x_train, x_test = pad_sequences(x_train, maxlen=maxlen), pad_sequences(x_test, maxlen=maxlen)

# Build GRU-based model
model = tf.keras.Sequential([
    Embedding(num_words, 128),     # Converts word indices to dense vectors
    GRU(64),                        # GRU layer for sequence learning
    Dense(1, activation='sigmoid') # Output layer for binary classification
])

# Compile and train model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3, batch_size=64, validation_split=0.2)

# Evaluate on test data
_, acc = model.evaluate(x_test, y_test)
print(f"\nTest Accuracy: {acc:.4f}")

# Prediction function
def sent(r):
    p = pad_sequences([r], maxlen=maxlen)  # pad the input review
    s = model.predict(p)[0][0]             # get prediction score
    print(f"Sentiment: {'positive' if s >= 0.5 else 'negative'} | Score: {s:.2f}")

# Show examples
sent(x_test[np.where(y_test == 1)[0][0]])  # sample positive
sent(x_test[np.where(y_test == 0)[0][0]])  # sample negative
