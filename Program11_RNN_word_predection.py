import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Step 1: Sample training data
text = [
    'i love to eat oranges',
    'she loves to eat oranges',
    'he like to eat guava'
]

# Step 2: Tokenize words
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text)

# Get total vocabulary size (+1 for padding)
vocab_size = len(tokenizer.word_index) + 1

# Convert text to sequence of word indexes
sequences = tokenizer.texts_to_sequences(text)

# Step 3: Prepare input-output pairs
x = []  # Input sequence (3 words)
y = []  # Output word (next word)

for seq in sequences:
    for i in range(3, len(seq)):
        x.append(seq[i-3:i])  # Last 3 words
        y.append(seq[i])      # Next word

# Step 4: Padding & one-hot encoding
x = pad_sequences(x, maxlen=3)
y = tf.keras.utils.to_categorical(y, num_classes=vocab_size)

# Step 5: Build the RNN model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=8, input_length=3),
    SimpleRNN(16, activation='relu'),
    Dense(vocab_size, activation='softmax')
])

# Step 6: Compile & train the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x, y, epochs=50)

# Step 7: Predict next word
seed_text = 'loves to eat'
seed_seq = tokenizer.texts_to_sequences([seed_text])[0]
seed_seq = pad_sequences([seed_seq], maxlen=3)

# Get predicted word index
pred = model.predict(seed_seq)
pred_index = np.argmax(pred)

# Convert index back to word
predicted_word = tokenizer.index_word.get(pred_index)

# Step 8: Output result
print(f"Next word after '{seed_text}': {predicted_word}")
