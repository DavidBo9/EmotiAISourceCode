import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

# Data preprocessing
def clean_text(text):
    if isinstance(text, str):
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^a-zA-Z0-9\s!?.,]', '', text)
        # Convert to lowercase
        text = text.lower().strip()
        return text
    return ""

# Load and prepare data
df = pd.read_csv('sentiment-emotion.csv')
df = df.dropna()

# Balance dataset
min_samples = df['sentiment'].value_counts().min()
balanced_data = []
for sentiment in df['sentiment'].unique():
    balanced_data.append(df[df['sentiment'] == sentiment].sample(n=min_samples, random_state=42))
df_balanced = pd.concat(balanced_data)

# Prepare text data
df_balanced['cleaned_text'] = df_balanced['Text'].apply(clean_text)

# Tokenization
MAX_WORDS = 15000  # Increased vocabulary size
MAX_LEN = 100     # Increased sequence length

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
tokenizer.fit_on_texts(df_balanced['cleaned_text'])
sequences = tokenizer.texts_to_sequences(df_balanced['cleaned_text'])
X = pad_sequences(sequences, maxlen=MAX_LEN)

# Encode labels
le = LabelEncoder()
y = le.fit_transform(df_balanced['sentiment'])
y = tf.keras.utils.to_categorical(y)

# Split data with a larger validation set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Build model with more regularization
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(MAX_WORDS, 200, input_length=MAX_LEN),
    tf.keras.layers.SpatialDropout1D(0.3),  # Spatial dropout for embeddings
    
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100, return_sequences=True)),
    tf.keras.layers.Dropout(0.3),
    
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50)),
    tf.keras.layers.Dropout(0.3),
    
    tf.keras.layers.Dense(100, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.4),
    
    tf.keras.layers.Dense(50, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.4),
    
    tf.keras.layers.Dense(3, activation='softmax')
])

# Use legacy optimizer as recommended for M1 Macs
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)  # Reduced learning rate

# Compile model
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Add more callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=7,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=0.00001
)

# Train model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=64,  # Increased batch size
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, reduce_lr],
    class_weight={0: 1.0, 1: 1.0, 2: 1.0}  # Adjust these weights if needed
)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nTest accuracy: {accuracy:.4f}")

# Function for making predictions
def predict_sentiment(text):
    cleaned = clean_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=MAX_LEN)
    prediction = model.predict(padded)[0]
    predicted_class = le.inverse_transform([np.argmax(prediction)])[0]
    return predicted_class, prediction

# Example usage
sample_text = "Chuby is a faggot for not opening the server"
sentiment, probabilities = predict_sentiment(sample_text)
print(f"\nSample text: {sample_text}")
print(f"Predicted sentiment: {sentiment}")
print(f"Probabilities: {probabilities}")

# Plot training history
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

model.save("analisis_tweets.h5")