import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

EMOTION_LABELS = {
    0: 'sadness',
    1: 'joy', 
    2: 'love',
    3: 'anger',
    4: 'fear',
}
def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'@\w+|#\w+', '', text)
        text = re.sub(r'[^a-zA-Z0-9\s!?.,]', '', text)
        text = text.lower().strip()
        return text
    return ""

# Load and prepare data
df = pd.read_csv('text.csv')
df = df.dropna()

# Print initial class distribution
print("label distribution before balancing:")
print(df['label'].value_counts())
print("\nTotal samples:", len(df))

# Balance dataset with a minimum threshold
min_samples = max(min(df['label'].value_counts()), 1000)  # Ensure at least 1000 samples per class
balanced_data = []
for label in df['label'].unique():
    emotion_data = df[df['label'] == label]
    if len(emotion_data) < min_samples:
        # If we have fewer samples than minimum, oversample
        resampled = emotion_data.sample(n=min_samples, replace=True, random_state=42)
        balanced_data.append(resampled)
    else:
        # If we have more samples than minimum, take a random sample
        balanced_data.append(emotion_data.sample(n=min_samples, random_state=42))

df_balanced = pd.concat(balanced_data)

print("\nEmotion distribution after balancing:")
print(df_balanced['label'].value_counts())

# Prepare text data
df_balanced['cleaned_text'] = df_balanced['text'].apply(clean_text)

# Enhanced tokenization
MAX_WORDS = 10000  # Increased vocabulary size
MAX_LEN = 100     # Increased sequence length

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
tokenizer.fit_on_texts(df_balanced['cleaned_text'])
sequences = tokenizer.texts_to_sequences(df_balanced['cleaned_text'])
X = pad_sequences(sequences, maxlen=MAX_LEN)

# Encode labels
le = LabelEncoder()
y = le.fit_transform(df_balanced['label'])
num_classes = len(le.classes_)
y = tf.keras.utils.to_categorical(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Build enhanced model
model = tf.keras.Sequential([
    # Reduced embedding dimensions
    tf.keras.layers.Embedding(MAX_WORDS, 200, input_length=MAX_LEN),
    tf.keras.layers.SpatialDropout1D(0.3),  # Increased dropout
    
    # BiLSTM with L2 regularization
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=False, use_bias=True,
                           kernel_regularizer=tf.keras.regularizers.l2(0.01))
    ),
    tf.keras.layers.Dropout(0.4),  # Increased dropout
    
    # Single dense layer with L2 regularization
    tf.keras.layers.Dense(64, activation='relu',
                         kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),  # Increased dropout
    
    tf.keras.layers.Dense(num_classes, activation='softmax')
])


# Use legacy optimizer for M1 Macs with adjusted learning rate
optimizer = tf.keras.optimizers.legacy.Adam(
    learning_rate=0.0001
)

# Compile with weighted metrics
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# Enhanced callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',  # Changed to monitor loss instead of accuracy
    patience=3,  # Reduced patience
    restore_best_weights=True,
    min_delta=0.001  # Minimum change to qualify as an improvement
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,  # More aggressive reduction
    patience=2,  # Reduced patience
    min_lr=0.00001
)


# Train with class weights
class_weights = dict(enumerate(
    [1.0] * num_classes
))

print("\nTraining model with {} classes: {}".format(num_classes, le.classes_))
model.summary()

# Train model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=64,  # Increased batch size
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, reduce_lr],
    class_weight=class_weights
)

# Evaluate model
test_loss, test_accuracy, test_precision, test_recall = model.evaluate(X_test, y_test)
print(f"\nTest accuracy: {test_accuracy:.4f}")
print(f"Test precision: {test_precision:.4f}")
print(f"Test recall: {test_recall:.4f}")

def predict_emotion(text):
    cleaned = clean_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=MAX_LEN)
    prediction = model.predict(padded)[0]
    
    # Get valid indices (0-4 since we have 5 emotions)
    valid_indices = np.argsort(prediction)[-3:][::-1]
    valid_indices = [idx for idx in valid_indices if idx in EMOTION_LABELS]
    
    predicted_idx = valid_indices[0]  # Take the highest valid prediction
    predicted_class = EMOTION_LABELS[predicted_idx]
    confidence = prediction[predicted_idx]
    
    # Get top 3 emotions (or fewer if we have fewer valid predictions)
    top_3_emotions = {
        EMOTION_LABELS[idx]: float(prediction[idx])
        for idx in valid_indices
    }
    
    return predicted_class, confidence, top_3_emotions


# Save model and tokenizer
model.save("emotion_classifier_v5.h5")

# Example prediction
sample_text = "This is amazing! I'm so happy about this achievement!"
label, confidence, top_3 = predict_emotion(sample_text)
print(f"\nSample text: {sample_text}")
print(f"Predicted label: {label} (confidence: {confidence:.2f})")
print("Top 3 emotions:")
for label, prob in top_3.items():
    print(f"{label}: {prob:.4f}")