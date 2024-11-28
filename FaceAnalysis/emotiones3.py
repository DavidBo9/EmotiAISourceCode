import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, classification_report
import seaborn as sns

# Define constants
IMG_SIZE = (96, 96)
BATCH_SIZE = 32
REGION_SIZES = {
    'upper': (32, 32),  # Eyes/eyebrows region
    'middle': (32, 32), # Nose/cheeks region
    'lower': (32, 32)   # Mouth/jaw region
}

def extract_face_regions(image):
    """Extract upper, middle, and lower face regions."""
    # Define region heights (as proportions of total face height)
    h = image.shape[0]
    upper_h = int(h * 0.3)    # Upper 30% for eyes/eyebrows
    middle_h = int(h * 0.3)   # Middle 30% for nose/cheeks
    lower_h = h - upper_h - middle_h  # Remaining for mouth/jaw
    
    # Extract regions
    upper_region = cv2.resize(image[:upper_h, :], REGION_SIZES['upper'])
    middle_region = cv2.resize(image[upper_h:upper_h+middle_h, :], REGION_SIZES['middle'])
    lower_region = cv2.resize(image[upper_h+middle_h:, :], REGION_SIZES['lower'])
    
    return upper_region, middle_region, lower_region

def load_train_set(dirname, map_characters, verbose=True):
    """Load and preprocess training data with facial regions."""
    X_upper = []
    X_middle = []
    X_lower = []
    y_train = []
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    for label, character in map_characters.items():
        files = glob(os.path.join(dirname, character, "*.jpg"))
        if verbose:
            print(f"Reading {len(files)} images from {character}")
        
        for file in tqdm(files):
            try:
                # Read and convert image
                image = cv2.imread(file)
                if image is None:
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Detect face
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                if len(faces) > 0:
                    # Get the largest face
                    x, y, w, h = max(faces, key=lambda x: x[2] * x[3])
                    face = image[y:y+h, x:x+w]
                    
                    # Resize face
                    face = cv2.resize(face, IMG_SIZE)
                    
                    # Extract regions
                    upper, middle, lower = extract_face_regions(face)
                    
                    # Normalize and append
                    X_upper.append(upper / 255.0)
                    X_middle.append(middle / 255.0)
                    X_lower.append(lower / 255.0)
                    y_train.append(label)
                
            except Exception as e:
                print(f"Error processing {file}: {e}")
    
    return (np.array(X_upper), np.array(X_middle), np.array(X_lower)), np.array(y_train)

def create_region_branch(input_shape, name):
    """Create a convolutional branch for processing each facial region."""
    inputs = tf.keras.Input(shape=input_shape)
    
    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    
    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    return tf.keras.Model(inputs=inputs, outputs=x, name=name)

def create_emotion_model(num_classes):
    """Create the complete emotion recognition model."""
    # Input branches
    upper_input = tf.keras.Input(shape=REGION_SIZES['upper'] + (3,))
    middle_input = tf.keras.Input(shape=REGION_SIZES['middle'] + (3,))
    lower_input = tf.keras.Input(shape=REGION_SIZES['lower'] + (3,))
    
    # Process each region
    upper_branch = create_region_branch(REGION_SIZES['upper'] + (3,), 'upper_branch')(upper_input)
    middle_branch = create_region_branch(REGION_SIZES['middle'] + (3,), 'middle_branch')(middle_input)
    lower_branch = create_region_branch(REGION_SIZES['lower'] + (3,), 'lower_branch')(lower_input)
    
    # Combine features with attention
    combined = tf.keras.layers.Concatenate()([upper_branch, middle_branch, lower_branch])
    
    # Final classification layers
    x = tf.keras.layers.Dense(256, activation='relu')(combined)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    return tf.keras.Model(
        inputs=[upper_input, middle_input, lower_input],
        outputs=outputs
    )

if __name__ == "__main__":
    # Define character mapping
    map_characters = {
        0: 'angry',
        1: 'contempt',
        2: 'disgust',
        3: 'fear',
        4: 'happy',
        5: 'neutral',
        6: 'sad',
        7: 'surprised'
    }
    
    # Load and prepare data
    print("Loading training data...")
    X_regions, y = load_train_set('trainingdataset', map_characters)
    
    def split_data(X_regions, y, test_size=0.2, random_state=42):
        """Split data while keeping regions together."""
        # Get indices for train/test split
        n_samples = len(y)
        indices = np.arange(n_samples)
        train_idx, val_idx = train_test_split(indices, test_size=test_size, random_state=random_state)
        
        # Split each region and labels using the indices
        X_train_regions = [X_regions[0][train_idx], X_regions[1][train_idx], X_regions[2][train_idx]]
        X_val_regions = [X_regions[0][val_idx], X_regions[1][val_idx], X_regions[2][val_idx]]
        y_train = y[train_idx]
        y_val = y[val_idx]
        
        return X_train_regions, X_val_regions, y_train, y_val

    # Replace the train_test_split call with:
    X_train_regions, X_val_regions, y_train, y_val = split_data(X_regions, y)
    
    # Convert labels
    num_classes = len(map_characters)
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_val = tf.keras.utils.to_categorical(y_val, num_classes)
    
    # Create and compile model
    model = create_emotion_model(num_classes)
    
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=True
    )
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            min_delta=0.001
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train model
    history = model.fit(
        X_train_regions,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=50,
        validation_data=(X_val_regions, y_val),
        callbacks=callbacks
    )
    
    # Evaluate and save results
    os.makedirs('evaluation1', exist_ok=True)
    
    val_loss, val_accuracy = model.evaluate(X_val_regions, y_val)
    predictions = model.predict(X_val_regions)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(y_val, axis=1)
    
    # Generate reports
    report = classification_report(y_true, y_pred, target_names=list(map_characters.values()))
    print("\nClassification Report:")
    print(report)
    
    with open('evaluation1/classification_report.txt', 'w') as f:
        f.write(report)
    
    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(map_characters.values()),
                yticklabels=list(map_characters.values()))
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('evaluation1/confusion_matrix.png')
    plt.close()
    
    # Plot training history
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('evaluation1/training_history.png')
    plt.close()
    
    # Save model
    model.save("emotion_region_classifier.h5")