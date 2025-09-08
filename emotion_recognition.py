import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras import layers, models
import os
from pathlib import Path
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Parameters
SAMPLE_RATE = 22050  # Standard sample rate for speech processing
DURATION = 3  # Duration in seconds (we'll pad or truncate to this)
N_MFCC = 40  # Increased number of MFCC coefficients for better features
MAX_PAD_LEN = 100  # Reduced padding length

# Emotion labels (based on RAVDESS dataset)
EMOTIONS = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}


def extract_mfcc(file_path, n_mfcc=40, max_pad_len=100):
    """
    Extract MFCC features from audio file and pad/truncate to fixed length
    """
    try:
        # Load audio file
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)

        # Extract MFCC features
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=n_mfcc,
            n_fft=2048,
            hop_length=512
        )

        # Pad or truncate to max_pad_len
        if mfccs.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]

        return mfccs
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def load_data(data_dir):
    """
    Load audio files and extract features
    """
    features = []
    labels = []

    # Check if directory exists and contains files
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} does not exist!")
        return None, None

    # Count WAV files
    wav_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".wav"):
                wav_files.append(os.path.join(root, file))

    print(f"Found {len(wav_files)} WAV files in {data_dir}")

    if len(wav_files) == 0:
        print("No WAV files found. Creating synthetic data for demonstration...")
        # Create more realistic synthetic MFCC data for demonstration
        num_samples = 200
        features = np.random.randn(num_samples, N_MFCC, MAX_PAD_LEN) * 0.1

        # Add some patterns to make the synthetic data more realistic
        for i in range(num_samples):
            emotion_idx = i % 8
            # Add some emotion-specific patterns
            features[i, :5, :] += np.sin(np.linspace(0, 2 * np.pi, MAX_PAD_LEN)) * (emotion_idx + 1) * 0.1
            features[i, 5:10, :] += np.cos(np.linspace(0, 2 * np.pi, MAX_PAD_LEN)) * (emotion_idx + 1) * 0.05

        labels = [list(EMOTIONS.values())[i % 8] for i in range(num_samples)]
        return np.array(features), np.array(labels)

    # Process WAV files
    for file_path in wav_files:
        # Extract emotion from filename (RAVDESS format)
        # Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor.wav
        file_name = os.path.basename(file_path)
        parts = file_name.split('-')

        if len(parts) >= 3:
            emotion_code = parts[2]
            if emotion_code in EMOTIONS:
                emotion = EMOTIONS[emotion_code]

                # Extract features
                mfccs = extract_mfcc(file_path, N_MFCC, MAX_PAD_LEN)

                if mfccs is not None:
                    features.append(mfccs)
                    labels.append(emotion)
                    print(f"Processed {file_name} -> {emotion}")
            else:
                print(f"Skipping {file_name}: emotion code {emotion_code} not recognized")
        else:
            print(f"Skipping {file_name}: filename format not recognized")

    return np.array(features), np.array(labels)


def create_model(input_shape, num_classes):
    """
    Create a CNN model for emotion recognition with appropriate dimensions
    """
    model = models.Sequential([
        # First CNN layer
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        layers.Dropout(0.25),

        # Second CNN layer
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        layers.Dropout(0.25),

        # Third CNN layer - using smaller kernel to avoid dimension issues
        layers.Conv2D(128, (2, 2), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        layers.Dropout(0.25),

        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])

    return model


def plot_training_history(history):
    """
    Plot training history
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Plot accuracy
    axes[0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()

    # Plot loss
    axes[1].plot(history.history['loss'], label='Training Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plot confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.show()


def main():
    # Data directory (update this path to your dataset location)
    data_dir = r"C:\Users\NITHIN\Downloads\Audio_Speech_Actors_01-24"

    # Load data
    print("Loading data and extracting features...")
    features, labels = load_data(data_dir)

    if features is None or len(features) == 0:
        print("No features extracted. Exiting...")
        return

    # Encode labels
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    num_classes = len(le.classes_)

    # Reshape features for CNN (add channel dimension)
    X = features[..., np.newaxis]  # Shape: (n_samples, n_mfcc, max_pad_len, 1)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Number of classes: {num_classes}")
    print(f"Input shape: {X_train[0].shape}")

    # Create model
    input_shape = (N_MFCC, MAX_PAD_LEN, 1)
    model = create_model(input_shape, num_classes)

    # Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Display model architecture
    model.summary()

    # Train model
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,  # Reduced epochs for faster training
        batch_size=32,
        verbose=1
    )

    # Evaluate model
    print("Evaluating model...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Plot training history
    plot_training_history(history)

    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred_classes, le.classes_)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=le.classes_))

    # Save model
    model.save('emotion_recognition_model.h5')
    print("Model saved as 'emotion_recognition_model.h5'")


if __name__ == "__main__":
    main()