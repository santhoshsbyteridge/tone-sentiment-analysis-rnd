import librosa
import librosa.display
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load audio and extract features
def extract_features(audio_path, sr=16000):
    y, sr = librosa.load(audio_path, sr=sr)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

# Define WaveNet model
def build_wavenet_model(input_shape, num_classes):
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Causal Convolution layers
    x = tf.keras.layers.Conv1D(64, kernel_size=2, dilation_rate=1, padding='causal', activation='relu')(inputs)
    x = tf.keras.layers.Conv1D(128, kernel_size=2, dilation_rate=2, padding='causal', activation='relu')(x)
    x = tf.keras.layers.Conv1D(256, kernel_size=2, dilation_rate=4, padding='causal', activation='relu')(x)

    # Global pooling and output layer
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Load an example audio file
audio_path = "angry-speech.mp3"
features = extract_features(audio_path)

# Reshape features for model input
features = np.expand_dims(features.T, axis=0)  # (1, TimeSteps, MelBands)

# Define and compile the model
num_classes = 4  # Example: Happy, Sad, Angry, Neutral
model = build_wavenet_model(input_shape=(features.shape[1], features.shape[2]), num_classes=num_classes)

# Predict emotion (assuming trained model weights are loaded)
predictions = model.predict(features)
predicted_emotion = np.argmax(predictions)

emotion_labels = ["Happy", "Sad", "Angry", "Neutral"]
print(f"Predicted Emotion: {emotion_labels[predicted_emotion]}")
