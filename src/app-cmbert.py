import torch
import librosa
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

# Load pre-trained model and feature extractor
MODEL_NAME = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_NAME)
model.config.gradient_checkpointing = False #disable gradient checkpointing

# Emotion labels from the model
EMOTIONS = ["sadness", "neutral", "happiness", "anger", "fear", "disgust", "surprise"]

def predict_emotion(audio_path):
    # Load and preprocess audio
    speech_array, sampling_rate = librosa.load(audio_path, sr=16000)
    input_values = feature_extractor(speech_array, sampling_rate=16000, return_tensors="pt", padding=True).input_values

    # Perform inference
    with torch.no_grad():
        logits = model(input_values).logits
        predicted_class = torch.argmax(logits, dim=-1).item()

    return EMOTIONS[predicted_class]

if __name__ == "__main__":
    audio_file = "your-audio-file.wav"  # Replace with your audio file
    emotion = predict_emotion(audio_file)
    print(f"Predicted Emotion: {emotion}")