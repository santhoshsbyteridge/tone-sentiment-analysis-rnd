import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import soundfile as sf
import io
import os
import streamlit as st
import xgboost as xgb # Import XGBoost

def extract_features(audio_data, sample_rate):
    """Extracts relevant audio features."""

    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)

    zcr = librosa.feature.zero_crossing_rate(y=audio_data)
    zcr_mean = np.mean(zcr)

    rmse = librosa.feature.rms(y=audio_data)
    rmse_mean = np.mean(rmse)

    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate)
    chroma_mean = np.mean(chroma.T, axis=0)

    features = np.concatenate((mfccs_mean, [zcr_mean, rmse_mean], chroma_mean))
    return features

# support vector machine (svm)
def train_model(audio_files, labels):
    """Trains an SVM model."""

    features = []
    for audio_data, sample_rate in audio_files:
        features.append(extract_features(audio_data, sample_rate))

    X = np.array(features)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = SVC(kernel='rbf', C=10, gamma=0.1)  # Tuned parameters
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy}")

    return model, scaler

# random forest
# def train_model(audio_files, labels):
#     features = []
#     for audio_data, sample_rate in audio_files:
#         features.append(extract_features(audio_data, sample_rate))

#     X = np.array(features)
#     y = np.array(labels)

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     model = RandomForestClassifier(n_estimators=100, random_state=42)
#     model.fit(X_train_scaled, y_train)

#     y_pred = model.predict(X_test_scaled)
#     accuracy = accuracy_score(y_test, y_pred)
#     print(f"Model Accuracy: {accuracy}")

#     return model, scaler

def predict_emotion(audio_data, sample_rate, model, scaler):
    """Predicts emotion from audio data."""

    features = extract_features(audio_data, sample_rate)
    features_scaled = scaler.transform(features.reshape(1, -1))
    prediction = model.predict(features_scaled)[0]
    return prediction

def main():
    st.title("Audio sentiment analysis")

    positive_datas = []
    negative_datas = []
    neutral_datas = []

    df_base = pd.read_csv('file-path/TRAIN.csv')

    for index, row in df_base.iterrows():
        data = row.to_dict()
        filename = data['Filename']
        category = data['Class']
        # print(f"Row {index}: {data}") 
        # print(f"Row {index}: {data['Filename']}", ' ---- ', f'your-audio-file-path/{filename}') 
        # print(f"Row {index}: {data['Class']}") 
        if(category == 'Positive'):
            positive_datas.append(
                (librosa.load(f"your-audio-file-path/{filename}")[0], librosa.load(f"your-audio-file-path/{filename}")[1])
            )
        elif(category == 'Negative'):
            negative_datas.append(
                (librosa.load(f"your-audio-file-path/{filename}")[0], librosa.load(f"your-audio-file-path/TRAIN/{filename}")[1])
            )
        elif(category == 'Neutral'):
            neutral_datas.append(
                (librosa.load(f"your-audio-file-path{filename}")[0], librosa.load(f"your-audio-file-path{filename}")[1])
            )

    
    sample_data = {
        "Positive": positive_datas,
        "Negative": negative_datas,
        "Neutral": neutral_datas
    }

    # print(sample_data)

    audio_files = []
    labels = []
    for emotion, files in sample_data.items():
        # print('emotion: ', emotion)
        for audio_data, sample_rate in files:
            # print('sample data: ', audio_data, ' - ', sample_rate)
            audio_files.append((audio_data, sample_rate))
            labels.append(emotion)

    model, scaler = train_model(audio_files, labels)

    # print('model: ', model)
    # print('scaler: ', scaler)


    audio_data, sample_rate = (librosa.load("your-audio-file-path/348.wav")[0], librosa.load("your-audio-file-path/348.wav")[1])
    # print('uploaded data: ', audio_data, ' - ', sample_rate)

    prediction = predict_emotion(audio_data, sample_rate, model, scaler)
    print('PREDICTION - ', prediction)
    # st.write(f"Predicted Emotion: {prediction}")


if __name__ == "__main__":
    main()