# Sentiment Analysis Using Audio Tone

## Overview
This project explores sentiment analysis using the tone of audio, leveraging **Amazon Chime SDK** and **machine learning models** including **Support Vector Machines (SVM), CM-BERT, and Wav2Vec2** for sentiment classification. The research was guided by multiple reference studies and implementations.

## Objectives
- Do a sentiment analysis using **Amazon Chime SDK**.
- Process and analyze tonal variations in audio.
- Utilize **SVM, CM-BERT, and Wav2Vec2** models for sentiment classification.
- Classify sentiments into predefined categories (e.g., Positive, Negative, Neutral).

## Technology Stack
- **Python** – Primary language for machine learning.
- **Librosa** – Feature extraction from audio.
- **Scikit-learn** – Implementation of SVM.
- **Hugging Face Transformers** – CM-BERT & Wav2Vec2 models.
- **NumPy & Pandas** – Data handling.
- **Matplotlib & Seaborn** – Visualization.
- **Amazon Chime SDK** – Readily avalaible SDK used for sentiment analysis of the audio.

## Implementation Steps

1. **Feature Extraction**
   - Using `Librosa`, extract Mel-frequency cepstral coefficients (MFCCs) and other audio features.
   - Normalize and preprocess the extracted features.

2. **Model Training & Evaluation**
   - **SVM Model** (Implemented in `app-svm.py`)
   - **CM-BERT Model** (Implemented in `app-cmbert.py`)
   - **Wav2Vec2 Model** (Implemented in `app-wavenet.py`)
   - Models are trained using labeled datasets.
   - Evaluated using accuracy, precision, recall, and F1-score.

3. **Sentiment Classification**
   - The trained models predict sentiments (Positive, Negative, Neutral) based on extracted features.

4. **Codebase Overview**
   - `app-chime-1.py` & `app-chime-2.py` – Amazon Chime SDK-based audio processing. (Standalone files)
   - `sentiment-analysis.ipynb` – Jupyter Notebook for exploratory data analysis and experiments.

## Installation & Setup
1. Clone the repository:
   ```sh
   git clone https://github.com/santhoshsbyteridge/tone-sentiment-analysis-rnd.git
   cd tone-sentiment-analysis-rnd
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the model:
   ```sh
   python app-svm.py  # or python app-cmbert.py / python app-wavenet.py
   ```

## References
- [Audio Sentiment Analysis Research](https://research.aimultiple.com/audio-sentiment-analysis/)
- [Open-source Sentiment Analysis Tools](https://research.aimultiple.com/open-source-sentiment-analysis/)
- [Kaggle: Audio Sentiment Analysis](https://www.kaggle.com/code/imsparsh/audio-sentiment-analysis/notebook)
- [Sentiment Analysis Datasets](https://research.aimultiple.com/sentiment-analysis-dataset/)
- [Amazon Chime SDK’s Voice Tone Analysis](https://www.amazon.science/blog/how-amazon-chime-sdks-voice-tone-analysis-works)
- [AWS Chime SDK](https://aws.amazon.com/chime/chime-sdk/)
- [Hugging Face Wav2Vec2 Model](https://huggingface.co/ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition)
- [GeminiAI]https://g.co/gemini/share/ca6041b3a532

---
This README now includes reference links, uploaded code files, and expanded explanations.

