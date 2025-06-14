import streamlit as st
import numpy as np
import librosa
import joblib
import os

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier

# Emotion map
emotion_map = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"
}
class_labels = sorted(list(set(emotion_map.values())))

# ----------- Feature Extraction -----------
def extract_features_from_signal(y, sr):
    mfcc  = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean, mfcc_std = np.mean(mfcc, axis=1), np.std(mfcc, axis=1)
    delta1 = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    spec_con = librosa.feature.spectral_contrast(y=y, sr=sr)
    chroma  = librosa.feature.chroma_stft(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(chroma=chroma, sr=sr)
    return np.hstack([
        mfcc_mean, mfcc_std,
        np.mean(delta1, axis=1), np.mean(delta2, axis=1),
        np.mean(spec_con, axis=1),
        np.mean(tonnetz, axis=1)
    ])

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None, res_type='kaiser_fast')
    return extract_features_from_signal(y, sr)

# ----------- Load Pretrained Model -----------
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    encoder = joblib.load("encoder.pkl")
    return model, scaler, encoder

# ----------- Streamlit UI -----------
st.set_page_config(page_title="Emotion Detector", layout="centered")
st.title("🎧 Audio Emotion Detection")
st.markdown("""
Upload your `.wav` file below and our emotion detection system will analyze it.  
This model detects **8 emotions**: 😐 Neutral, 😌 Calm, 😊 Happy, 😢 Sad, 😠 Angry, 😨 Fearful, 🤢 Disgust, 😲 Surprised.
""")
# st.write("Upload a `.wav` file and this app will classify the emotion expressed in it.")

uploaded_file = st.file_uploader("Choose an audio file (.wav only)", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    try:
        # Load model and preprocessors
        model, scaler, encoder = load_model()

        # Extract features
        features = extract_features("temp.wav")
        features_scaled = scaler.transform([features])

        # Predict
        prediction = model.predict(features_scaled)
        predicted_emotion = encoder.inverse_transform(prediction)[0]
        proba = model.predict_proba(features_scaled)[0]

        # Output
        st.markdown(f"### 🎭 Emotion Detected: `{predicted_emotion}`")
        st.bar_chart(data=dict(zip(encoder.classes_, proba)))

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
