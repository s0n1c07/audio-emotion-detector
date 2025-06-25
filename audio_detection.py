import streamlit as st
import numpy as np
import librosa
import joblib
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from lightgbm import LGBMClassifier

# --- Emotion Labels ---
class_labels = ["angry", "calm", "fearful", "happy", "neutral", "sad", "surprised"]
emotion_emojis = {
    "neutral": "😐", "calm": "😌", "happy": "😊", "sad": "😢",
    "angry": "😠", "fearful": "😨", "surprised": "😲"
}
emotions_to_display = ", ".join([f"{emotion_emojis[e]} {e.capitalize()}" for e in class_labels])

# --- Feature Extraction ---
def extract_features_from_signal(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc, axis=1)
    return mfcc_mean

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None, res_type='kaiser_fast')
    return extract_features_from_signal(y, sr)

# --- Load Assets ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load("emotion_model_7class.pkl")
        scaler = joblib.load("scaler_7class.pkl")
        encoder = joblib.load("label_encoder_7class.pkl")
        return model, scaler, encoder
    except FileNotFoundError as e:
        st.error(f"Missing model/scaler/encoder file: {e}")
        st.stop()

# --- UI ---
st.set_page_config(page_title="Audio Emotion Detector", layout="centered")
st.title("🎧 Audio Emotion Detection")
st.markdown(f"""
Upload your `.wav` file and our AI will analyze the emotion expressed in the audio.  
This model was trained to detect **7 emotions**: {emotions_to_display}.
""")

uploaded_file = st.file_uploader("Choose an audio file (.wav only)", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    temp_file_path = "temp_uploaded_audio.wav"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        with st.spinner('Analyzing the audio...'):
            model, scaler, encoder = load_assets()
            features = extract_features(temp_file_path)
            features_scaled = scaler.transform([features])
            prediction = model.predict(features_scaled)
            proba = model.predict_proba(features_scaled)[0]
            predicted_emotion = encoder.inverse_transform(prediction)[0]
            predicted_emoji = emotion_emojis.get(predicted_emotion, "❓")

        st.success("Analysis Complete!")
        st.markdown(f"### 🎭 Emotion Detected: **{predicted_emoji} {predicted_emotion.capitalize()}**")

        # Show probability distribution as bar chart
        proba_dict = dict(zip(encoder.classes_, proba))
        st.bar_chart(proba_dict)

    except Exception as e:
        st.error(f"⚠️ An error occurred during processing:\n\n{str(e)}")

    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
