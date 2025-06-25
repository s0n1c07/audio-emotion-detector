import streamlit as st
import librosa
import numpy as np
import joblib
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="MARS: Emotion Recognition",
    page_icon="🎵",
    layout="centered"
)

# --- Emotion Emojis ---
emotion_emojis = {
    "calm": "😌", "happy": "😊", "sad": "😢",
    "angry": "😠", "fearful": "😨", "neutral": "😐", "surprised": "😲"
}

# --- Feature Extraction Function ---
def extract_features(file, n_mfcc=50):
    """Extracts MFCCs from an audio file."""
    try:
        y, sr = librosa.load(file, sr=None, res_type='kaiser_fast')
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfcc.T, axis=0)
    except Exception as e:
        st.error(f"❌ Error processing audio file: {e}")
        return None

# --- Asset Loader ---
@st.cache_resource
def load_assets():
    model_path = "emotion.pkl"
    scaler_path = "scaler.pkl"
    le_path = "label.pkl"

    if not all(os.path.exists(p) for p in [model_path, scaler_path, le_path]):
        st.error("🚫 Required model files not found. Please upload the .pkl files.")
        st.stop()

    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        le = joblib.load(le_path)
    except Exception as e:
        st.error(f"⚠️ Failed to load model or preprocessing objects: {e}")
        st.stop()

    return model, scaler, le

# Load model, scaler, encoder
model, scaler, le = load_assets()

# --- UI Layout ---
st.title("🎧 MARS: Music & Speech Emotion Recognition")

st.markdown("""
This app predicts the **emotion** expressed in `.wav` audio files.  
It was trained on the RAVDESS dataset and can recognize 7 emotions:

**😐 Neutral, 😌 Calm, 😊 Happy, 😢 Sad, 😠 Angry, 😨 Fearful, 😲 Surprised**
""")

st.info("Upload a `.wav` file and click **Classify Emotion** to get the prediction.", icon="📂")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload a WAV file", type="wav")

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    if st.button("🎙️ Classify Emotion"):
        with st.spinner("🔍 Extracting features and predicting..."):
            features = extract_features(uploaded_file, n_mfcc=50)

            if features is None:
                st.stop()

            if features.shape[0] != 50:
                st.error("❌ Extracted MFCCs do not match expected shape (50). Check training setup.")
                st.stop()

            try:
                features_scaled = scaler.transform(features.reshape(1, -1))
                prediction_encoded = model.predict(features_scaled)
                prediction_proba = model.predict_proba(features_scaled)[0]

                emotion = le.inverse_transform(prediction_encoded)[0]
                emoji = emotion_emojis.get(emotion, "🎭")

                st.success(f"### Predicted Emotion: **{emoji} {emotion.capitalize()}**")

                st.markdown("#### 📊 Prediction Probabilities:")
                proba_dict = dict(zip(le.classes_, prediction_proba))

                for emotion_class, prob in sorted(proba_dict.items(), key=lambda x: x[1], reverse=True):
                    col1, col2 = st.columns([3, 7])
                    with col1:
                        st.write(f"**{emotion_emojis.get(emotion_class)} {emotion_class.capitalize()}**")
                    with col2:
                        st.progress(prob, text=f"{prob:.1%}")

            except Exception as e:
                st.error(f"⚠️ Prediction failed: {e}")
