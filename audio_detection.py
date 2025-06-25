# app.py

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

# --- Feature Extraction Function ---
# This function must be identical to the one used for training
def extract_features(file, n_mfcc=50):
    """Extracts MFCCs from an audio file."""
    try:
        # For Streamlit's UploadedFile object, 'file' is the object itself
        y, sr = librosa.load(file, sr=None, res_type='kaiser_fast')
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfcc.T, axis=0)
    except Exception as e:
        st.error(f"Error processing audio file: {e}")
        return None

# --- Asset Loading ---
# Use st.cache_resource to load model and objects only once
@st.cache_resource
def load_assets():
    """Loads the pre-trained model, scaler, and label encoder."""
    model_path = "emotion_model_7class.pkl"
    scaler_path = "scaler_7class.pkl"
    le_path = "label_encoder_7class.pkl"

    if not all(os.path.exists(p) for p in [model_path, scaler_path, le_path]):
        st.error("Model assets not found! Please run the `train_model.py` script first to generate them.")
        st.stop()
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    le = joblib.load(le_path)
    return model, scaler, le

model, scaler, le = load_assets()

# --- UI Layout ---
st.title("MARS: Music & Speech Emotion Recognition 🎵")

st.markdown("""
This application predicts emotion from audio files. It was trained on the RAVDESS dataset to recognize 7 emotions: 
**calm, happy, sad, angry, fearful, neutral, and surprised.**
""")

st.info("Upload a `.wav` file and click 'Classify Emotion' to see the prediction.", icon="💡")

# File uploader
uploaded_file = st.file_uploader("Choose a WAV file", type="wav")

if uploaded_file is not None:
    # Display audio player
    st.audio(uploaded_file, format='audio/wav')

    # Classify button
    if st.button("Classify Emotion", type="primary"):
        with st.spinner('Analyzing the audio...'):
            # 1. Extract features
            features = extract_features(uploaded_file, n_mfcc=50)

            if features is not None:
                # 2. Reshape and scale features
                features_reshaped = features.reshape(1, -1)
                features_scaled = scaler.transform(features_reshaped)

                # 3. Predict emotion
                prediction_encoded = model.predict(features_scaled)
                prediction_proba = model.predict_proba(features_scaled)

                # 4. Decode prediction to a human-readable label
                emotion = le.inverse_transform(prediction_encoded)[0]

                # --- Display Results ---
                st.success(f"### Predicted Emotion: **{emotion.capitalize()}**")

                st.write("#### Prediction Probabilities:")
                # Create a dictionary of emotions and their probabilities
                probabilities = dict(zip(le.classes_, prediction_proba[0]))
                
                # Display probabilities in a neat way
                for emotion_class, prob in sorted(probabilities.items(), key=lambda item: item[1], reverse=True):
                    col1, col2 = st.columns([3, 7])
                    with col1:
                        st.write(f"**{emotion_class.capitalize()}**")
                    with col2:
                        st.progress(prob, text=f"{prob:.1%}")
