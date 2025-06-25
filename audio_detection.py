import streamlit as st
import numpy as np
import librosa
import joblib
import os

# Important: These libraries are required to unpickle the saved files.
# Make sure you have lightgbm and scikit-learn installed.
from sklearn.preprocessing import RobustScaler, LabelEncoder
from lightgbm import LGBMClassifier

# --- Emotion Labels for the 7-class model ---
class_labels = ["angry", "calm", "fearful", "happy", "neutral", "sad", "surprised"]
emotion_emojis = {
    "neutral": "😐", "calm": "😌", "happy": "😊", "sad": "😢",
    "angry": "😠", "fearful": "😨", "surprised": "😲"
}
emotions_to_display = ", ".join([f"{emotion_emojis[e]} {e.capitalize()}" for e in class_labels])


# --- Feature Extraction (CORRECTED to match the 40-feature model) ---

def extract_features_from_signal(y, sr):
    """
    Extracts a 40-feature vector (MFCC mean) from an audio signal.
    This MUST match the feature engineering used to train the model.
    """
    # Calculate 40 MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    # Calculate the mean of the MFCCs over time
    mfcc_mean = np.mean(mfcc, axis=1)
    # Return the 40-element feature vector
    return mfcc_mean

def extract_features(file_path):
    """Loads an audio file and extracts features."""
    # res_type='kaiser_fast' is good for speed. sr=None preserves the original sampling rate.
    y, sr = librosa.load(file_path, sr=None, res_type='kaiser_fast')
    return extract_features_from_signal(y, sr)


# --- Load Pretrained Model, Scaler, and Encoder ---
@st.cache_resource
def load_assets():
    """Loads the pre-trained model, scaler, and encoder."""
    model = joblib.load("emotion_model_7class.pkl")
    scaler = joblib.load("scaler_7class.pkl")
    encoder = joblib.load("label_encoder_7class.pkl")
    return model, scaler, encoder


# --- Streamlit User Interface ---
st.set_page_config(page_title="Audio Emotion Detector", layout="centered")
st.title("🎧 Audio Emotion Detection")
st.markdown(f"""
Upload your `.wav` file and our AI will analyze the emotion expressed in the audio.  
This model was trained to detect **7 emotions**: {emotions_to_display}.
""")

uploaded_file = st.file_uploader("Choose an audio file (.wav only)", type=["wav"])

if uploaded_file is not None:
    # Display the audio player
    st.audio(uploaded_file, format='audio/wav')

    # To process the file, we save it temporarily
    temp_file_path = "temp_uploaded_audio.wav"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        # Show a spinner during processing
        with st.spinner('Analyzing the audio...'):
            # Load the trained assets
            model, scaler, encoder = load_assets()

            # Extract features from the audio file (now returns 40 features)
            features = extract_features(temp_file_path)

            # Scale the 40 features using the loaded RobustScaler
            # The input to the scaler must be 2D, so we wrap `features` in a list
            features_scaled = scaler.transform([features])

            # Make a prediction and get probabilities
            prediction = model.predict(features_scaled)
            proba = model.predict_proba(features_scaled)[0]

            # Decode the predicted label from a number to the emotion name
            predicted_emotion = encoder.inverse_transform(prediction)[0]
            predicted_emoji = emotion_emojis.get(predicted_emotion, "❓")

        # Display the results
        st.success("Analysis Complete!")
        st.markdown(f"### 🎭 Emotion Detected: **{predicted_emoji} {predicted_emotion.capitalize()}**")

        # Create a dictionary of emotions and their probabilities for the chart
        proba_dict = dict(zip(encoder.classes_, proba))
        st.bar_chart(proba_dict)

    except Exception as e:
        st.error(f"⚠️ An error occurred during processing: {e}")
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
