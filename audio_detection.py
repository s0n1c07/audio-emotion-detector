import streamlit as st
import numpy as np
import librosa
import joblib
import os

# --- Feature Extraction (from the final model in mars.py) ---
# This function must be identical to the one used for training.
def extract_features(file_path):
    """Extracts MFCC features from an audio file."""
    try:
        # Load audio file. sr=None preserves the original sampling rate.
        # res_type='kaiser_fast' is used for faster processing.
        y, sr = librosa.load(file_path, sr=None, res_type='kaiser_fast')
        
        # Extract 50 MFCCs, as specified in the final model training.
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=50)
        
        # Return the mean of the MFCCs across time.
        return np.mean(mfcc.T, axis=0)
    except Exception as e:
        st.error(f"Error loading or processing audio file: {e}")
        return None

# --- Load Pretrained Model, Scaler, and Encoder ---
# Use st.cache_resource to load these assets only once, improving performance.
@st.cache_resource
def load_assets():
    """Loads the pre-trained model, scaler, and label encoder."""
    try:
        model = joblib.load("emotion_model_7class.pkl")
        scaler = joblib.load("scaler_7class.pkl")
        encoder = joblib.load("label_encoder_7class.pkl")
        return model, scaler, encoder
    except FileNotFoundError:
        st.error(
            "Model files not found. Please ensure 'emotion_model_7class.pkl', "
            "'scaler_7class.pkl', and 'label_encoder_7class.pkl' are in the same directory as the app."
        )
        return None, None, None

# --- Streamlit UI ---
st.set_page_config(page_title="MARS Emotion Detector", layout="centered")
st.title("🎧 MARS: Mood & Audio Recognition System")

# Load model assets
model, scaler, encoder = load_assets()

# Only proceed if the assets were loaded successfully
if model and scaler and encoder:
    # Dynamically generate the list of supported emotions from the encoder
    class_names = [name.capitalize() for name in encoder.classes_]
    class_list = ", ".join(f"`{name}`" for name in class_names)

    st.markdown(f"""
    Upload a `.wav` audio file and our system will analyze the emotion conveyed.
    This model recognizes **7 emotions**: {class_list}.
    """)

    # File uploader widget
    uploaded_file = st.file_uploader("Choose an audio file (.wav only)", type=["wav"])

    if uploaded_file is not None:
        # Display the audio player for the uploaded file
        st.audio(uploaded_file, format='audio/wav')

        # To use librosa, we need a file path. So we save the uploaded file temporarily.
        temp_file_path = "temp_audio.wav"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Display a status message while processing
        with st.spinner("Analyzing audio..."):
            try:
                # 1. Extract features from the temporary audio file
                features = extract_features(temp_file_path)

                if features is not None:
                    # 2. Scale the features using the loaded scaler
                    # Reshape features to a 2D array for the scaler
                    features_scaled = scaler.transform([features])

                    # 3. Predict emotion and probabilities
                    prediction_index = model.predict(features_scaled)
                    predicted_emotion = encoder.inverse_transform(prediction_index)[0]
                    probabilities = model.predict_proba(features_scaled)[0]

                    # 4. Display the results
                    st.success("Analysis Complete!")
                    st.markdown(f"### 🎭 Detected Emotion: **{predicted_emotion.capitalize()}**")

                    # Create a dictionary of emotions and their probabilities for the bar chart
                    prob_dict = {emotion: prob for emotion, prob in zip(encoder.classes_, probabilities)}
                    st.bar_chart(prob_dict)

            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
            finally:
                # Clean up the temporary file
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
else:
    st.warning("Application is not ready. Model assets could not be loaded.")
