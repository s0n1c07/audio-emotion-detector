import streamlit as st
import torch
import torchaudio
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# Load model and feature extractor
@st.cache_resource
def load_model():
    extractor = AutoFeatureExtractor.from_pretrained("Hatman/audio-emotion-detection")
    model = AutoModelForAudioClassification.from_pretrained("Hatman/audio-emotion-detection")
    model.eval().to("cuda" if torch.cuda.is_available() else "cpu")
    return extractor, model, model.config.id2label

extractor, model, id2label = load_model()

st.title("🎙️ Audio Emotion Detection")
st.write("Upload a `.wav` file and get the predicted emotion.")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav"])

if uploaded_file is not None:
    # Save and load audio
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())
    waveform, sr = torchaudio.load("temp.wav")

    # Resample if needed
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)
    waveform = waveform.squeeze()

    # Extract features and predict
    inputs = extractor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        pred = torch.argmax(logits, dim=-1).item()
        predicted_label = id2label[pred]

    # Display result
    st.audio("temp.wav")
    st.success(f"**Predicted Emotion:** {predicted_label}")
