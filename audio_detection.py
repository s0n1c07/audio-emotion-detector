import streamlit as st
import torch
import io
from scipy.io import wavfile
import numpy as np
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# Load model
@st.cache_resource
def load_model():
    extractor = AutoFeatureExtractor.from_pretrained("Hatman/audio-emotion-detection")
    model = AutoModelForAudioClassification.from_pretrained("Hatman/audio-emotion-detection")
    model.eval().to("cuda" if torch.cuda.is_available() else "cpu")
    return extractor, model, model.config.id2label

extractor, model, id2label = load_model()

st.title("🎙️ Audio Emotion Detection")
st.write("Upload a `.wav` file to classify the emotion.")

uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file)

    # Read audio from memory using scipy
    sr, waveform_np = wavfile.read(io.BytesIO(uploaded_file.read()))

    # Normalize and convert to torch tensor
    if waveform_np.dtype != np.float32:
        waveform_np = waveform_np.astype(np.float32) / np.iinfo(waveform_np.dtype).max

    waveform = torch.tensor(waveform_np)

    # Convert to mono if stereo
    if len(waveform.shape) > 1:
        waveform = waveform.mean(dim=-1)

    # Resample to 16000 if needed
    if sr != 16000:
        import torchaudio
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform.unsqueeze(0)).squeeze(0)

    # Extract features and run inference
    with st.spinner("🔍 Analyzing emotion..."):
    # Extract features and run inference
        inputs = extractor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits
            pred = torch.argmax(logits, dim=-1).item()
            predicted_emotion = id2label[pred]

    st.success(f"🎯 **Predicted Emotion:** {predicted_emotion}")

