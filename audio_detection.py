import streamlit as st
import numpy as np
import librosa
import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE

# -----------------------
# Emotion Mapping
# -----------------------
emotion_map = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"
}
augment_classes = {"happy", "sad"}

# -----------------------
# Feature Extraction
# -----------------------
def extract_features_from_signal(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean, mfcc_std = np.mean(mfcc, axis=1), np.std(mfcc, axis=1)
    delta1 = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    spec_con = librosa.feature.spectral_contrast(y=y, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
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

# -----------------------
# Load and Prepare Data
# -----------------------
@st.cache_resource
def train_model():
    data, labels = [], []
    folders = ["Audio_Speech_Actors", "Audio_Song_Actors"]  # These should be in your working dir

    for folder in folders:
        if not os.path.exists(folder): continue
        for sub in os.listdir(folder):
            sub_path = os.path.join(folder, sub)
            if not os.path.isdir(sub_path): continue
            for fname in os.listdir(sub_path):
                if not fname.endswith(".wav"): continue
                emo_code = fname.split("-")[2]
                emo = emotion_map.get(emo_code)
                if emo is None: continue
                full_path = os.path.join(sub_path, fname)
                feat = extract_features(full_path)
                data.append(feat)
                labels.append(emo)

                if emo in augment_classes:
                    y, sr = librosa.load(full_path, sr=None, res_type='kaiser_fast')
                    noise = np.random.randn(len(y))
                    y_noisy = y + 0.003 * noise
                    data.append(extract_features_from_signal(y_noisy, sr))
                    labels.append(emo)

    df = pd.DataFrame(data)
    df['emotion'] = labels

    X = df.drop('emotion', axis=1).values
    encoder = LabelEncoder()
    y = encoder.fit_transform(df['emotion'])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_scaled, y)

    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, stratify=y_res, random_state=42)

    model = ExtraTreesClassifier(class_weight='balanced', n_estimators=600, max_depth=25,
                                 min_samples_split=3, max_features=0.8, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return model, scaler, encoder, X_test, y_test, y_pred

# -----------------------
# Streamlit UI
# -----------------------
st.title("🎵 Audio Emotion Detection")
st.markdown("Upload a `.wav` file to detect its emotion based on voice features.")

# Train model (cached for performance)
model, scaler, encoder, X_test, y_test, y_pred = train_model()

# Confusion Matrix
st.subheader("Model Evaluation")
cm = confusion_matrix(encoder.inverse_transform(y_test), encoder.inverse_transform(y_pred), labels=encoder.classes_)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

fig, ax = plt.subplots()
disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=encoder.classes_)
disp.plot(cmap='Blues', xticks_rotation=45, ax=ax)
st.pyplot(fig)

# Upload section
st.subheader("🎤 Predict Emotion from Audio")
uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file, format='audio/wav')
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())
    try:
        feat = extract_features("temp.wav")
        feat_scaled = scaler.transform([feat])
        pred = model.predict(feat_scaled)
        emotion_label = encoder.inverse_transform(pred)[0]
        st.success(f"Predicted Emotion: **{emotion_label}** 🎯")
    except Exception as e:
        st.error(f"Error processing the file: {e}")
