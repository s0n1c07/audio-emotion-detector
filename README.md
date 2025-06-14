# 🎧 Audio Emotion Classification

This repository contains:

- A machine learning pipeline for classifying emotions in speech audio using Python, `librosa`, and `scikit-learn`.
- A **Streamlit web app** that accepts a `.wav` file and predicts the emotion expressed in it.

---

## 📁 Files in the Repository

- `audio_classification.ipynb` – Notebook for data preprocessing, feature extraction, augmentation, model training & evaluation.
- `audio_detection.py` – The Streamlit application for live audio emotion prediction.
- `model.pkl`, `scaler.pkl`, `encoder.pkl` – Serialized model and preprocessing tools.
- `requirements.txt` – List of dependencies to run this project.

---

## 🚀 Streamlit App

The Streamlit app lets you:

- Upload a `.wav` file
- Automatically extract features and classify the emotion
- View the emotion prediction and probability distribution via bar chart

### 🔧 Run the App Locally

Make sure you have Python 3.8+ installed. Then:

```bash
pip install -r requirements.txt
streamlit run app.py
```
## 📊 Model Training Overview (audio_classification.ipynb)
Workflow Summary:

- Data Loading – Reads .wav files from folders like Audio_Speech_Actors and Audio_Song_Actors.

- Feature Extraction – Extracts MFCCs, delta features, spectral contrast, chroma, and tonnetz features using librosa.

- Data Augmentation – Adds background noise to selected emotions (happy, sad) to increase dataset variability.

- Model Training – Uses ExtraTreesClassifier along with SMOTE to balance classes.

- Evaluation – Outputs classification report and confusion matrix.

- Saving Model – The model, scaler, and label encoder are saved as .pkl files for inference in the web app.

## 📦 Dependencies
nginx
Copy
Edit
streamlit
librosa
scikit-learn
matplotlib
numpy
pandas
joblib
imblearn

## 🎯 Emotion Labels Supported
neutral
calm
happy
sad
angry
fearful
disgust
surprised

## 🙋‍♂️ Author
Rajveer Singh
Student @ IIT Roorkee
Project: Audio Emotion Detection with ML & Streamlit
