# Audio Emotion Classification

Machine learning pipeline for classifying emotions in speech audio with a Streamlit web interface.

## 🔗 Live Demo
**[Try the app here](https://audio-emotion-detector-tcvwfxrwvjlp9fcfe97iwz.streamlit.app/)**

## Features
- Real-time emotion detection from `.wav` files
- 8 emotion categories: neutral, calm, happy, sad, angry, fearful, disgust, surprised
- Interactive web interface with probability visualization
- Feature extraction using MFCC, spectral contrast, chroma, and tonnetz

## Quick Start

### Prerequisites
- Python 3.8+
- pip

### Installation
```bash
git clone https://github.com/s0n1c07/audio-emotion-detector/
cd audio-emotion-classification
pip install -r requirements.txt
```

### Run Locally
```bash
streamlit run audio_detection.py
```

## Project Structure
```
├── audio_detection.py          # Streamlit web app
├── MARS.ipynb                  # Model training notebook
├── model.pkl                   # Trained ExtraTreesClassifier
├── scaler.pkl                  # Feature scaler
├── encoder.pkl                 # Label encoder
└── requirements.txt            # Dependencies
```

## Model Pipeline

### 1. Feature Extraction
- **MFCC**: 40 coefficients + mean/std
- **Delta features**: First and second derivatives
- **Spectral contrast**: Frequency band energy differences
- **Chroma**: Pitch class profiles
- **Tonnetz**: Harmonic network features

### 2. Data Preprocessing
- Audio normalization using librosa
- Feature standardization with StandardScaler
- Class balancing with SMOTE

### 3. Model Training
- **Algorithm**: LGBClassifier
- **Dataset**: Audio_Speech_Actors and Audio_Song_Actors
- **Augmentation**: Background noise for happy/sad emotions
- **Evaluation**: Classification report + confusion matrix

## Usage

### Web App
1. Upload a `.wav` file
2. View detected emotion and confidence scores
3. See probability distribution across all emotions

### Programmatic Usage
```python
import joblib
from audio_detection import extract_features

# Load models
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")

# Predict emotion
features = extract_features("your_audio.wav")
features_scaled = scaler.transform([features])
prediction = model.predict(features_scaled)
emotion = encoder.inverse_transform(prediction)[0]
```

## Dependencies
```
streamlit
librosa
numpy
scikit-learn
joblib
```

## Supported Emotions
| Code | Emotion   | Emoji |
|------|-----------|-------|
| 01   | neutral   | 😐    |
| 02   | calm      | 😌    |
| 03   | happy     | 😊    |
| 04   | sad       | 😢    |
| 05   | angry     | 😠    |
| 06   | fearful   | 😨    |
| 07   | disgust   | 🤢    |
| 08   | surprised | 😲    |

## Technical Details
- **Audio format**: WAV files only
- **Feature vector**: 65 dimensions
- **Model accuracy**: See `MARS.ipynb` for detailed metrics
- **Processing time**: ~1-2 seconds per file

## Author
**Rajveer Singh**  
Student @ IIT Roorkee

---

For detailed implementation and training process, see `audio_classification.ipynb`.
