from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import os
import numpy as np
import librosa
import joblib
import tempfile
import traceback
from werkzeug.utils import secure_filename
import requests
from io import BytesIO

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'temp_uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'flac', 'ogg'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables for models
model = None
scaler = None
label_encoder = None
model_loaded = False

# GitHub URLs for your models (replace with your actual GitHub URLs)
MODEL_URLS = {
    'model': 'https://github.com/s0n1c07/audio-emotion-detector/main/emotion.pkl',
    'scaler': 'https://github.com/s0n1c07/audio-emotion-detector/main/scaler.pkl',
    'encoder': 'https://github.com/s0n1c07/audio-emotion-detector/main/label.pkl'
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def download_model_from_github(url):
    """Download model file from GitHub"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return BytesIO(response.content)
    except Exception as e:
        print(f"Error downloading model from {url}: {e}")
        return None

def load_models():
    """Load models from GitHub or local files"""
    global model, scaler, label_encoder, model_loaded
    
    try:
        # Try to load from local files first
        if (os.path.exists('emotion_model_7class.pkl') and 
            os.path.exists('scaler_7class.pkl') and 
            os.path.exists('label_encoder_7class.pkl')):
            
            print("Loading models from local files...")
            model = joblib.load('emotion_model_7class.pkl')
            scaler = joblib.load('scaler_7class.pkl')
            label_encoder = joblib.load('label_encoder_7class.pkl')
            
        else:
            print("Loading models from GitHub...")
            # Load from GitHub
            model_file = download_model_from_github(MODEL_URLS['model'])
            scaler_file = download_model_from_github(MODEL_URLS['scaler'])
            encoder_file = download_model_from_github(MODEL_URLS['encoder'])
            
            if model_file and scaler_file and encoder_file:
                model = joblib.load(model_file)
                scaler = joblib.load(scaler_file)
                label_encoder = joblib.load(encoder_file)
            else:
                raise Exception("Failed to download models from GitHub")
        
        model_loaded = True
        print("Models loaded successfully!")
        print(f"Emotion classes: {label_encoder.classes_}")
        
    except Exception as e:
        print(f"Error loading models: {e}")
        model_loaded = False

def extract_features(file_path):
    """Extract MFCC features from audio file"""
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=None, res_type='kaiser_fast')
        
        # Extract MFCC features (matching your training code)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=50)
        
        # Return mean of MFCCs
        return np.mean(mfcc.T, axis=0)
        
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

@app.route('/')
def index():
    """Serve the main HTML page"""
    html_template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MARS - Emotion Recognition System</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }

        .container {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            max-width: 800px;
            width: 100%;
            text-align: center;
        }

        .header h1 {
            color: #4a5568;
            font-size: 2.5rem;
            margin-bottom: 10px;
            background: linear-gradient(45deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .upload-area {
            border: 3px dashed #cbd5e0;
            border-radius: 15px;
            padding: 40px;
            margin: 30px 0;
            transition: all 0.3s ease;
            cursor: pointer;
            background: linear-gradient(45deg, rgba(102, 126, 234, 0.05), rgba(118, 75, 162, 0.05));
        }

        .upload-area:hover {
            border-color: #667eea;
            background: linear-gradient(45deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
            transform: translateY(-2px);
        }

        .btn {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 25px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            margin: 10px;
        }

        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
        }

        .btn:disabled {
            background: #cbd5e0;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }

        .loading {
            display: none;
            margin: 20px 0;
        }

        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .result {
            display: none;
            margin-top: 30px;
            padding: 25px;
            background: linear-gradient(45deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
            border-radius: 15px;
        }

        .emotion-display {
            font-size: 2rem;
            font-weight: bold;
            margin-bottom: 15px;
            color: #4a5568;
        }

        .confidence-bar {
            background: #f1f5f9;
            border-radius: 10px;
            height: 20px;
            margin: 10px 0;
            overflow: hidden;
        }

        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            border-radius: 10px;
            transition: width 0.5s ease;
        }

        .emotions-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }

        .emotion-card {
            background: rgba(255, 255, 255, 0.8);
            padding: 15px;
            border-radius: 10px;
        }

        .error {
            color: #e53e3e;
            background: rgba(229, 62, 62, 0.1);
            padding: 15px;
            border-radius: 10px;
            margin: 15px 0;
            display: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>MARS</h1>
            <p>Multimodal Audio Recognition System</p>
        </div>

        <div class="upload-area" onclick="document.getElementById('audioFile').click()">
            <div style="font-size: 3rem; margin-bottom: 15px;">🎵</div>
            <div>Click to upload your audio file</div>
            <div style="font-size: 0.9rem; color: #718096;">WAV, MP3, M4A (Max: 10MB)</div>
        </div>

        <input type="file" id="audioFile" accept="audio/*" style="display: none;" />

        <button class="btn" id="analyzeBtn" onclick="analyzeEmotion()" disabled>
            🧠 Analyze Emotion
        </button>

        <div class="loading" id="loading">
            <div class="spinner"></div>
            <div>Processing audio...</div>
        </div>

        <div class="result" id="result">
            <div class="emotion-display" id="emotionResult"></div>
            <div class="confidence-bar">
                <div class="confidence-fill" id="confidenceFill"></div>
            </div>
            <div id="confidenceText"></div>
            <div class="emotions-grid" id="emotionsGrid"></div>
        </div>

        <div class="error" id="error"></div>
    </div>

    <script>
        let currentFile = null;

        const emotionEmojis = {
            'angry': '😠', 'calm': '😌', 'fearful': '😨', 'happy': '😊',
            'neutral': '😐', 'sad': '😢', 'surprised': '😲'
        };

        document.getElementById('audioFile').addEventListener('change', function(e) {
            if (e.target.files.length > 0) {
                currentFile = e.target.files[0];
                document.getElementById('analyzeBtn').disabled = false;
                hideError();
            }
        });

        async function analyzeEmotion() {
            if (!currentFile) {
                showError('Please select an audio file first.');
                return;
            }

            const formData = new FormData();
            formData.append('audio', currentFile);

            showLoading(true);
            hideError();

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (data.success) {
                    displayResults(data.prediction, data.probabilities);
                } else {
                    showError(data.error || 'Error analyzing audio');
                }
            } catch (error) {
                showError('Network error: ' + error.message);
            } finally {
                showLoading(false);
            }
        }

        function displayResults(prediction, probabilities) {
            const emoji = emotionEmojis[prediction] || '🤔';
            const confidence = probabilities[prediction];

            document.getElementById('emotionResult').innerHTML = `${emoji} ${prediction.toUpperCase()}`;
            document.getElementById('confidenceFill').style.width = `${confidence * 100}%`;
            document.getElementById('confidenceText').textContent = `Confidence: ${(confidence * 100).toFixed(1)}%`;

            const grid = document.getElementById('emotionsGrid');
            grid.innerHTML = '';
            
            Object.entries(probabilities)
                .sort(([,a], [,b]) => b - a)
                .forEach(([emotion, prob]) => {
                    const card = document.createElement('div');
                    card.className = 'emotion-card';
                    card.innerHTML = `
                        <div><strong>${emotionEmojis[emotion]} ${emotion}</strong></div>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${prob * 100}%"></div>
                        </div>
                        <div>${(prob * 100).toFixed(1)}%</div>
                    `;
                    grid.appendChild(card);
                });

            document.getElementById('result').style.display = 'block';
        }

        function showLoading(show) {
            document.getElementById('loading').style.display = show ? 'block' : 'none';
            document.getElementById('analyzeBtn').disabled = show;
        }

        function showError(message) {
            const errorDiv = document.getElementById('error');
            errorDiv.textContent = message;
            errorDiv.style.display = 'block';
        }

        function hideError() {
            document.getElementById('error').style.display = 'none';
        }
    </script>
</body>
</html>
    '''
    return render_template_string(html_template)

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'message': 'MARS Emotion Recognition API is running'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict emotion from uploaded audio file"""
    try:
        # Check if models are loaded
        if not model_loaded:
            return jsonify({
                'success': False,
                'error': 'Models not loaded. Please try again later.'
            }), 500

        # Check if file is present
        if 'audio' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No audio file provided'
            }), 400

        file = request.files['audio']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400

        # Validate file
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': 'Invalid file type. Please upload WAV, MP3, M4A, FLAC, or OGG files.'
            }), 400

        # Check file size
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > MAX_FILE_SIZE:
            return jsonify({
                'success': False,
                'error': 'File too large. Maximum size is 10MB.'
            }), 400

        # Save file temporarily
        filename = secure_filename(file.filename)
        temp_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(temp_path)

        try:
            # Extract features
            features = extract_features(temp_path)
            
            if features is None:
                return jsonify({
                    'success': False,
                    'error': 'Failed to extract features from audio file'
                }), 400

            # Scale features
            features_scaled = scaler.transform([features])
            
            # Make prediction
            prediction_proba = model.predict_proba(features_scaled)[0]
            prediction_idx = np.argmax(prediction_proba)
            predicted_emotion = label_encoder.classes_[prediction_idx]
            
            # Create probabilities dictionary
            probabilities = {}
            for i, emotion in enumerate(label_encoder.classes_):
                probabilities[emotion] = float(prediction_proba[i])

            return jsonify({
                'success': True,
                'prediction': predicted_emotion,
                'probabilities': probabilities,
                'confidence': float(prediction_proba[prediction_idx])
            })

        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except Exception as e:
        print(f"Error in prediction: {e}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

@app.route('/emotions')
def get_emotions():
    """Get list of supported emotions"""
    if model_loaded and label_encoder is not None:
        return jsonify({
            'emotions': label_encoder.classes_.tolist()
        })
    else:
        return jsonify({
            'emotions': ['angry', 'calm', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
        })

if __name__ == '__main__':
    print("Starting MARS Emotion Recognition Server...")
    print("Loading models...")
    load_models()
    
    if model_loaded:
        print("✅ Models loaded successfully!")
        print("🚀 Server starting on http://localhost:5000")
    else:
        print("⚠️  Models failed to load. Server will start but predictions may not work.")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
