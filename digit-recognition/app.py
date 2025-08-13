import os
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import cv2

app = Flask(__name__)

# Get the absolute path to the model file
model_path = os.path.join(os.path.dirname(__file__), 'models/digit_recognizer.h5')

# Verify model exists before loading
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}")

# Load model with error handling
try:
    model = load_model(model_path)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
    raise

@app.route('/')
def home():
    return render_template('index.html')

def preprocess_image(image_bytes):
    """Preprocess image for digit recognition"""
    try:
        # Convert to grayscale and resize
        img = Image.open(io.BytesIO(image_bytes)).convert('L')
        img = img.resize((28, 28))
        
        # Convert to numpy array and invert (MNIST-style)
        img_array = 255 - np.array(img)
        
        # Normalize and reshape for model input
        img_array = img_array.astype('float32') / 255.0
        return img_array.reshape(1, 28, 28, 1)
    except Exception as e:
        raise ValueError(f"Image processing error: {str(e)}")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Preprocess and predict
        img_array = preprocess_image(file.read())
        prediction = model.predict(img_array)
        predicted_digit = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        
        return jsonify({
            'digit': predicted_digit,
            'confidence': confidence,
            'all_predictions': {str(i): float(score) for i, score in enumerate(prediction[0])}
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)