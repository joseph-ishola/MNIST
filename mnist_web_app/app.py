import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from PIL import Image
import io
import base64
import re
import cv2
import json
import glob
import scipy.ndimage

app = Flask(__name__)

# Dictionary to store loaded models
models = {}

def load_models():
    """Load all available models from the models directory"""
    model_files = glob.glob('models/*.h5')
    
    if not model_files:
        print("Warning: No model files found in models directory")
        # Create a placeholder model
        placeholder = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        placeholder.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        models['placeholder_model'] = {
            'model': placeholder,
            'display_name': 'Demo Model',
            'description': 'A placeholder model for demonstration purposes'
        }
        return
    
    # Load all .h5 models from the models directory
    for model_path in model_files:
        model_name = os.path.basename(model_path).replace('.h5', '')
        
        # Load model metadata if available
        metadata_path = os.path.join('models', f"{model_name}.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                display_name = metadata.get('display_name', model_name)
                description = metadata.get('description', f'Model: {model_name}')
        else:
            display_name = model_name
            description = f'Model: {model_name}'
        
        # Load the model
        try:
            model = tf.keras.models.load_model(model_path)
            models[model_name] = {
                'model': model,
                'display_name': display_name,
                'description': description
            }
            print(f"Loaded model: {model_name}")
        except Exception as e:
            print(f"Error loading model {model_name}: {str(e)}")

# Load models at startup
load_models()

@app.route('/')
def home():
    # Pass model information to the template
    model_info = {name: {
        'display_name': info['display_name'], 
        'description': info['description']
    } for name, info in models.items()}
    
    return render_template('index.html', models=model_info)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the request
        data = request.json
        image_data = data['image']
        model_name = data.get('model', list(models.keys())[0])  # Default to first model if not specified
        
        # Check if model exists
        if model_name not in models:
            return jsonify({
                'error': f'Model {model_name} not found'
            }), 404
        
        # Get the selected model
        model = models[model_name]['model']
        
        # Process the base64 image data
        # Remove the data URL prefix
        image_data = re.sub('^data:image/.+;base64,', '', image_data)
        
        # Decode the base64 data
        image_bytes = base64.b64decode(image_data)
        
        # Convert to PIL Image
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('L')
        
        # Preprocess with your successful pipeline
        processed_image = preprocess_image_opencv(pil_image)
        
        # For debugging: Save the processed image
        # A unique filename based on the model and timestamp could be useful
        #debug_filename = f'debug_processed_{model_name}.png'
        #cv2.imwrite(debug_filename, (processed_image * 255).astype(np.uint8))
        
        # Model expects 4D input
        image_array = processed_image.reshape(1, 28, 28, 1)
        
        # Make prediction with the selected model
        prediction = model.predict(image_array)
        digit = int(np.argmax(prediction))
        confidence = float(prediction[0][digit])
        
        # Get top 3 predictions
        top_3_indices = np.argsort(prediction[0])[-3:][::-1]
        top_3_predictions = [
            {"digit": int(idx), "confidence": float(prediction[0][idx])}
            for idx in top_3_indices
        ]
        
        # Convert the processed image back to base64 for sending to frontend
        processed_display = (processed_image * 255).astype(np.uint8)
        _, buffer = cv2.imencode('.png', processed_display)
        processed_img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Return the prediction results including model info
        return jsonify({
            'model': model_name,
            'model_display_name': models[model_name]['display_name'],
            'digit': digit,
            'confidence': confidence,
            'top_3': top_3_predictions,
            'processed_image': processed_img_base64
        })
    except Exception as e:
        print(f"Error in predict route: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/list_models', methods=['GET'])
def list_models():
    # Return a list of available models
    model_info = {name: {
        'display_name': info['display_name'], 
        'description': info['description']
    } for name, info in models.items()}
    
    return jsonify(model_info)

@app.route('/reload_models', methods=['POST'])
def reload_models():
    # Admin route to reload models without restarting the app
    try:
        models.clear()
        load_models()
        return jsonify({
            'success': True,
            'message': f'Reloaded {len(models)} models',
            'models': list(models.keys())
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def preprocess_image_opencv(pil_image):
    """
    Preprocess a handwritten digit image to match MNIST format using your successful pipeline.
    """
    # Convert PIL to OpenCV (RGB → BGR → Grayscale)
    img = cv2.cvtColor(np.array(pil_image.convert('RGB')), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur and threshold (no inversion)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Find contours to crop the digit
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        digit = thresh[y:y+h, x:x+w]
    else:
        digit = thresh

    # Resize to 20x20 box
    h, w = digit.shape
    if h > 0 and w > 0:
        scaling = 20.0 / max(h, w)
        digit = cv2.resize(digit, (int(w * scaling), int(h * scaling)), interpolation=cv2.INTER_AREA)

    # Pad to 28x28 with digit centered by center of mass
    rows, cols = digit.shape
    padded = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - cols) // 2
    y_offset = (28 - rows) // 2
    padded[y_offset:y_offset + rows, x_offset:x_offset + cols] = digit

    # Compute center of mass and shift accordingly
    cy, cx = scipy.ndimage.center_of_mass(padded)
    shiftx = int(np.round(14 - cx))
    shifty = int(np.round(14 - cy))
    padded = scipy.ndimage.shift(padded, shift=[shifty, shiftx], mode='constant')

    # Normalize and return
    final = padded.astype('float32') / 255.0
    return final

if __name__ == '__main__':
    app.run(debug=True)