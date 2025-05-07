import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from PIL import Image
import io
import base64
import re
import cv2

app = Flask(__name__)

# Create the models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Load the pre-trained model
# Note: You'll need to place your trained model in the models directory
model_path = 'models/best_mnist_model.h5'
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
else:
    print(f"Warning: Model file not found at {model_path}")
    # Create a placeholder model for demonstration if the real model isn't available
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print("Using a placeholder model for demonstration. Replace with your trained model.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image data from the request
        data = request.json
        image_data = data['image']
        
        # Process the base64 image data
        # Remove the data URL prefix
        image_data = re.sub('^data:image/.+;base64,', '', image_data)
        
        # Decode the base64 data
        image_bytes = base64.b64decode(image_data)
        
        # Convert to OpenCV format
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Apply preprocessing pipeline
        processed_img = preprocess_handwritten_digit(img)
        
        # Reshape for the model
        processed_img_for_model = processed_img.reshape(1, 28, 28, 1)
        
        # Make prediction
        prediction = model.predict(processed_img_for_model)
        digit = np.argmax(prediction)
        confidence = float(prediction[0][digit])
        
        # Get top 3 predictions
        top_3_indices = np.argsort(prediction[0])[-3:][::-1]
        top_3_predictions = [
            {"digit": int(idx), "confidence": float(prediction[0][idx])}
            for idx in top_3_indices
        ]
        
        # Convert the processed image back to base64 for sending to frontend
        processed_img_2d = processed_img.copy() * 255  # Scale back to 0-255 range
        processed_img_2d = processed_img_2d.astype(np.uint8)
        _, buffer = cv2.imencode('.png', processed_img_2d)
        processed_img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Return the prediction and processed image
        return jsonify({
            'digit': int(digit),
            'confidence': confidence,
            'top_3': top_3_predictions,
            'processed_image': processed_img_base64
        })
    except Exception as e:
        print(f"Error in predict route: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500

def preprocess_handwritten_digit(img):
    """
    Preprocess a handwritten digit image to match MNIST format:
    - Convert to grayscale
    - Apply Gaussian blur to reduce noise
    - Threshold to get binary image
    - Center the digit using contours
    - Resize to 28x28 pixels
    - Normalize pixel values to range [0,1]
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply thresholding to get a binary image (invert to match MNIST format - white digit on black background)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours to center the digit
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour (should be the digit)
        cnt = max(contours, key=cv2.contourArea)
        
        # Get bounding box for the digit
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Create a square bounding box (to maintain aspect ratio)
        size = max(w, h)
        
        # Add some padding
        padding = int(size * 0.2)
        size += padding * 2
        
        # Calculate the new center
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Define the new square bounding box
        x1 = max(0, center_x - size // 2)
        y1 = max(0, center_y - size // 2)
        x2 = min(thresh.shape[1], x1 + size)
        y2 = min(thresh.shape[0], y1 + size)
        
        # Extract the digit
        digit = thresh[y1:y2, x1:x2]
        
        # Resize to 28x28 pixels
        digit = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_AREA)
    else:
        # If no contours found, just resize the whole image
        digit = cv2.resize(thresh, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Normalize pixel values to [0,1]
    digit = digit.astype('float32') / 255.0
    
    return digit

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)