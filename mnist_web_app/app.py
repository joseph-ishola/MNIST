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

# Load the pre-trained model
model = tf.keras.models.load_model('models/exponential_cnn_model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image data from the request
    data = request.json
    image_data = data['image']
    
    # Process the base64 image data
    # Remove the data URL prefix
    image_data = re.sub('^data:image/.+;base64,', '', image_data)

    # Save the original image to disk before any processing
    original_image_path = 'original_uploaded_image.png'
    with open(original_image_path, 'wb') as f:
        f.write(base64.b64decode(image_data))
    
    # Decode the base64 data
    image_bytes = base64.b64decode(image_data)
    
    # Convert to PIL Image
    pil_image = Image.open(io.BytesIO(image_bytes)).convert('L')

    # Preprocess with OpenCV
    processed_image = preprocess_image_opencv(pil_image)

    # Save preprocessed image for comparison/debugging
    cv2.imwrite('debug_processed.png', (processed_image * 255).astype(np.uint8))

    # Model expects 4D input
    image_array = processed_image.reshape(1, 28, 28, 1)

    prediction = model.predict(image_array)
    digit = int(np.argmax(prediction))
    confidence = float(prediction[0][digit])

    top_3_indices = np.argsort(prediction[0])[-3:][::-1]
    top_3_predictions = [
        {"digit": int(idx), "confidence": float(prediction[0][idx])}
        for idx in top_3_indices
    ]

    # Send processed image as base64
    processed_display = (processed_image * 255).astype(np.uint8)
    _, buffer = cv2.imencode('.png', processed_display)
    processed_img_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        'digit': digit,
        'confidence': confidence,
        'top_3': top_3_predictions,
        'processed_image': processed_img_base64
    })


import scipy.ndimage

def preprocess_image_opencv(pil_image):
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