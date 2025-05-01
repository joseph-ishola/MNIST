"""
MNIST CNN Module - A modular implementation of CNN for MNIST digit classification.

This module provides functions for:
1. Data preprocessing of MNIST dataset
2. Creating and training CNN models
3. Processing and predicting on handwritten digit images
4. Evaluating model performance
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow_datasets as tfds
import cv2
import os
import random
from typing import Tuple, List, Dict, Any, Union, Optional

# Class for tracking metrics during training
class MetricsHistory(tf.keras.callbacks.Callback):
    """Custom callback to track and store training metrics history."""
    
    def on_train_begin(self, logs=None):
        self.losses = []
        self.val_losses = []
        self.accuracy = []
        self.val_accuracy = []
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.accuracy.append(logs.get('accuracy'))
        self.val_accuracy.append(logs.get('val_accuracy'))


def preprocess_mnist(buffer_size: int = 10000, batch_size: int = 100) -> Tuple:
    """
    Preprocess the MNIST dataset for training and evaluation.
    
    Parameters:
    -----------
    buffer_size : int
        Size of the buffer for shuffling data
    batch_size : int
        Batch size for training
        
    Returns:
    --------
    tuple
        (train_data, validation_data, test_data) - Preprocessed TensorFlow datasets
    """
    # Load the data from tfds
    mnist_datasets, mnist_info = tfds.load(name='mnist', with_info=True, as_supervised=True)
    
    # Extract training and testing datasets
    mnist_train, mnist_test = mnist_datasets['train'], mnist_datasets['test']
    
    # Define validation samples (10% of training data)
    num_validation_samples = 0.1 * mnist_info.splits["train"].num_examples
    num_validation_samples = tf.cast(num_validation_samples, tf.int64)
    
    # Get test samples count
    num_test_samples = tf.cast(mnist_info.splits['test'].num_examples, tf.int64)
    
    # Define scaling function to normalize pixel values to [0,1]
    scale = lambda x, y: (tf.cast(x, tf.float32) / 255., y)
    
    # Apply scaling to train and test data
    scaled_train_and_validation_data = mnist_train.map(scale)
    scaled_test_data = mnist_test.map(scale)
    
    # Shuffle the training data
    shuffled_train_and_validation_data = scaled_train_and_validation_data.shuffle(
        buffer_size=buffer_size, seed=42)
    
    # Split into validation and training data
    validation_data = shuffled_train_and_validation_data.take(num_validation_samples)
    train_data = shuffled_train_and_validation_data.skip(num_validation_samples)
    
    # Batch the data
    train_data = train_data.batch(batch_size)
    validation_data = validation_data.batch(num_validation_samples)
    test_data = scaled_test_data.batch(num_test_samples)
    
    return train_data, validation_data, test_data


def create_cnn_model() -> keras.Sequential:
    """
    Create a CNN model for MNIST digit classification.
    
    Returns:
    --------
    keras.Sequential
        Compiled CNN model ready for training
    """
    model = keras.Sequential([
        # First Convolutional Block
        keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', 
                           input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Second Convolutional Block
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Third Convolutional Block
        keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Flatten the output to feed into dense layers
        keras.layers.Flatten(),
        
        # Fully connected layers
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),  # Dropout for regularization
        
        # Output layer
        keras.layers.Dense(10, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_model(model: keras.Sequential, 
                train_data: tf.data.Dataset, 
                validation_data: tf.data.Dataset,
                epochs: int = 20, 
                model_path: str = 'mnist_cnn_model.h5',
                patience: int = 5,
                verbose: int = 1) -> Tuple[keras.Sequential, MetricsHistory]:
    """
    Train the CNN model on MNIST data.
    
    Parameters:
    -----------
    model : keras.Sequential
        The model to train
    train_data : tf.data.Dataset
        Training dataset
    validation_data : tf.data.Dataset
        Validation dataset (can be a tuple of (inputs, targets) or a dataset)
    epochs : int
        Number of epochs to train
    model_path : str
        Path to save the best model
    patience : int
        Number of epochs with no improvement after which training will be stopped
    verbose : int
        Verbosity mode (0, 1, or 2)
        
    Returns:
    --------
    tuple
        (trained_model, metrics_history)
    """
    # Create an instance of the callback
    metrics_history = MetricsHistory()
    
    # Adding model checkpoint to save the best model
    checkpoint = ModelCheckpoint(
        model_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=verbose
    )
    
    # Adding early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=verbose
    )
    
    # Extract validation inputs and targets if it's a dataset
    if isinstance(validation_data, tf.data.Dataset):
        validation_inputs, validation_targets = next(iter(validation_data))
        validation_data = (validation_inputs, validation_targets)
    
    # Train the model
    model.fit(
        train_data,
        epochs=epochs,
        validation_data=validation_data,
        callbacks=[metrics_history, checkpoint, early_stopping],
        verbose=verbose
    )
    
    return model, metrics_history


def evaluate_model(model: keras.Sequential, 
                   test_data: tf.data.Dataset,
                   verbose: int = 1) -> Tuple[float, float]:
    """
    Evaluate the trained model on test data.
    
    Parameters:
    -----------
    model : keras.Sequential
        Trained model to evaluate
    test_data : tf.data.Dataset
        Test dataset
    verbose : int
        Verbosity mode
        
    Returns:
    --------
    tuple
        (test_loss, test_accuracy)
    """
    test_loss, test_accuracy = model.evaluate(test_data, verbose=verbose)
    print(f'Test loss: {test_loss:.4f}. Test accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)')
    return test_loss, test_accuracy


def plot_metrics(history: MetricsHistory) -> None:
    """
    Plot training and validation metrics (loss and accuracy).
    
    Parameters:
    -----------
    history : MetricsHistory
        Object containing training history
    """
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plotting the loss metrics on the first subplot
    ax1.plot(history.losses, label='Training Loss')
    ax1.plot(history.val_losses, label='Validation Loss')
    ax1.set_title('CNN Model Loss Over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plotting the accuracy metrics on the second subplot
    ax2.plot(history.accuracy, label='Training Accuracy')
    ax2.plot(history.val_accuracy, label='Validation Accuracy')
    ax2.set_title('CNN Model Accuracy Over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    # Display the plots
    plt.tight_layout()
    plt.show()


def preprocess_handwritten_digit(image_path: str, 
                                 display: bool = True) -> np.ndarray:
    """
    Preprocess a handwritten digit image to match MNIST format.
    
    Parameters:
    -----------
    image_path : str
        Path to the handwritten digit image
    display : bool
        Whether to display the preprocessed image
        
    Returns:
    --------
    np.ndarray
        Preprocessed image in format ready for model prediction
    """
    # Read the image
    img = cv2.imread(image_path)
    
    # Handle case where image couldn't be read
    if img is None:
        raise ValueError(f"Could not read image from {image_path}. Please check the file path.")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply thresholding to get a binary image
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
    
    # Display the processed image if requested
    if display:
        plt.figure(figsize=(5, 5))
        plt.imshow(digit, cmap='gray')
        plt.title("Processed Handwritten Digit")
        plt.axis('off')
        plt.show()
    
    return digit


def predict_digit(model: keras.Sequential, 
                  preprocessed_digit: np.ndarray,
                  display: bool = True) -> Tuple[int, float, np.ndarray]:
    """
    Make a prediction using the model on a preprocessed digit.
    
    Parameters:
    -----------
    model : keras.Sequential
        Trained model for prediction
    preprocessed_digit : np.ndarray
        Preprocessed digit image
    display : bool
        Whether to display prediction probabilities
        
    Returns:
    --------
    tuple
        (predicted_digit, confidence, prediction_array)
    """
    # Reshape to match model input shape: (1, 28, 28, 1)
    sample = preprocessed_digit.reshape(1, 28, 28, 1)
    
    # Get prediction
    prediction = model.predict(sample)[0]
    predicted_digit = np.argmax(prediction)
    confidence = prediction[predicted_digit] * 100
    
    # Display the prediction
    print(f"Predicted digit: {predicted_digit}")
    print(f"Confidence: {confidence:.2f}%")
    
    # Plot the prediction probabilities if requested
    if display:
        plt.figure(figsize=(10, 4))
        plt.bar(range(10), prediction)
        plt.xticks(range(10))
        plt.xlabel('Digit')
        plt.ylabel('Probability')
        plt.title(f'Prediction probabilities (Predicted: {predicted_digit}, Confidence: {confidence:.2f}%)')
        plt.ylim(0, 1)
        plt.show()
    
    return predicted_digit, confidence, prediction


def batch_predict_digits(model: keras.Sequential, 
                         image_paths: List[str],
                         display: bool = True) -> List[Dict[str, Any]]:
    """
    Process and predict multiple handwritten digit images.
    
    Parameters:
    -----------
    model : keras.Sequential
        Trained model for prediction
    image_paths : List[str]
        List of paths to handwritten digit images
    display : bool
        Whether to display processed images and predictions
        
    Returns:
    --------
    List[Dict[str, Any]]
        List of dictionaries containing prediction results for each image
    """
    results = []
    
    for i, image_path in enumerate(image_paths):
        try:
            print(f"\nProcessing image {i+1}/{len(image_paths)}: {image_path}")
            
            # Preprocess the image
            processed_digit = preprocess_handwritten_digit(image_path, display=display)
            
            # Make prediction
            predicted_digit, confidence, prediction_array = predict_digit(
                model, processed_digit, display=display)
            
            # Store results
            results.append({
                'image_path': image_path,
                'predicted_digit': int(predicted_digit),
                'confidence': float(confidence),
                'prediction_array': prediction_array.tolist()
            })
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            results.append({
                'image_path': image_path,
                'error': str(e)
            })
    
    return results


def load_model(model_path: str) -> keras.Sequential:
    """
    Load a saved model from disk.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model
        
    Returns:
    --------
    keras.Sequential
        Loaded model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    model = keras.models.load_model(model_path)
    print(f"Model loaded successfully from {model_path}")
    return model


def sample_and_predict_test_data(model: keras.Sequential, 
                                test_data: tf.data.Dataset,
                                num_samples: int = 10) -> None:
    """
    Sample random images from test data and make predictions.
    
    Parameters:
    -----------
    model : keras.Sequential
        Trained model for prediction
    test_data : tf.data.Dataset
        Test dataset
    num_samples : int
        Number of random samples to select
    """
    # Get the test dataset as NumPy arrays
    test_images = []
    test_labels = []

    # Take the first batch from test_data
    for images, labels in test_data.take(1):
        test_images = images.numpy()
        test_labels = labels.numpy()

    # Select a random subset of images
    sample_indices = random.sample(range(len(test_images)), num_samples)
    sample_images = test_images[sample_indices]
    sample_labels = test_labels[sample_indices]

    # Make predictions on the sample images
    predictions = model.predict(sample_images)
    predicted_labels = np.argmax(predictions, axis=1)

    # Plot the sample images with their true and predicted labels
    plt.figure(figsize=(15, 8))
    for i in range(num_samples):
        plt.subplot(2, 5, i+1)
        
        # Reshape and display the image (removing the channel dimension)
        plt.imshow(sample_images[i].reshape(28, 28), cmap='gray')
        
        # Display true and predicted labels
        title_color = 'green' if predicted_labels[i] == sample_labels[i] else 'red'
        plt.title(f"True: {sample_labels[i]}\nPred: {predicted_labels[i]}", 
                color=title_color)
        plt.axis('off')

    plt.tight_layout()
    plt.suptitle("Sample Test Images with Predictions", fontsize=16)
    plt.subplots_adjust(top=0.88)
    plt.show()

    # Calculate accuracy on the sample
    sample_accuracy = np.mean(predicted_labels == sample_labels) * 100
    print(f"Accuracy on the {num_samples} sample images: {sample_accuracy:.2f}%")

    # Print detailed results for each sample
    print("\nDetailed prediction results:")
    print("------------------------")
    for i in range(num_samples):
        confidence = predictions[i][predicted_labels[i]] * 100
        result = "✓" if predicted_labels[i] == sample_labels[i] else "✗"
        print(f"Sample {i+1}: True={sample_labels[i]}, Predicted={predicted_labels[i]}, Confidence={confidence:.2f}%, {result}")