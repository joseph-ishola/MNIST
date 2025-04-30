# MNIST Digit Classification Using Deep Neural Networks

This project implements and evaluates a Deep Neural Network (DNN) for handwritten digit classification using the MNIST dataset.

## Project Overview

The MNIST dataset consists of 70,000 grayscale images (28×28 pixels) of handwritten digits (0-9). The goal is to build a neural network that accurately classifies these digits. This project serves as a baseline implementation before moving to more complex architectures like CNNs.

## Dataset

- **Source**: MNIST handwritten digit database
- **Features**: 28×28 pixel grayscale images (784 features when flattened)
- **Target**: Digit class (0-9)
- **Size**: 70,000 total images
  - 60,000 training images (with 6,000 used for validation)
  - 10,000 test images

## Data Preprocessing

The following preprocessing steps were applied:
- Scaling pixel values from [0-255] to [0-1] for numerical stability
- Flattening 28×28 images to 784-dimensional vectors for the DNN
- Shuffling the training data with a fixed seed (42) for reproducibility
- Batching the data with a batch size of 100 (determined through experimentation)

## Model Development

### Initial Model Architecture
- Input layer: 784 neurons (flattened 28×28 images)
- Hidden layers: 2 layers with 50 neurons each using ReLU activation
- Output layer: 10 neurons with softmax activation
- Loss function: Sparse Categorical Cross-Entropy
- Optimizer: Adam with default settings

### Hyperparameter Tuning

#### Manual Tuning
We experimented with several hyperparameters:

1. **Width of Hidden Layers**:
   - 50 neurons → val_accuracy: 0.9740
   - 100 neurons → val_accuracy: 0.9808
   - 200 neurons → val_accuracy: 0.9885

2. **Depth of Network**:
   - 2 hidden layers (baseline)
   - 3 hidden layers (200 neurons each) → val_accuracy: 0.9860
   - 4 hidden layers (200 neurons each) → val_accuracy: 0.9845
   - 4 hidden layers (300 neurons each) → val_accuracy: 0.9878
   - 5 hidden layers (300 neurons each) → val_accuracy: 0.9828

3. **Activation Functions**:
   - ReLU (baseline)
   - Sigmoid → val_accuracy: 0.9725
   - Tanh → val_accuracy: 0.9825

4. **Batch Size**:
   - 100 (baseline)
   - 1000 → val_accuracy: 0.9685
   - 10000 → val_accuracy: 0.9053
   - 1 → val_accuracy: 0.9433 (with much longer training time)

5. **Learning Rate**:
   - Adam default (baseline)
   - 0.0001 → val_accuracy: 0.9608 (slower convergence)
   - 0.02 → similar performance to baseline

Based on manual tuning, the best configuration was:
- 2 hidden layers with 200 neurons each
- ReLU activation
- Batch size of 100
- Adam optimizer with default learning rate

#### Automated Random Search
We used Keras Tuner to perform random search across:
- Number of layers (1-6)
- Number of neurons per layer (32-512)
- Activation functions (ReLU, sigmoid, tanh)
- Learning rate (10^-4 to 10^-2)

The best hyperparameters found through random search were:
- 1 hidden layer with 400 neurons
- Sigmoid activation
- Learning rate of approximately 0.002

## Model Results

### Base Model (Initial Configuration)
- Training accuracy: 96.15%
- Validation accuracy: 98.40%

### Manually Tuned Model
- Training accuracy: 98.04% 
- Validation accuracy: 99.85%

### Best Model (Random Search)
- Training accuracy: 99.96%
- Validation accuracy: 100.00%
- Test accuracy: 98.36%
- Total parameters: 318,010

## Model Testing and Evaluation

### MNIST Test Set
The model achieved 98.36% accuracy on the test set, which is strong performance for a DNN model on this dataset.

### Random Samples
Testing on 10 random samples from the test set showed:
- 100% accuracy
- High confidence (nearly all predictions at 100% confidence)

### Custom Handwritten Digits
Testing on custom handwritten digits showed mixed results:
- Handwritten '2': Correctly classified with 99.87% confidence
- Handwritten '7': Misclassified as '1' with 64.07% confidence

This highlights the model's limitations when dealing with inputs that differ from the training distribution.

## Limitations and Future Directions

### Limitations of the DNN Approach
- Lacks spatial understanding of the image structure
- Sensitive to variations in positioning, orientation, and style
- Requires more parameters than CNNs for comparable performance
- Less robust to real-world handwritten digits

### Future Improvements
1. Implement Convolutional Neural Networks (CNNs) to capture spatial relationships
2. Apply data augmentation techniques to improve robustness
3. Explore regularization methods to enhance generalization
4. Implement early stopping to prevent overfitting
5. Test with more diverse handwritten samples

## Usage

### Dependencies
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- Keras Tuner (for hyperparameter optimization)
- OpenCV (for custom image preprocessing)

### Running the Code
1. Ensure all dependencies are installed
2. Run the Jupyter notebook to train and evaluate the model
3. Use the provided functions to test on custom handwritten digits

```python
# Example of loading a saved model
from tensorflow import keras
model = keras.models.load_model('mnist_dnn_model.h5')

# Example of testing on a custom image
processed_digit = preprocess_handwritten_digit("your_digit.jpg")
predicted_digit, confidence, _ = predict_digit(model, processed_digit)
print(f"Predicted digit: {predicted_digit}, Confidence: {confidence:.2f}%")
```

## Conclusion

While our DNN achieves impressive accuracy on the MNIST test set, its performance on custom handwritten digits reveals limitations in generalization. This makes a strong case for transitioning to CNN architectures, which are better suited for image classification tasks due to their ability to capture spatial features and provide translation invariance.

The project demonstrates the value of establishing a strong baseline model before moving to more complex architectures, and highlights the importance of evaluating models on diverse inputs beyond the standard test set.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
