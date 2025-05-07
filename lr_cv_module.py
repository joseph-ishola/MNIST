"""
MNIST CNN Module with Learning Rate Scheduling and Cross-Validation

This module combines:
1. Learning rate scheduling (step, exponential, plateau, custom)
2. K-fold cross-validation
3. Model training and evaluation

The goal is to systematically find the best learning rate schedule and 
evaluate model performance across multiple data splits.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
)
from sklearn.model_selection import KFold
import pandas as pd
import time
import os
from typing import Tuple, List, Dict, Any, Union, Optional, Callable


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


class LRHistory(keras.callbacks.Callback):
    """Custom callback to track learning rates during training."""
    
    def __init__(self):
        super().__init__()
        self.lr_values = []
        
    def on_epoch_end(self, epoch, logs=None):
        self.lr_values.append(self.model.optimizer.lr.numpy())


# Learning Rate Scheduler Functions

def step_decay_schedule(initial_lr=0.001, decay_factor=0.5, step_size=5):
    """Step decay schedule - reduces learning rate by a factor at regular intervals."""
    def schedule(epoch):
        return initial_lr * (decay_factor ** (epoch // step_size))
    return schedule


def exponential_decay_schedule(initial_lr=0.001, decay_rate=0.95):
    """Exponential decay schedule - continuously reduces learning rate."""
    def schedule(epoch):
        return initial_lr * (decay_rate ** epoch)
    return schedule


def custom_schedule(initial_lr=0.001, max_lr=0.01, min_lr=0.0001, warmup_epochs=3, decay_epochs=7):
    """Custom decay schedule combining initial warmup with later cosine decay."""
    def schedule(epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            return initial_lr + (max_lr - initial_lr) * (epoch / warmup_epochs)
        else:
            # Cosine decay
            decay_progress = (epoch - warmup_epochs) / decay_epochs
            cosine_decay = 0.5 * (1 + np.cos(np.pi * min(decay_progress, 1.0)))
            return min_lr + (max_lr - min_lr) * cosine_decay
    return schedule


def load_and_prepare_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and prepare MNIST data for cross-validation.
    
    Returns:
    --------
    tuple
        (x_train, y_train, x_test, y_test) - Preprocessed NumPy arrays
    """
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Reshape and normalize data
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    
    return x_train, y_train, x_test, y_test


def create_cnn_model(learning_rate: float = 0.001) -> keras.Sequential:
    """
    Create a CNN model for MNIST with specified learning rate.
    
    Parameters:
    -----------
    learning_rate : float
        Learning rate for the Adam optimizer
        
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
    
    # Compile the model with specified learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def get_lr_callbacks(schedule_type: str, initial_lr: float = 0.001) -> Tuple[List, LRHistory]:
    """
    Create callbacks for the specified learning rate schedule.
    
    Parameters:
    -----------
    schedule_type : str
        Type of learning rate schedule ('plateau', 'step', 'exponential', or 'custom')
    initial_lr : float
        Initial learning rate
        
    Returns:
    --------
    tuple
        (callbacks_list, lr_history) - List of callbacks and LRHistory object
    """
    callbacks = []
    
    # Create LRHistory callback to track learning rates
    lr_history = LRHistory()
    callbacks.append(lr_history)
    
    # Add the specific learning rate scheduler
    if schedule_type == 'plateau':
        # ReduceLROnPlateau: reduce learning rate when a metric plateaus
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,       # reduce LR by half
            patience=3,       # wait 3 epochs before reducing
            min_lr=0.00001,   # don't go below this LR
            verbose=1
        )
        callbacks.append(reduce_lr)
        
    elif schedule_type == 'step':
        # Step decay: reduce learning rate after fixed intervals
        lr_scheduler = LearningRateScheduler(
            step_decay_schedule(initial_lr=initial_lr),
            verbose=1
        )
        callbacks.append(lr_scheduler)
        
    elif schedule_type == 'exponential':
        # Exponential decay: continuously reduce learning rate
        lr_scheduler = LearningRateScheduler(
            exponential_decay_schedule(initial_lr=initial_lr),
            verbose=1
        )
        callbacks.append(lr_scheduler)
        
    elif schedule_type == 'custom':
        # Custom schedule with warmup and decay
        lr_scheduler = LearningRateScheduler(
            custom_schedule(initial_lr=initial_lr),
            verbose=1
        )
        callbacks.append(lr_scheduler)
        
    # Add early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stopping)
    
    return callbacks, lr_history


def train_fold(x_train: np.ndarray, y_train: np.ndarray, 
               x_val: np.ndarray, y_val: np.ndarray,
               schedule_type: str, fold: int,
               epochs: int = 20, batch_size: int = 128,
               initial_lr: float = 0.001) -> Dict[str, Any]:
    """
    Train a model on a single fold with a specific learning rate schedule.
    
    Parameters:
    -----------
    x_train : np.ndarray
        Training data for this fold
    y_train : np.ndarray
        Training labels for this fold
    x_val : np.ndarray
        Validation data for this fold
    y_val : np.ndarray
        Validation labels for this fold
    schedule_type : str
        Type of learning rate schedule
    fold : int
        Fold number
    epochs : int
        Maximum number of epochs to train
    batch_size : int
        Batch size for training
    initial_lr : float
        Initial learning rate
        
    Returns:
    --------
    dict
        Dictionary containing training results, model, and history
    """
    print(f"\nTraining Fold {fold} with {schedule_type} learning rate schedule")
    
    # Create a fresh model
    model = create_cnn_model(learning_rate=initial_lr)
    
    # Get learning rate callbacks
    callbacks, lr_history = get_lr_callbacks(schedule_type, initial_lr)
    
    # Add metrics history callback
    metrics_history = MetricsHistory()
    callbacks.append(metrics_history)
    
    # Train the model
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate the model
    val_loss, val_accuracy = model.evaluate(x_val, y_val, verbose=0)
    print(f"Fold {fold} validation accuracy: {val_accuracy:.4f}")
    
    # Return results
    return {
        'fold': fold,
        'schedule_type': schedule_type,
        'val_accuracy': val_accuracy,
        'val_loss': val_loss,
        'model': model,
        'history': history,
        'lr_history': lr_history.lr_values,
        'metrics_history': metrics_history
    }


def cross_validate_schedules(schedules: List[str], 
                            num_folds: int = 5,
                            epochs: int = 20,
                            batch_size: int = 128,
                            initial_lr: float = 0.001) -> Dict[str, List[Dict]]:
    """
    Perform cross-validation for multiple learning rate schedules.
    
    Parameters:
    -----------
    schedules : List[str]
        List of learning rate schedule types to evaluate
    num_folds : int
        Number of folds for cross-validation
    epochs : int
        Maximum number of epochs per fold
    batch_size : int
        Batch size for training
    initial_lr : float
        Initial learning rate
        
    Returns:
    --------
    dict
        Dictionary with schedule types as keys and lists of fold results as values
    """
    # Load and prepare data
    x_train, y_train, x_test, y_test = load_and_prepare_data()
    
    # Set up k-fold cross-validation
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    
    # Dictionary to store results for each schedule
    schedule_results = {schedule: [] for schedule in schedules}
    
    # Track total time
    start_time = time.time()
    
    # Run cross-validation for each schedule
    for schedule_type in schedules:
        print(f"\n{'='*50}")
        print(f"Starting cross-validation for {schedule_type} schedule")
        print(f"{'='*50}")
        
        # Run each fold
        fold_no = 1
        for train_idx, val_idx in kfold.split(x_train):
            # Split data for this fold
            x_train_fold, x_val_fold = x_train[train_idx], x_train[val_idx]
            y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
            
            # Train and evaluate on this fold
            fold_result = train_fold(
                x_train_fold, y_train_fold,
                x_val_fold, y_val_fold,
                schedule_type, fold_no,
                epochs, batch_size, initial_lr
            )
            
            # Store results
            schedule_results[schedule_type].append(fold_result)
            fold_no += 1
    
    # Total time for cross-validation
    total_time = time.time() - start_time
    print(f"\nTotal cross-validation time: {total_time:.2f} seconds")
    
    return schedule_results


def analyze_cv_results(schedule_results: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """
    Analyze cross-validation results across different learning rate schedules.
    
    Parameters:
    -----------
    schedule_results : Dict[str, List[Dict]]
        Results from cross-validation
        
    Returns:
    --------
    dict
        Analysis results, including best schedule
    """
    # Create a list to store results for DataFrame
    all_results = []
    
    # Calculate mean and std for each schedule
    summary = {}
    
    for schedule, folds in schedule_results.items():
        # Extract accuracies
        accuracies = [fold['val_accuracy'] for fold in folds]
        
        # Calculate statistics
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        max_acc = np.max(accuracies)
        
        # Store in summary
        summary[schedule] = {
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'max_accuracy': max_acc,
            'folds': len(folds)
        }
        
        # Add results to overall list
        for fold in folds:
            all_results.append({
                'schedule': schedule,
                'fold': fold['fold'],
                'val_accuracy': fold['val_accuracy'],
                'val_loss': fold['val_loss']
            })
    
    # Create DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Find best schedule based on mean accuracy
    schedules_by_mean = sorted(summary.items(), key=lambda x: x[1]['mean_accuracy'], reverse=True)
    best_schedule = schedules_by_mean[0][0]
    
    # Find best single fold
    best_fold_row = results_df.loc[results_df['val_accuracy'].idxmax()]
    best_fold_schedule = best_fold_row['schedule']
    best_fold_number = best_fold_row['fold']
    best_fold_accuracy = best_fold_row['val_accuracy']
    
    # Return analysis results
    return {
        'summary': summary,
        'results_df': results_df,
        'best_schedule': best_schedule,
        'best_fold_schedule': best_fold_schedule,
        'best_fold_number': best_fold_number,
        'best_fold_accuracy': best_fold_accuracy
    }


def plot_cv_results(schedule_results: Dict[str, List[Dict]], analysis: Dict) -> None:
    """
    Plot cross-validation results for different learning rate schedules.
    
    Parameters:
    -----------
    schedule_results : Dict[str, List[Dict]]
        Results from cross-validation
    analysis : Dict
        Analysis results from analyze_cv_results
    """
    # Set up figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(18, 16))
    
    # 1. Plot validation accuracies for each schedule and fold
    ax = axs[0, 0]
    results_df = analysis['results_df']
    for schedule in results_df['schedule'].unique():
        schedule_data = results_df[results_df['schedule'] == schedule]
        ax.plot(schedule_data['fold'], schedule_data['val_accuracy'], 
                'o-', label=schedule)
    
    ax.set_title('Validation Accuracy by Fold and Schedule')
    ax.set_xlabel('Fold')
    ax.set_ylabel('Validation Accuracy')
    ax.legend()
    ax.grid(True)
    
    # 2. Plot learning rates over epochs for each schedule (using first fold)
    ax = axs[0, 1]
    for schedule, folds in schedule_results.items():
        lr_history = folds[0]['lr_history']  # Use first fold
        epochs = range(1, len(lr_history) + 1)
        ax.plot(epochs, lr_history, label=schedule)
    
    ax.set_title('Learning Rate Over Epochs (First Fold)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True)
    
    # 3. Plot boxplot of validation accuracies by schedule
    ax = axs[1, 0]
    schedules = results_df['schedule'].unique()
    data = [results_df[results_df['schedule'] == s]['val_accuracy'].values 
            for s in schedules]
    
    box = ax.boxplot(data, labels=schedules, patch_artist=True)
    
    # Add colors to boxplots
    colors = ['lightblue', 'lightgreen', 'lightpink', 'lightyellow']
    for patch, color in zip(box['boxes'], colors[:len(schedules)]):
        patch.set_facecolor(color)
    
    ax.set_title('Distribution of Validation Accuracies by Schedule')
    ax.set_xlabel('Schedule')
    ax.set_ylabel('Validation Accuracy')
    ax.grid(True)
    
    # 4. Plot mean validation accuracies with error bars
    ax = axs[1, 1]
    summary = analysis['summary']
    
    schedules = list(summary.keys())
    means = [summary[s]['mean_accuracy'] for s in schedules]
    stds = [summary[s]['std_accuracy'] for s in schedules]
    
    ax.bar(schedules, means, yerr=stds, alpha=0.7, capsize=10)
    ax.set_title('Mean Validation Accuracy by Schedule')
    ax.set_xlabel('Schedule')
    ax.set_ylabel('Mean Validation Accuracy')
    ax.grid(True)
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()


def plot_fold_histories(schedule_results: Dict[str, List[Dict]], best_schedule: str) -> None:
    """
    Plot training histories for all folds of the best schedule.
    
    Parameters:
    -----------
    schedule_results : Dict[str, List[Dict]]
        Results from cross-validation
    best_schedule : str
        Name of the best learning rate schedule
    """
    # Get folds for the best schedule
    folds = schedule_results[best_schedule]
    
    # Set up figure with subplots
    fig, axs = plt.subplots(2, 1, figsize=(14, 12))
    
    # 1. Plot accuracy for each fold
    ax = axs[0]
    for fold in folds:
        metrics = fold['metrics_history']
        epochs = range(1, len(metrics.accuracy) + 1)
        ax.plot(epochs, metrics.accuracy, 
                label=f"Train (Fold {fold['fold']})")
        ax.plot(epochs, metrics.val_accuracy, '--',
                label=f"Val (Fold {fold['fold']})")
    
    ax.set_title(f'Accuracy Over Epochs for {best_schedule} Schedule')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True)
    
    # 2. Plot loss for each fold
    ax = axs[1]
    for fold in folds:
        metrics = fold['metrics_history']
        epochs = range(1, len(metrics.losses) + 1)
        ax.plot(epochs, metrics.losses, 
                label=f"Train (Fold {fold['fold']})")
        ax.plot(epochs, metrics.val_losses, '--',
                label=f"Val (Fold {fold['fold']})")
    
    ax.set_title(f'Loss Over Epochs for {best_schedule} Schedule')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()


def train_final_model(best_schedule: str, 
                     initial_lr: float = 0.001,
                     epochs: int = 30, 
                     batch_size: int = 128) -> Dict[str, Any]:
    """
    Train final model on all training data using the best learning rate schedule.
    
    Parameters:
    -----------
    best_schedule : str
        Best learning rate schedule type
    initial_lr : float
        Initial learning rate
    epochs : int
        Maximum number of epochs
    batch_size : int
        Batch size for training
        
    Returns:
    --------
    dict
        Dictionary containing the final model and results
    """
    print(f"\n{'='*50}")
    print(f"Training final model with {best_schedule} schedule on all data")
    print(f"{'='*50}")
    
    # Load and prepare data
    x_train, y_train, x_test, y_test = load_and_prepare_data()
    
    # Create a fresh model
    model = create_cnn_model(learning_rate=initial_lr)
    
    # Get learning rate callbacks
    callbacks, lr_history = get_lr_callbacks(best_schedule, initial_lr)
    
    # Add metrics history callback
    metrics_history = MetricsHistory()
    callbacks.append(metrics_history)
    
    # Add model checkpoint to save best model
    checkpoint_path = f"final_model_{best_schedule}.h5"
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    callbacks.append(checkpoint)
    
    # Split some validation data
    val_split = 0.1
    val_samples = int(len(x_train) * val_split)
    
    x_train_final = x_train[val_samples:]
    y_train_final = y_train[val_samples:]
    x_val_final = x_train[:val_samples]
    y_val_final = y_train[:val_samples]
    
    # Train final model
    history = model.fit(
        x_train_final, y_train_final,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val_final, y_val_final),
        callbacks=callbacks,
        verbose=1
    )
    
    # Load best weights
    if os.path.exists(checkpoint_path):
        print(f"\nLoading best model from {checkpoint_path}")
        model = keras.models.load_model(checkpoint_path)
    
    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
    print(f"\nFinal model test accuracy: {test_accuracy:.4f}")
    
    return {
        'model': model,
        'history': history,
        'lr_history': lr_history.lr_values,
        'metrics_history': metrics_history,
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'checkpoint_path': checkpoint_path
    }


def plot_final_model_results(final_results: Dict[str, Any], best_schedule: str) -> None:
    """
    Plot training results for the final model.
    
    Parameters:
    -----------
    final_results : Dict[str, Any]
        Results from final model training
    best_schedule : str
        Name of the best learning rate schedule
    """
    metrics = final_results['metrics_history']
    history = final_results['history']
    
    # Set up figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(18, 14))
    
    # 1. Plot accuracy
    ax = axs[0, 0]
    epochs = range(1, len(metrics.accuracy) + 1)
    ax.plot(epochs, metrics.accuracy, label="Training")
    ax.plot(epochs, metrics.val_accuracy, label="Validation")
    
    ax.set_title(f'Final Model Accuracy ({best_schedule} schedule)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True)
    
    # 2. Plot loss
    ax = axs[0, 1]
    ax.plot(epochs, metrics.losses, label="Training")
    ax.plot(epochs, metrics.val_losses, label="Validation")
    
    ax.set_title(f'Final Model Loss ({best_schedule} schedule)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)
    
    # 3. Plot learning rate
    ax = axs[1, 0]
    lr_history = final_results['lr_history']
    epochs_lr = range(1, len(lr_history) + 1)
    ax.plot(epochs_lr, lr_history)
    
    ax.set_title(f'Learning Rate Over Epochs ({best_schedule} schedule)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_yscale('log')
    ax.grid(True)
    
    # 4. Plot test accuracy
    ax = axs[1, 1]
    test_acc = final_results['test_accuracy']
    ax.bar(['Test Accuracy'], [test_acc], color='green')
    ax.set_ylim([0.95, 1.0])  # Assuming accuracy is high
    
    ax.set_title('Final Model Test Accuracy')
    ax.set_ylabel('Accuracy')
    for i, v in enumerate([test_acc]):
        ax.text(i, v-0.01, f"{v:.4f}", ha='center', fontweight='bold')
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()


def run_complete_experiment(
    schedules: List[str] = ['plateau', 'step', 'exponential', 'custom'],
    num_folds: int = 5,
    cv_epochs: int = 20,
    final_epochs: int = 30,
    batch_size: int = 128,
    initial_lr: float = 0.001
) -> Dict[str, Any]:
    """
    Run a complete experiment with cross-validation and final model training.
    
    Parameters:
    -----------
    schedules : List[str]
        List of learning rate schedule types to evaluate
    num_folds : int
        Number of folds for cross-validation
    cv_epochs : int
        Maximum number of epochs for cross-validation
    final_epochs : int
        Maximum number of epochs for final model
    batch_size : int
        Batch size for training
    initial_lr : float
        Initial learning rate
        
    Returns:
    --------
    dict
        Dictionary containing all experiment results
    """
    print(f"Starting complete experiment with {len(schedules)} schedules and {num_folds} folds")
    experiment_start = time.time()
    
    # 1. Cross-validate all learning rate schedules
    cv_results = cross_validate_schedules(
        schedules, num_folds, cv_epochs, batch_size, initial_lr
    )
    
    # 2. Analyze cross-validation results
    analysis = analyze_cv_results(cv_results)
    
    # Print cross-validation summary
    print("\nCross-Validation Summary:")
    print("========================")
    for schedule, stats in analysis['summary'].items():
        print(f"{schedule.capitalize()} Schedule:")
        print(f"  Mean Accuracy: {stats['mean_accuracy']:.4f} ± {stats['std_accuracy']:.4f}")
        print(f"  Max Accuracy: {stats['max_accuracy']:.4f}")
    
    best_schedule = analysis['best_schedule']
    print(f"\nBest Schedule: {best_schedule} (Mean Accuracy: {analysis['summary'][best_schedule]['mean_accuracy']:.4f})")
    
    # 3. Plot cross-validation results
    plot_cv_results(cv_results, analysis)
    
    # 4. Plot training histories for best schedule
    plot_fold_histories(cv_results, best_schedule)
    
    # 5. Train final model with best schedule
    final_results = train_final_model(
        best_schedule, initial_lr, final_epochs, batch_size
    )
    
    # 6. Plot final model results
    plot_final_model_results(final_results, best_schedule)
    
    # Calculate total experiment time
    total_time = time.time() - experiment_start
    print(f"\nTotal experiment time: {total_time/60:.2f} minutes")
    
    # Return all results
    return {
        'cv_results': cv_results,
        'analysis': analysis,
        'best_schedule': best_schedule,
        'final_results': final_results
    }


if __name__ == "__main__":
    # Example usage
    experiment_results = run_complete_experiment(
        schedules=['plateau', 'step', 'exponential', 'custom'],
        num_folds=5,
        cv_epochs=15,  # Reduced for demonstration
        final_epochs=20
    )