# deep-learningFacial Expression Recognition and Affective Computing
Project Overview

This project implements a deep learning system for recognizing facial expressions and predicting emotional states (valence and arousal) from facial images. It uses multi-task learning to simultaneously predict:

Expression classification – discrete emotion labels such as happy, sad, angry, etc.

Valence and Arousal regression – continuous emotional dimensions.

The system leverages pre-trained CNN architectures (VGG16, ResNet50, EfficientNetB0) and a custom CNN to compare performance.

Features

Multi-task prediction: expression classification + valence-arousal regression.

Supports flexible datasets: images, CSV annotations, or .npy files.

Preprocessing includes resizing, normalization, and augmentation.

Training supports early stopping, learning rate adjustment, and checkpoint saving.

Comprehensive evaluation metrics:

Classification: Accuracy, F1-score, Cohen’s Kappa, ROC-AUC

Regression: RMSE, Pearson correlation, CCC, Sign Agreement

Visualization of training curves, confusion matrices, and sample predictions.

Modular code design for easy experimentation with different architectures.

Installation

Clone the repository:

git clone <repository-url>
cd facial-expression-recognition


Create a Python virtual environment (recommended):

python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows


Install required packages:

pip install -r requirements.txt


Dependencies include:

Python 3.8+

TensorFlow / Keras

NumPy

Pandas

OpenCV

Matplotlib / Seaborn

Scikit-learn

Usage

Prepare Dataset

Place images in the images/ directory.

Provide annotations in CSV or .npy format (expressions.npy, valence.npy, arousal.npy).

Preprocess Data

Run the dataset loader to resize, normalize, and split data.

Train Model

Choose a model: VGG16, ResNet50, EfficientNetB0, or CustomCNN.

Configure training parameters: batch size, epochs, learning rate.

Start training:

python train_model.py --model VGG16 --epochs 50


Evaluate Model

Evaluate metrics on test set:

python evaluate_model.py --model VGG16


Visualize Results

Plot training curves and confusion matrices:

python visualize_results.py --model VGG16


valuation Metrics

Classification

Accuracy, F1-score, Cohen’s Kappa, ROC-AUC

Regression

RMSE, Pearson correlation, Concordance Correlation Coefficient (CCC), Sign Agreement

Multi-Model Comparison

Pre-trained models converge faster due to transfer learning.

Custom CNN allows multi-scale feature extraction and attention mechanisms.

Data augmentation reduces overfitting and improves generalization.

Side-by-side metrics comparison identifies the best model for both classification and regression tasks.
