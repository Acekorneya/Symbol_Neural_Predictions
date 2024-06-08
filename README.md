# Neural Network for Optimizing Symbol Sequences
This repository contains the implementation of a neural network model designed to optimize symbol sequences based on a target vector. The project includes data preprocessing, model training using K-fold cross-validation, and prediction of optimal symbol sequences. The implementation also includes a graphical user interface for visualizing symbols and managing training data.

## Introduction
This project aims to predict and optimize symbol sequences based on given training data and a target vector. The neural network is trained using LSTM layers and utilizes K-fold cross-validation to ensure robust model performance. Additionally, a graphical user interface is provided to display images corresponding to symbols and to facilitate data management.

## Features
- Training data management with persistent storage using pickle files.
- Neural network model with LSTM layers for sequence prediction.
- K-fold cross-validation for robust model evaluation.
- GUI for visualizing symbols and managing training data.
- Parallel processing to optimize symbol sequences efficiently.

## Model Architecture
The model is built using TensorFlow and Keras with the following architecture:
- Input Layer
- Embedding Layer (pre-trained)
- LSTM Layer 1
- Dropout Layer 1
- LSTM Layer 2
- Dropout Layer 2
- LSTM Layer 3
- Dense Output Layer

The model is compiled using the Mean Squared Error (MSE) loss function and the Adam optimizer.

## Training
The training process involves K-fold cross-validation to ensure model robustness. The training data is tokenized, padded, and fed into the model. Early stopping is used to prevent overfitting.

## Evaluation
The best model is selected based on the validation loss. The script evaluates the model's performance by predicting the target vector and comparing it to the actual game vector data.

