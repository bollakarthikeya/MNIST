# Neural Network Experiments on MNIST

## Overview

This project explores multiple implementations of a **fully connected neural network (written from scratch) for handwritten digit classification** using  MNIST dataset. Three training approaches are implemented to compare neural-network construction with a framework-based implementation, while keeping the underlying classification task and model structure broadly consistent.

## Dataset

The project uses the standard MNIST handwritten-digit dataset:

- **60,000** training images
- **10,000** test images
- **10 classes:** digits `0`–`9`
- Image dimensions: **28 × 28**
- Each image is flattened into **784 input features**
- Pixel values are normalized to the range `[0, 1]`

## Neural Network Design

The experiments use a feed-forward, fully connected neural-network architecture with:

```text
784 Input Features → Hidden Layer → 10-Class Output
```

Common architectural elements include:

- A high-dimensional fully connected hidden layer
- **ReLU** activation in the hidden layer
- **Softmax** output for multiclass classification
- **Cross-entropy** loss
- **L2 regularization**
- Gradient-based parameter optimization
- Weight initialization appropriate for ReLU-based networks

The implementations primarily use **1,000 hidden units**, with the output layer representing the ten MNIST digit classes.

## Implementations

### 1. Neural Network with Batched Gradient Updates

`nn_batch_GD.py`

Implements the neural network largely from scratch using NumPy matrix operations. Forward propagation, Softmax probabilities, cross-entropy loss, L2 regularization, backpropagation, and parameter updates are explicitly implemented.

Training uses shuffled training/validation partitions and processes the training data in batches while tracking loss and classification accuracy across epochs.

### 2. Neural Network with Mini-Batch Gradient Descent

`nn_mini_batch_GD.py`

Implements the same general neural-network pipeline from scratch with **mini-batch gradient descent** and **10-fold cross-validation**.

The implementation explicitly computes:

- Forward propagation
- ReLU activation
- Softmax probabilities
- Cross-entropy and regularization loss
- Backpropagated gradients
- Weight and bias updates

Training examples are divided into mini-batches of **30 observations**.

### 3. Keras Neural Network

`nn_keras.py`

Implements the network using the **Keras Sequential API** as a higher-level counterpart to the from-scratch implementations.

The model uses:

- Dense hidden and output layers
- ReLU and Softmax activations
- He/Glorot weight initialization
- L2 regularization
- Stochastic Gradient Descent
- Categorical cross-entropy
- Batch size of **30**
- **10-fold cross-validation**

## Training Configuration

Across the experiments, the principal configuration is approximately:

| Parameter | Configuration |
|---|---|
| Input dimension | 784 |
| Hidden units | 1,000 |
| Output classes | 10 |
| Hidden activation | ReLU |
| Output activation | Softmax |
| Loss | Cross-entropy |
| Regularization | L2 (`0.001`) |
| Learning rate | `0.09` |
| Validation | Holdout and/or 10-fold cross-validation |
| Evaluation | Classification accuracy |

## Focus

The work demonstrates the same multi-class neural-network problem through different implementation strategies:

- implementing forward and backward propagation directly with matrix operations;
- experimenting with batch and mini-batch optimization workflows;
- applying cross-validation for model evaluation; and
- reproducing the neural network architecture with a high-level deep-learning framework.

This provides a practical comparison between **from-scratch neural-network mechanics** and **framework-based model construction** while retaining a common  classification objective.

## Tech Stack

- Python
- NumPy / SciPy
- Keras
- scikit-learn
- Matplotlib
- idx2numpy
