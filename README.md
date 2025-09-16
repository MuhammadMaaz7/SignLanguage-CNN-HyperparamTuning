# Sign Language Digit Recognition using CNNs

This project implements a Convolutional Neural Network (CNN) for 
recognizing digits (0–9) from the [Sign Language Digits Dataset](https://github.com/ardamavzi/Sign-Language-Digits-Dataset).  
The main purpose was to **analyze the effect of different hyperparameters** on model performance.

## 📊 Experiments
We tuned the following hyperparameters:
- Batch size: 16, 32, 64
- Learning rate: 1e-2, 1e-3, 1e-4
- Dropout ratio: 0.2, 0.3, 0.5
- L2 regularization: 0, 1e-5, 1e-4
- Early stopping patience: 5, 10, 15
- Epochs: max 60 with early stopping
- Normalization applied to input images

## ✅ Best Model
- Batch size = 32  
- Dropout = 0.2  
- L2 = 1e-5  
- Patience = 12  
- Learning rate = 1e-3  

**Results:**  
- Validation Accuracy: 92.5%  
- Macro F1-score: 0.92  
- AUC: 0.995  

## 📂 Repository Structure
