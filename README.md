# Medical Image Classification Using CNN

## Overview

This project implements a Medical Image Classification system using a Convolutional Neural Network (CNN) to classify chest X-ray images as Normal and Pneumonia. The model is trained using TensorFlow and Python and can automatically predict the presence of pneumonia from X-ray images.

## Features

* Chest X-ray image classification
* Deep Learning using CNN
* Automatic disease prediction
* Trained model saved as `model.h5`
* Prediction on new X-ray images

## Dataset Structure

Dataset/
├── Train/
│ ├── Normal/
│ └── Pneumonia/
│
└── Test/
├── Normal/
└── Pneumonia/

Dataset is not included in this repository due to GitHub storage limitations.
The Chest X-Ray dataset can be downloaded from Kaggle.

## Technologies Used

* Python
* TensorFlow
* NumPy
* Keras
* VS Code

## Project Structure

Medical_Image_Classification/
├── Dataset/
├── train_model.py
├── predict.py
├── model.h5
├── README.md

## Model Architecture

* Conv2D Layer
* MaxPooling2D Layer
* Conv2D Layer
* MaxPooling2D Layer
* Flatten Layer
* Dense Layer
* Output Layer

## Workflow

1. Load Dataset
2. Preprocess Images
3. Build CNN Model
4. Train Model
5. Save Model
6. Predict New Images

## Training

Run:

python train_model.py

The trained model will be saved as:

model.h5

The trained model file (model.h5) is not included due to GitHub upload limitations.
Run train_model.py to generate the model locally.

## Prediction

Run:

python predict.py

Enter the path of a chest X-ray image and the model will predict:

* Normal
* Pneumonia

## Applications

* Healthcare Systems
* Hospitals
* Diagnostic Centers
* Medical Research

## Future Scope

* Multi-disease classification
* Improved accuracy with larger datasets
* Web application deployment
* Real-time diagnosis support

## Conclusion

This project demonstrates the use of Convolutional Neural Networks (CNNs) for medical image classification. The system helps automate pneumonia detection from chest X-ray images and showcases the application of deep learning in healthcare.

