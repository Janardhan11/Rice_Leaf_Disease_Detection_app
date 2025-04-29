# 🌾 Rice Leaf Disease Recognition using Deep Learning
## Overview
This project aims to develop a lightweight and accurate deep learning model to recognize and classify diseases in rice leaves. Early detection of leaf diseases plays a crucial role in preventing crop loss and improving agricultural productivity. The model is designed to be efficient enough for real-time usage, making it suitable for farmers and agricultural advisors even on low-resource devices.

## Problem Statement
Rice is a major food crop worldwide, and diseases affecting rice leaves can significantly reduce crop yields. Traditional disease detection methods are time-consuming and require expert knowledge. Automating this process using deep learning can provide a faster, more accessible solution to farmers, leading to early treatment and better crop management.

## Approach
### Data Collection and Preprocessing:
Collected a dataset of 5,000+ images covering four categories: three major rice diseases and healthy leaves.Applied image augmentation techniques (rotation, zoom, horizontal flip, vertical flip, brightness adjustment, and cropping) to expand the dataset 3x, improving the model's robustness.

### Model Architecture:
Used MobileNetV2, a lightweight and efficient Convolutional Neural Network (CNN), ideal for real-time applications. Fine-tuned the pre-trained MobileNetV2 model on the rice leaf dataset.

### Training:
Implemented using TensorFlow/Keras. Achieved 90%+ classification accuracy with low training time and efficient model size. Used techniques like early stopping and learning rate adjustment for better performance.

### Deployment:
Built a real-time prediction web app using Gradio. The app allows users to upload rice leaf images and receive instant disease classification with an average inference time of 3 seconds per image.

## Tech Stack
Python, TensorFlow / Keras, MobileNetV2, DNN, OpenCV, Gradio, Git & GitHub

## Key Features
Lightweight and fast model suitable for mobile and edge devices.

Real-time disease detection through a user-friendly web interface.

Dataset expansion through smart augmentation to improve model generalization.

High classification accuracy with minimal resource usage.

## Future Work
Expand the dataset to include more disease classes and real-field images.

Deploy the model on Android/mobile devices.

Integrate with farm management systems for better usage at scale.

## Hugging Faces link
[🖥️ **Try the Real-Time App Here**](https://huggingface.co/spaces/JanardhanM/RiceLeafDiseaseDetection)
