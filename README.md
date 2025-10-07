# 🌾 Rice Leaf Disease Recognition using Deep Learning

This project is part of my **M.Sc. Data Science Thesis (GITAM University, 2024)**.  
The goal is to detect and classify **rice leaf diseases** using **Convolutional Neural Networks (CNNs)** with **Transfer Learning** for deployment on edge devices (like mobile phones).

---

## 🚀 Project Overview
Rice is a staple food for more than half of the global population. Early and accurate detection of rice leaf diseases is crucial for sustainable agriculture.  
Manual detection is time-consuming and error-prone. This project leverages **MobileNetV2** and compares its performance against **Custom VGG16** for efficient disease recognition.

- **Dataset:** 5,932 rice leaf images (Bacterial Blight, Blast, Brown Spot, Tungro)  
- **Techniques:** Data augmentation, Transfer Learning, Regularization  
- **Goal:** Develop a lightweight model that can run on **edge devices** for farmers

---

## 🛠️ Tech Stack
- **Programming Language:** Python  
- **Deep Learning Frameworks:** TensorFlow, Keras  
- **Other Libraries:** NumPy, Pandas, OpenCV, Matplotlib, Scikit-learn  
- **Deployment Tools:** (Optional) Gradio / Streamlit for demo  

---

## 📂 Dataset
- Source: [Mendeley Dataset](https://data.mendeley.com/datasets/fwcj7stb8r/draft?a=d8923d70-cfc6-4c6c-adc0-640f10152fdf)  
- Classes:
  - 🌱 Bacterial Blight – 1,584 images  
  - 🍂 Blast – 1,440 images  
  - 🍁 Brown Spot – 1,600 images  
  - 🍃 Tungro – 1,308 images  

Total: **5,932 images**, expanded to **28,465 images** using augmentation.

---

## 📊 Results
- **MobileNetV2 Performance**
  - Accuracy: **99.83%**
  - Precision: **0.998**
  - Recall: **0.998**
  - F1 Score: **0.998**
- **Comparison with VGG16**
  - MobileNetV2 = 99.45% accuracy
  - VGG16 = 99.57% accuracy
- **Observation:** MobileNetV2 is **lighter and faster** → suitable for mobile apps.

---

## Sample Output
- Input: Rice leaf image (captured by camera/mobile)  
- Output: Predicted class (e.g., "Brown Spot") + Confidence score  

## Sample demo video: 
- Deployment video: [Video](https://drive.google.com/file/d/1u7wo0aXab3fhRlnG6X85Ny7lRTf2z213/view?usp=sharing)
---

## ⚙️ How to Run
```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/rice-leaf-disease.git
cd rice-leaf-disease

# Install dependencies
pip install -r requirements.txt

# Train model
python train.py

# Test model
python evaluate.py
