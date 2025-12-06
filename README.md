<h1 align="center">🧠🔍 Deepfake Photo Detection App (Streamlit)</h1>

<p align="center">
A lightweight and efficient Deepfake Image Detection Web App built using <b>Streamlit</b>, <b>Machine Learning</b>, and <b>OpenCV</b>.  
Upload an image → the model analyzes it → gives you the probability of the image being REAL or FAKE.
</p>

---

## 🚀 Overview

Deepfake images are becoming increasingly common, posing security and authenticity risks.  
This project provides a **simple and fast web app** that detects deepfake images using a trained ML model.

The app is deployed using **Streamlit**, making it easy to run locally or deploy on the cloud.

---

## ✨ Features

- 📤 **Upload any face image (JPG/PNG)**
- 🤖 **Machine Learning model predicts Real vs Fake**
- 📊 **Displays prediction probability**
- 🧠 **Uses pre-trained classifier**
- ⚡ **Fast and lightweight**
- 🌐 **Streamlit-based clean UI**

---

## 🧐 How It Works

1. User uploads an image  
2. App preprocesses the image (resize → normalize)  
3. Image is passed through the ML model  
4. Model outputs real/fake probability  
5. Streamlit displays the result with colors/percentage  

---

## 🛠 Technology Stack

| Component | Usage |
|----------|--------|
| **Streamlit** | Web UI framework |
| **Python** | Main programming language |
| **OpenCV** | Image processing |
| **scikit-learn / ML model** | Deepfake classification |
| **Pickle (.pkl)** | Model serialization |
| **NumPy** | Array processing |

---

## 📁 Folder Structure
```
deepfake_Photo_streamlit_app/
│
├─ ml_artifacts/
│ └─ model.pkl # Trained ML model
│
├─ deepfake_app.py # Streamlit App
├─ requirements.txt # Dependencies
├─ .gitattributes
└─ README.md
```



