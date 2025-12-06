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
```
## ▶️ Run the App 

https://deepfake-photodetector.streamlit.app/

### 1. Clone the Repository

git clone https://github.com/Rishikesh23a/deepfake_Photo_streamlit_app.git
cd deepfake_Photo_streamlit_app

2. Install Requirements
pip install -r requirements.txt

3. Run Streamlit App
streamlit run deepfake_app.py


The app will open in your browser automatically.
```

## 📸 App Screenshots

<p align="center">
  <img src="screenshots/screenshot1.png" width="600">
</p>

<p align="center">
  <img src="screenshots/screenshot2.png" width="600">
</p>

<p align="center">
  <img src="screenshots/screenshot3.png" width="600">
</p>

📌 Future Enhancements

• Add video deepfake detection

• Train CNN model for better accuracy

• Deploy online using Streamlit Cloud / Render

• Add explainability (GradCam / Heatmap)

• Add face detection + ROI processing

```
👨‍💻 Developer

Rushikesh Sable

MIT AOE College

📧 rushikeshsable9850@gmail.com
```
