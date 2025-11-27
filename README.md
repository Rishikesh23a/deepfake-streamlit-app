# Deepfake Image Detector – Streamlit App

This repository contains a **Deepfake Image Detection Web App** built using **Streamlit** and a **Convolutional Neural Network (CNN)** trained in **PyTorch**.  
The app allows users to upload an image and get a prediction whether the face is **real** or **fake (deepfake)** along with a confidence score.

---

## 🧠 Project Overview

Deepfakes are AI-generated or AI-manipulated images and videos that can closely resemble real people.  
They pose serious risks like misinformation, identity misuse, and privacy violations.

This project focuses on the **detection side** of deepfakes:

- A CNN-based model is trained on a **real vs fake face dataset (RVF10k)**.
- The best-performing model is saved as `best_model.pth`.
- A **Streamlit app (`app.py`)** loads this model and provides an **interactive UI** for classification.

---

## ✨ Features

- ✅ Upload an image in `.jpg`, `.jpeg`, or `.png` format  
- ✅ Automatic preprocessing (resize, normalization, tensor conversion)  
- ✅ Deep learning–based prediction: **Real** or **Fake**  
- ✅ Confidence score (e.g., 92.34%)  
- ✅ Simple and clean web interface using Streamlit  

---

## 🏗️ Tech Stack

- **Python**
- **PyTorch** – for model definition and inference
- **torchvision** – transforms and image utilities
- **Streamlit** – web UI for interaction
- **Pillow (PIL)** – image loading and processing

See `requirements.txt` for the minimal dependencies. :contentReference[oaicite:1]{index=1}  

---

## 📁 Project Structure

```text
deepfake-streamlit-app/
│
├── app.py             # Streamlit app – loads model and handles UI & prediction
├── best_model.pth     # Trained CNN model weights
├── requirements.txt   # Python dependencies
└── .gitattributes
