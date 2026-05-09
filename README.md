# 🌤 Weather Image Recognition System
A deep learning-based web application that automatically classifies weather conditions from images using EfficientNetB0 transfer learning, achieving over 90% validation accuracy across 11 weather classes.

# 📌 Overview
This project builds an end-to-end intelligent system for weather image recognition — from raw data exploration and model training to a fully functional Flask web application where users can upload any weather image and receive an instant prediction with confidence scores.

# 📂 Dataset

Source: Weather Image Recognition — Kaggle
Total Images: 6,862
Classes (11): Dew Fog/Smog Frost Glaze Hail Lightning Rain Rainbow Rime Sandstorm Snow
Image sizes: Vary from 117×91 to 4863×3174 — all resized to 300×300 during preprocessing


# 🔬 Exploratory Data Analysis (EDA)

Analyzed the distribution of image dimensions across the full dataset
Identified class imbalance across the 11 weather categories
Used findings to inform preprocessing and augmentation strategy


# ⚙️ Preprocessing & Augmentation

Resize   All images resized to 300×300  || 
Normalization    EfficientNetB0.preprocess_input (expects 0–255, handles normalization internally) || 
Rotation        ±10°  || 
Zoom            20%  || 
Width/Height Shift   10%  || 
Shear              10%  || 
Horizontal Flip    ✅ Yes (no vertical flip — rain/snow falls downward)  || 
BrightnessRange    [0.8, 1.2]  ||  
Validation Split   20% (clean generator — no augmentation)  ||  
Class Weights       Computed via sklearn.utils.class_weight.compute_class_weight to handle imbalance  || 


# 🧠 Model Architecture
Base Model

EfficientNetB0 pretrained on ImageNet (include_top=False)
Base frozen during Phase 1 training

Classification Head
GlobalAveragePooling2D
BatchNormalization
Dense(256, activation='relu')
Dropout(0.4)
Dense(128, activation='relu')
Dropout(0.3)
Dense(11, activation='softmax')

# 🏋️ Training Strategy

Phase 1 — Head Training (Base Frozen)

Optimizer: Adam(lr=1e-3)
Loss: categorical_crossentropy
Epochs: up to 20 (with Early Stopping)
Callbacks: EarlyStopping(monitor='val_accuracy', patience=5) + ModelCheckpoint


# Model Saving

Best model saved automatically via ModelCheckpoint
On subsequent runs, model is loaded from disk — no retraining needed


# 📊 Results

Validation Accuracy    +90%  ||  
Number of Classes      11    || 
Training Images       5,493   || 
Validation Images     1,369   || 

# 🌐 Web Application
Built with Flask — a clean, dark-mode interface where users can:

Upload any weather image (jpg, png, webp, bmp)
Drag and drop support
View the predicted weather class with emoji
See confidence percentage with animated bar
Browse probability scores for all 11 classes

Run Locally
bashpip install flask tensorflow pillow
python flask_app.py
Then open: http://localhost:5000

# 📁 Project Structure
Weather-Recognition-System/
├── flask_app.py                              # Flask backend
├── Weather image recognition system.ipynb   # Training notebook
├── templates/
│   └── index.html                           # Frontend UI
├── static/                                  # Static assets
└── README.md

Note: The trained model file (weather_best.keras) is not included due to size. Train the model using the notebook and it will be saved automatically.


# 🛠️ Tech Stack

Category         ||   Tools
------------------------------------------------------------
Language         ||   Python
Deep Learning    ||   TensorFlow, Keras
Model            ||   EfficientNetB0 (Transfer Learning)
Data Processing  ||   NumPy, Pandas
ML Utilities     ||   Scikit-learn
Web Framework    ||   Flask
Frontend         ||   HTML, CSS, JavaScript

# 🚀 How to Use

Clone the repository

bashgit clone https://github.com/YOUR_USERNAME/Weather-Recognition-System.git

Install dependencies

bashpip install flask tensorflow pillow scikit-learn numpy pandas

Download the dataset from Kaggle and train the model using the notebook
Run the Flask app

bashpython flask_app.py

Open http://localhost:5000 and upload a weather image
