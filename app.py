import streamlit as st
import tensorflow as tf
import numpy as np
from keras.preprocessing import image
import gdown
import os

# --- Custom CSS Styling ---
st.markdown("""
    <style>
    .stApp {
        background-color: #f8fafc;
        font-family: "Segoe UI", sans-serif;
    }
    h1 {
        color: #0f172a;
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
    }
    h2, h3 {
        color: #334155;
        font-weight: 600;
    }
    .stFileUploader {
        border: 2px dashed #3b82f6;
        background-color: #eff6ff;
        padding: 20px;
        border-radius: 12px;
    }
    .stButton button {
        background-color: #3b82f6;
        color: white;
        border-radius: 12px;
        padding: 0.6em 1.2em;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton button:hover {
        background-color: #2563eb;
        transform: scale(1.05);
    }
    .prediction-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        text-align: center;
        margin-top: 20px;
    }
    .normal {
        color: #16a34a;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .pneumonia {
        color: #dc2626;
        font-size: 1.5rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# --- Step 1: Download model from Google Drive if not exists ---
file_id = "1HaeB7vzHUEbIpIRuICOIOpbAbDVNLMBD"  # Your model file ID
url = f"https://drive.google.com/uc?id={file_id}"
model_path = "pneumonia_detector.h5"

if not os.path.exists(model_path):
    with st.spinner("📥 Downloading model... Please wait."):
        gdown.download(url, model_path, quiet=False)

# --- Step 2: Load Model ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(model_path)

model = load_model()

# --- Step 3: App UI ---
st.title("🩺 Pneumonia Detection from Chest X-Ray")
st.write("Upload a chest X-ray image, and the model will predict if it's **Normal** or **Pneumonia**.")

# File uploader
uploaded_file = st.file_uploader("📂 Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    img = image.load_img(uploaded_file, target_size=(224, 224))
    st.image(img, caption="Uploaded X-ray", use_column_width=True)

    # Preprocess
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Prediction
    prediction = model.predict(img_array)
    prob = float(prediction[0][0])
    result = "PNEUMONIA" if prob > 0.5 else "NORMAL"

    # Display prediction with styled box
    if result == "NORMAL":
        st.markdown(f"<div class='prediction-box'><p class='normal'>Prediction: {result}</p></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='prediction-box'><p class='pneumonia'>Prediction: {result}</p></div>", unsafe_allow_html=True)

    # Confidence bar
    st.write("### 📊 Confidence")
    st.progress(prob if result == "PNEUMONIA" else 1 - prob)
    st.write(f"**Probability (Pneumonia): {prob:.2f}**")
