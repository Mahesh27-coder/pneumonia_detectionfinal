import streamlit as st
import tensorflow as tf
import numpy as np
from keras.preprocessing import image
import gdown
import os

# --- Step 1: Download model from Google Drive if not exists ---
file_id = "1HaeB7vzHUEbIpIRuICOIOpbAbDVNLMBD"  # Your file ID
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

# --- Step 3: Get model input shape automatically ---
# Usually (None, height, width, channels)
input_shape = model.input_shape[1:3]  # e.g., (150, 150)
st.write(f"🔍 Model expects input size: {input_shape}")

# --- Step 4: Custom CSS Styling ---
st.markdown(
    """
    <style>
    /* Background - Uber dark with subtle gradient */
    .stApp {
        background: linear-gradient(135deg, #000000, #0a0a0a, #001100);
        background-attachment: fixed;
        color: #cfcfcf;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Title */
    h1, h2, h3 {
        color: #1db954 !important; /* Dark professional green */
        text-align: center;
        font-weight: 800;
        text-shadow: 0px 0px 10px rgba(0, 80, 0, 0.6);
    }

    /* Subtext */
    p, label {
        color: #aaaaaa !important;
        font-size: 1rem;
    }

    /* File uploader */
    .stFileUploader {
        border: 2px dashed #1db954 !important;
        border-radius: 12px;
        background-color: rgba(10, 10, 10, 0.85);
        padding: 15px;
        box-shadow: 0px 0px 12px rgba(29, 185, 84, 0.3);
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #1db954, #0a3d0a);
        color: white;
        border: none;
        padding: 10px 22px;
        font-size: 1rem;
        border-radius: 6px;
        font-weight: bold;
        box-shadow: 0px 0px 12px rgba(29, 185, 84, 0.4);
        transition: 0.3s ease;
    }

    .stButton>button:hover {
        background: linear-gradient(90deg, #ff2e2e, #1db954); /* Green to Red Uber-like effect */
        box-shadow: 0px 0px 18px rgba(255, 46, 46, 0.6);
    }

    /* Prediction Result Box */
    .result-box {
        font-size: 1.3rem;
        text-align: center;
        margin-top: 20px;
        padding: 15px;
        border-radius: 10px;
        background: rgba(5, 5, 5, 0.95);
        color: #1db954;
        font-weight: bold;
        box-shadow: 0px 0px 15px rgba(29,185,84,0.4);
    }

    /* Accent for Pneumonia (Red Warning) */
    .result-box.pneumonia {
        color: #ff4d4d;
        text-shadow: 0px 0px 10px rgba(255, 50, 50, 0.8);
    }
    </style>
    """,
    unsafe_allow_html=True
)



# --- Step 5: App UI ---
st.title("🩺 Pneumonia Detection from Chest X-Ray")
st.write("Upload a chest X-ray image, and the model will predict if it's **Normal** or **Pneumonia**.")

# File uploader
uploaded_file = st.file_uploader("📤 Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image with the correct target size from model
    img = image.load_img(uploaded_file, target_size=input_shape)
    st.image(img, caption="Uploaded X-ray", use_column_width=True)

    # Preprocess
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Prediction
    prediction = model.predict(img_array)
    result = "🛑 PNEUMONIA" if prediction[0][0] > 0.5 else "✅ NORMAL"

    st.subheader(f"Prediction: **{result}**")
    st.write(f"Confidence: {prediction[0][0]:.2f}")
