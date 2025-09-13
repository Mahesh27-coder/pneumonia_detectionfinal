import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# --- CSS Styling ---
st.markdown(
    """
    <style>
    /* Background gradient */
    .stApp {
        background: linear-gradient(135deg, #1d2b64, #f8cdda);
        color: white;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Title styling */
    h1 {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 700;
        color: #ffffff;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.4);
        margin-bottom: 1rem;
    }

    /* Upload box styling */
    .stFileUploader {
        border: 2px dashed #fff !important;
        border-radius: 15px;
        padding: 20px;
        background-color: rgba(255,255,255,0.1);
        margin-bottom: 1.5rem;
    }

    /* Buttons */
    button[kind="primary"] {
        background: linear-gradient(90deg, #ff512f, #dd2476);
        color: white !important;
        font-weight: bold;
        border-radius: 12px;
        padding: 0.6rem 1.2rem;
        transition: 0.3s;
    }

    button[kind="primary"]:hover {
        transform: scale(1.05);
        background: linear-gradient(90deg, #dd2476, #ff512f);
    }

    /* Uploaded image preview */
    img {
        border-radius: 20px;
        box-shadow: 0px 8px 20px rgba(0,0,0,0.3);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Title ---
st.title("🩺 Pneumonia Detection from Chest X-ray")

# --- Download model from Google Drive ---
model_path = "pneumonia_model.h5"
if not os.path.exists(model_path):
    with st.spinner("Downloading model... Please wait."):
        gdown.download(
            "https://drive.google.com/uc?id=1fN_WBaN6z8cnT2Yg4w1lwHgKk9sTprls",
            model_path,
            quiet=False
        )

# Load model
model = tf.keras.models.load_model(model_path)

# --- File uploader ---
uploaded_file = st.file_uploader("📂 Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    result = "Pneumonia Detected 🛑" if prediction[0][0] > 0.5 else "Normal ✅"

    st.subheader("🔎 Prediction Result:")
    st.success(result if result == "Normal ✅" else result)
