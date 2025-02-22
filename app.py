import time

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# --- DARK MODE PAGE CONFIGURATION ---
st.set_page_config(page_title="Face Recognition App Using CNN", layout="wide")

# --- CUSTOM DARK THEME CSS ---
st.markdown(
    """
    <style>
    /* Background & text color */
    .main {
        background-color: #0e1117 !important;
        color: white !important;
    }
    
    /* Title style */
    .stTitle {
        font-size: 32px !important;
        text-align: center;
        color: #f4f4f4;
    }
    
    /* Image styling */
    .stImage {
        border-radius: 15px;
        border: 2px solid #1e90ff;
        box-shadow: 0px 0px 15px rgba(30, 144, 255, 0.5);
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #1e90ff !important;
        color: white !important;
        font-weight: bold;
        border-radius: 8px;
    }
    
    /* File uploader */
    .stFileUploader {
        border: 2px dashed #1e90ff !important;
        padding: 10px;
    }

    /* Sidebar styling */
    .stSidebar {
        background-color: #1c1e26 !important;
        color: white;
    }

    /* Image history section */
    .history-container {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0px 4px 10px rgba(255, 255, 255, 0.2);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- HEADER ---
st.title("🔍 Face Recognition System")
st.markdown("<h4 style='text-align: center; color: #a8a8a8;'>Upload an image to identify the person.</h4>", unsafe_allow_html=True)

# --- LOAD MODEL ---
path = r"C:\Users\USER\Downloads\Team-Member-Classification-using-CNN\CNN_Model.h5"
model = tf.keras.models.load_model(path)

class_names = ['Dewang', 'Gaurav', 'Hitesh', 'Narendra', 'Omkar', 'Pranay', 'Saurabh', 'Vaishnavi']

# --- SESSION STATE FOR HISTORY ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- FILE UPLOAD ---
uploaded_file = st.file_uploader("📸 Upload an Image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # --- DISPLAY IMAGE ---
    image = Image.open(uploaded_file)
    imageUI = image.resize((500,500))
    st.image(imageUI, caption="🖼️ Uploaded Image", output_format="JPEG")

    # --- PREPROCESS IMAGE ---
    img = image.resize((200, 200))  # Resize to match the input size of the model
    img_array = tf.keras.utils.img_to_array(img) 
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension

    # --- LOADING ANIMATION ---
    with st.spinner("⏳ Analyzing the image..."):
        time.sleep(2)  # Simulate processing time

        # --- MAKE PREDICTION ---
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        # --- GET PREDICTED CLASS ---
        predicted_class = class_names[np.argmax(score)]
        confidence = 100 * np.max(score)

    # --- DISPLAY RESULTS ---
    st.success(f"✅ Prediction: **{predicted_class}**")
    st.metric(label="Confidence", value=f"{confidence:.2f} %")

    # --- SHOW BAR CHART OF CONFIDENCES ---
    st.bar_chart({class_names[i]: float(score[i]) for i in range(len(class_names))})

    # --- STORE HISTORY ---
    st.session_state.history.append({"image": image, "name": predicted_class, "confidence": confidence})

# --- IMAGE HISTORY SECTION ---
if st.session_state.history:
    st.markdown("---")
    st.subheader("🕒 History of Recognized Faces")

    cols = st.columns(3)  # Display images in three columns

    for idx, entry in enumerate(reversed(st.session_state.history[-20:])):  # Show last 20 images
        with cols[idx % 3]:  # Arrange in columns
            st.markdown("<div class='history-container'>", unsafe_allow_html=True)
            st.image(entry["image"], caption=f"🧑 {entry['name']} ({entry['confidence']:.2f}%)", use_column_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
