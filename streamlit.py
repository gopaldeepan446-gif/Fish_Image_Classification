import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

MODEL_PATH = "fast_fish_model.h5" 
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at {MODEL_PATH}")
else:
    model = load_model(MODEL_PATH)
    st.success("Model loaded successfully!")

st.title("Fish Image Classifier")
st.write("Upload a fish image and the model will predict its species.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

if uploaded_file:
    img = image.load_img(uploaded_file, target_size=(96,96))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    class_names = list(model.class_names) if hasattr(model, 'class_names') else [
        "FishClass1", "FishClass2", "FishClass3"
    ]

    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.write(f"**Predicted Class:** {class_names[class_idx]}")
    st.write(f"**Confidence:** {confidence:.2f}%")
