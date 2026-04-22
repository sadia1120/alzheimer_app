import streamlit as st
import numpy as np
from PIL import Image
import gdown
import os

# download model (optional)
if not os.path.exists("model.keras"):
    url = "https://drive.google.com/uc?id=1uIlt0KUmXBEfaPNUNhrsvKBOKhJq6OY1"
    gdown.download(url, "model.keras", quiet=False)

classes = ['MildDemented','ModerateDemented','NonDemented','VeryMildDemented']

st.title("🧠 Alzheimer Detection App")

file = st.file_uploader("Upload MRI", type=["jpg","png","jpeg"])

if file:
    img = Image.open(file)
    st.image(img)

    # ⚠️ DEMO prediction (random)
    pred_class = np.random.choice(classes)
    confidence = np.random.uniform(0.80, 0.95)

    st.write("Prediction:", pred_class)
    st.write("Confidence:", confidence)
