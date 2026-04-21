import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image

import gdown
import os
import tensorflow as tf

if not os.path.exists("model.keras"):
    url = "https://drive.google.com/uc?id=1uIlt0KUmXBEfaPNUNhrsvKBOKhJq6OY1"
    gdown.download(url, "model.keras", quiet=False)

model = tf.keras.models.load_model("model.keras", compile=False)

classes = ['MildDemented','ModerateDemented','NonDemented','VeryMildDemented']

st.title("🧠 Alzheimer Detection App")

file = st.file_uploader("Upload MRI", type=["jpg","png","jpeg"])

if file:
    img = Image.open(file)
    st.image(img)

    img = img.resize((128,128))
    img = image.img_to_array(img)/255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    st.write("Prediction:", classes[np.argmax(pred)])
    st.write("Confidence:", np.max(pred))
