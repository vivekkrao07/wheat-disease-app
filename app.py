import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import gdown
import os

# ----------------------------
# Download model from Google Drive
# ----------------------------
if not os.path.exists("wheat_model.h5"):
    url = "https://drive.google.com/uc?id=1ah8F8tAdEuXXXo1tY4csyP8pgilWD_r0"
    gdown.download(url, "wheat_model.h5", quiet=False)

# Load model
model = tf.keras.models.load_model("wheat_model.h5")

# Class names
class_names = ['Healthy', 'Brown_Rust', 'Yellow_Rust']

# UI
st.title("🌾 Wheat Disease Prediction")

uploaded_file = st.file_uploader("Upload Wheat Leaf Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB").resize((224,224))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    predicted_class = predicted_class.replace("_"," ")

    st.subheader(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")

    if "healthy" in predicted_class.lower():
        st.success("The wheat leaf is healthy and shows no signs of disease.")
    else:
        st.error(f"The wheat leaf is affected by {predicted_class} with a confidence of {confidence:.2f}%.")
