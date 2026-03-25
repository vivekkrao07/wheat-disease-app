import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# ----------------------------
# Load Model
# ----------------------------
model = tf.keras.models.load_model("wheat_model.h5")

# Class names (match your dataset)
class_names = ['Healthy', 'Brown_Rust', 'Yellow_Rust']

# ----------------------------
# App UI
# ----------------------------
st.set_page_config(page_title="Wheat Disease Predictor", page_icon="🌾")

st.title("🌾 Wheat Disease Prediction System")
st.write("Upload a wheat leaf image to detect disease.")

# ----------------------------
# File Upload
# ----------------------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    
    # Load image
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((224, 224))
    
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Clean label
    predicted_class_clean = predicted_class.replace("_", " ")

    # ----------------------------
    # Output Section
    # ----------------------------
    st.subheader("🔍 Prediction Result")
    st.write(f"**Disease:** {predicted_class_clean}")
    st.write(f"**Confidence:** {confidence:.2f}%")

    # Final conclusion line
    if "healthy" in predicted_class_clean.lower():
        st.success("The wheat leaf is healthy and shows no signs of disease.")
    else:
        st.error(f"The wheat leaf is affected by {predicted_class_clean} with a confidence of {confidence:.2f}%.")

    # ----------------------------
    # Optional: Show Probabilities
    # ----------------------------
    st.subheader("📊 Class Probabilities")
    for i, prob in enumerate(prediction[0]):
        st.write(f"{class_names[i].replace('_',' ')}: {prob*100:.2f}%")
