import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load trained model
MODEL_PATH = "D:/Janardhan resume/model/rice_leaf_disease_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Define class labels (update as per your dataset)
class_labels = ["Bacterialblight", "Blast", "Brownspot", "Tungro"]

# Streamlit UI
st.title("🌾 Rice Leaf Disease Detection")
st.write("Upload a rice leaf image and the AI will predict the disease!")

# Upload image
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Leaf Image", use_column_width=True)
    
    # Preprocess image for model
    image = image.resize((224, 224))  # Resize to match MobileNetV2 input
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Predict disease
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)  # Get highest probability index
    confidence = np.max(prediction)  # Confidence score

    # Show results
    st.success(f"**Prediction:** {class_labels[predicted_class]}")
    st.info(f"**Confidence Score:** {confidence:.2f}")

    # Add download button (optional)
    st.download_button(
        label="📥 Download Result",
        data=f"Disease: {class_labels[predicted_class]}\nConfidence: {confidence:.2f}",
        file_name="prediction.txt"
    )
