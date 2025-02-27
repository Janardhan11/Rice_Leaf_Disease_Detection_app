import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# Load trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "rice_leaf_disease_model.keras")
model = tf.keras.models.load_model(MODEL_PATH)

# Define class labels
class_labels = ["Bacterial Blight", "Blast", "Brown Spot", "Tungro"]

# Streamlit UI
st.title("🌾 Rice Leaf Disease Detection")
st.write("Upload a rice leaf image and the AI will predict the disease!")

# Upload image
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        # Check file size (Max: 200MB)
        uploaded_file.seek(0, os.SEEK_END)
        file_size = uploaded_file.tell() / (1024 * 1024)  # Convert to MB
        uploaded_file.seek(0)

        if file_size > 200:
            st.error(f"❌ File size is {file_size:.2f}MB. Please upload an image under 200MB.")
        else:
            # Open and display image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Leaf Image", use_container_width=True)

            # Convert to RGB (Fix for grayscale images)
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Preprocess image
            image = image.resize((224, 224))  # Resize for MobileNetV2
            image = np.array(image) / 255.0  # Normalize
            image = np.expand_dims(image, axis=0)  # Add batch dimension

            # Debugging: Print shape
            st.write(f"Image Shape Before Prediction: {image.shape}")

            # Predict disease
            prediction = model.predict(image)
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction)

            # Show results
            st.success(f"🌿 **Prediction:** {class_labels[predicted_class]}")
            st.info(f"📊 **Confidence Score:** {confidence:.2f}")

            # Add download button
            result_text = f"Disease: {class_labels[predicted_class]}\nConfidence: {confidence:.2f}"
            st.download_button(
                label="📥 Download Result",
                data=result_text,
                file_name="prediction.txt"
            )
    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
