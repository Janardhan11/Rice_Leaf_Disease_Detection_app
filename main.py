from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
import tensorflow as tf
from PIL import Image
import io

# Initialize FastAPI
app = FastAPI()

# Load the trained model
MODEL_PATH = "D:/Janardhan resume/model/rice_leaf_disease_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Define class labels (Update these according to your dataset)
CLASS_NAMES = ["Bacterialblight", "Blast", "Brownspot", "Tungro"]

@app.get("/")
def home():
    return {"message": "Welcome to the Rice Leaf Disease Detection API!"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read image
    image = await file.read()
    image = Image.open(io.BytesIO(image)).convert("RGB")
    image = image.resize((224, 224))  # Resize to model input size
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(image)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]

    return {"prediction": predicted_class, "confidence": float(np.max(prediction))}

# Run the API
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
