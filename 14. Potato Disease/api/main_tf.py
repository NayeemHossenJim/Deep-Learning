from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import httpx  # Using httpx for async HTTP requests
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf


app = FastAPI()

# CORS middleware for local development
origins = ["http://localhost", "http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# TensorFlow Model Endpoint
endpoint = "http://localhost:8501/v1/models/potatoes_model:predict"

# Class names for predictions
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return {"message": "Hello, I am alive!"}

# Function to convert bytes to image
def read_file_as_image(data: bytes) -> np.ndarray:
    with Image.open(BytesIO(data)) as img:
        return np.array(img)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read and preprocess image
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, axis=0)

    # Prepare payload for prediction
    json_data = {"instances": img_batch.tolist()}

    # Send request to TensorFlow Serving API
    async with httpx.AsyncClient() as client:
        response = await client.post(endpoint, json=json_data)

    # Get prediction result
    prediction = np.array(response.json()["predictions"][0])
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Return prediction response
    return {"class": predicted_class, "confidence": float(confidence)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
