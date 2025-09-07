from fastapi import FastAPI, File, UploadFile
from tensorflow import keras
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = keras.layers.TFSMLayer('../saved_model/1', call_endpoint='serving_default')

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    
    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
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

# Load the model once when the app starts, not per request
MODEL = tf.keras.models.load_model("../saved_model/1")

# Define the classes that the model predicts
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return {"message": "Hello, I am alive!"}

def read_file_as_image(data: bytes) -> np.ndarray:
    """Converts image bytes to a numpy array."""
    with Image.open(BytesIO(data)) as img:
        img = img.convert("RGB")  # Ensure image is in RGB mode
        img = img.resize((224, 224))  # Resize image to fit model input
        return np.array(img)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predicts the class of a potato image."""
    try:
        # Read the uploaded file as an image
        image = read_file_as_image(await file.read())
        
        # Expand dimensions to match the model input shape
        img_batch = np.expand_dims(image, axis=0)

        # Get predictions from the model
        predictions = MODEL.predict(img_batch)

        # Get the predicted class and confidence
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])

        return {"class": predicted_class, "confidence": float(confidence)}
    
    except Exception as e:
        # Handle errors during image processing or prediction
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
