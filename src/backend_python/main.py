import os
import io
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import tensorflow as tf

app = FastAPI(title="PneumoGuard AI - Python Backend")

app.add_middleware(
    CORSMiddleware,
   allow_origins=[
    "http://localhost:3000",
    "https://pneumoguard-ai-l61m.vercel.app"
],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "pneumonia_model_v3.keras")
MODEL = None

try:
    MODEL = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model loaded from: {MODEL_PATH}")
except Exception as e:
    print(f"Model load failed: {e}")


@app.get("/")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Place pneumonia_model_v3.keras in src/backend_python/")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize((224, 224))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = MODEL.predict(img_array)
        probability = float(prediction[0][0])  # sigmoid output

        if probability >= 0.5:
            label = "PNEUMONIA"
            confidence = round(probability * 100, 2)
        else:
            label = "NORMAL"
            confidence = round((1 - probability) * 100, 2)

        return {"prediction": label, "confidence": confidence}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")