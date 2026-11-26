
import os
import io
import numpy as np
import tensorflow as tf

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

# Initialize FastAPI
app = FastAPI(title="PneumoGuard AI - Python Backend")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "https://pneumoguard-ai-l61m.vercel.app"
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model Configuration
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "pneumonia_model_v3.keras"
)

MODEL = None

# Load trained model
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    MODEL = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model loaded successfully: {MODEL_PATH}")

except Exception as e:
    print(f"Model loading failed: {e}")


# Health Check Endpoint
@app.get("/")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None
    }


# Prediction Endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded."
        )

    # Validate image type
    allowed_types = [
        "image/jpeg",
        "image/png"
    ]

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Only JPG and PNG images are supported."
        )

    try:
        contents = await file.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty."
            )

        # Open and preprocess image
        image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")

        image = image.resize((224, 224))

        img_array = np.array(image) / 255.0

        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        # Model prediction
        prediction = MODEL.predict(
            img_array,
            verbose=0
        )

        probability = float(prediction[0][0])

        if probability >= 0.5:
            label = "PNEUMONIA"
            confidence = round(probability * 100, 2)
        else:
            label = "NORMAL"
            confidence = round(
                (1 - probability) * 100,
                2
            )

        return {
            "prediction": label,
            "confidence": confidence
        }

    except HTTPException:
        raise

    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file."
        )

    except Exception as e:
        print(f"Prediction error: {e}")

        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Please try again."
        )

    finally:
        await file.close()
