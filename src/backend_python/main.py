
import os
import io
import json
import asyncio
import numpy as np
import tensorflow as tf

from dotenv import load_dotenv
from google import genai
from google.genai import types

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError


# Load environment variables
BASE_DIR = os.path.dirname(__file__)
ENV_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "../../.env")
)
load_dotenv(ENV_PATH)


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
    BASE_DIR,
    "pneumonia_model_v3.keras"
)

MODEL = None


# Initialize Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_CLIENT = None

if GEMINI_API_KEY:
    try:
        GEMINI_CLIENT = genai.Client(
            api_key=GEMINI_API_KEY
        )
        print("Gemini client initialized successfully.")

    except Exception as e:
        print(f"Gemini initialization failed: {e}")

else:
    print("GEMINI_API_KEY not configured.")


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


# Generate image-based educational explanation
def generate_gemini_explanation(
    label,
    confidence,
    image_bytes,
    mime_type
):

    if GEMINI_CLIENT is None:
        raise RuntimeError("Gemini API is not configured.")

    prompt = f"""
You are an educational image-analysis assistant
for a student pneumonia classification project.

You are given:
1. A chest X-ray image.
2. The output of a separate MobileNetV2 classifier.

CLASSIFIER OUTPUT:
Prediction: {label}
Confidence score: {confidence}%

Analyze the supplied image carefully for educational purposes.

IMPORTANT SAFETY INSTRUCTIONS:

- Describe only visual characteristics that are reasonably
  visible in the supplied image.
- If image quality, positioning, or visibility is limited,
  explicitly mention that limitation.
- Do not invent abnormalities or findings.
- Do not claim that the image confirms pneumonia.
- Do not treat the classifier's confidence as the probability
  that a patient has a disease.
- Do not claim to know why MobileNetV2 made its prediction.
- Keep your visual observations separate from the
  MobileNetV2 classification output.
- Do not recommend medication or treatment.
- Recommend evaluation by a qualified radiologist or clinician.
- If you cannot reliably interpret a feature, say so.

Return valid JSON only, using this exact structure:

{{
  "reasoning": "A concise educational description of visible
  image characteristics, limitations, and the separate
  MobileNetV2 classification result.",
  "clinicalNotes": "A cautious explanation that this is an
  AI-assisted educational result, not a medical diagnosis,
  and that professional medical evaluation is necessary."
}}

Use clear, understandable language.
Avoid unsupported medical claims.
"""

    image_part = types.Part.from_bytes(
        data=image_bytes,
        mime_type=mime_type
    )

    response = GEMINI_CLIENT.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            prompt,
            image_part
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2
        )
    )

    if not response.text:
        raise ValueError("Gemini returned an empty response.")

    result = json.loads(response.text)

    reasoning = result.get("reasoning")
    clinical_notes = result.get("clinicalNotes")

    if not reasoning or not clinical_notes:
        raise ValueError("Gemini response is incomplete.")

    return {
        "reasoning": reasoning,
        "clinicalNotes": clinical_notes
    }


# Fallback explanation
def get_fallback_explanation(label):

    return {
        "reasoning": (
            f"The MobileNetV2 classification model predicted "
            f"{label}. Image-based explanation is currently "
            "unavailable. This result does not represent "
            "verified radiological findings."
        ),
        "clinicalNotes": (
            "This AI-generated classification is not a medical "
            "diagnosis. Please consult a qualified healthcare "
            "professional for proper medical evaluation."
        )
    }


# Health Check Endpoint
@app.get("/")
def health():

    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "gemini_configured": GEMINI_CLIENT is not None
    }


# Prediction Endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded."
        )

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

        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="Image size must be less than 10 MB."
            )

        # Validate and preprocess image
        image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")

        image = image.resize((224, 224))

        img_array = np.array(image) / 255.0

        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        # MobileNetV2 classification
        prediction = MODEL.predict(
            img_array,
            verbose=0
        )

        probability = float(prediction[0][0])

        if probability >= 0.5:
            label = "PNEUMONIA"
            confidence = round(
                probability * 100,
                2
            )
        else:
            label = "NORMAL"
            confidence = round(
                (1 - probability) * 100,
                2
            )

        # Gemini image analysis
        try:
            explanation = await asyncio.to_thread(
                generate_gemini_explanation,
                label,
                confidence,
                contents,
                file.content_type
            )

            print("Gemini image explanation generated successfully.")

        except Exception as gemini_error:
            print(
                f"Gemini image analysis failed: {gemini_error}"
            )

            explanation = get_fallback_explanation(label)

        return {
            "prediction": label,
            "confidence": confidence,
            "reasoning": explanation["reasoning"],
            "clinicalNotes": explanation["clinicalNotes"]
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
