
import os
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))

TEST_DIR = os.path.join(
    PROJECT_DIR, "dataset", "chest_xray", "chest_xray", "test"
)

MODEL_PATH = os.path.join(BASE_DIR, "pneumonia_model_v4.keras")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Load test dataset
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

# Predictions
probabilities = model.predict(test_data)
predictions = (probabilities.ravel() >= 0.5).astype(int)
actual = test_data.classes

# Evaluation
print("\nTest Accuracy:", accuracy_score(actual, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(actual, predictions))

print("\nClassification Report:")
print(
    classification_report(
        actual,
        predictions,
        target_names=["NORMAL", "PNEUMONIA"],
        digits=4
    )
)
