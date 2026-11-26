from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))

DATA_DIR = os.path.join(
    PROJECT_DIR, "dataset", "chest_xray", "chest_xray"
)

V3_MODEL_PATH = os.path.join(BASE_DIR, "pneumonia_model_v3.keras")
MODEL_PATH = os.path.join(BASE_DIR, "pneumonia_model_v4.keras")

# Training configuration
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# Load and preprocess dataset

# Load and preprocess dataset
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.15
)

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0
)

# Training data (85%)
train_data = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    seed=42
)

# Validation data (15%)
val_data = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    seed=42,
    shuffle=False
)

# Test data (untouched)
test_data = test_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "test"),
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

print("Class mapping:", train_data.class_indices)
print("Training images:", train_data.samples)
print("Validation images:", val_data.samples)
# Calculate class weights to handle dataset imbalance
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_data.classes),
    y=train_data.classes
)

class_weights = dict(enumerate(class_weights))

print("Class weights:", class_weights)


# Load MobileNetV2 pretrained base
# base_model = MobileNetV2(
#     weights="imagenet",
#     include_top=False,
#     input_shape=(224, 224, 3)
# )
# Load the best V3 model
model = tf.keras.models.load_model(V3_MODEL_PATH)

# Find the MobileNetV2 base model
base_model = next(
    layer for layer in model.layers
    if isinstance(layer, tf.keras.Model)
)

# Unfreeze the base model
base_model.trainable = True

# Freeze all but the last 30 layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Keep Batch Normalization layers frozen
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

# Recompile with a very low learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall()
    ]
)

model.summary()

base_model.trainable = False

# Build classification model
model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Precision(),
             tf.keras.metrics.Recall()]
)

model.summary()

# Training callbacks
callbacks = [
    ModelCheckpoint(
        MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        verbose=1
    )
]

# Train
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights
)

# Evaluate on test dataset
print("\nEvaluating on test dataset...")
model = tf.keras.models.load_model(MODEL_PATH)

results = model.evaluate(test_data)
print("Test results:", dict(zip(model.metrics_names, results)))

print("\nTraining complete!")
print("Model saved at:", MODEL_PATH)
