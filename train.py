"""
CropWise AI – Training Script
Trains a MobileNetV2-based crop disease classifier.
"""
import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

DATA_DIR   = "dataset_reduced"
MODEL_DIR  = "model"
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
EPOCHS     = 20

# ── Data generators ──
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    zoom_range=0.1,
    rotation_range=15,
    brightness_range=[0.8, 1.2]
)

train_data = datagen.flow_from_directory(
    DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, subset="training")

val_data = datagen.flow_from_directory(
    DATA_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, subset="validation")

num_classes = train_data.num_classes
print(f"✅  Classes found: {num_classes}")

# ── Model ──
base = tf.keras.applications.MobileNetV2(
    input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet")
base.trainable = False

x = base.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.3)(x)
output = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs=base.input, outputs=output)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# ── Callbacks ──
os.makedirs(MODEL_DIR, exist_ok=True)

callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(patience=2, factor=0.5, verbose=1),
    ModelCheckpoint(f"{MODEL_DIR}/best_model.keras", save_best_only=True, verbose=1)
]

# ── Train ──
history = model.fit(train_data, validation_data=val_data, epochs=EPOCHS, callbacks=callbacks)

# ── Save ──
model.save(f"{MODEL_DIR}/crop_disease_model.h5")

# Save class names
with open(f"{MODEL_DIR}/classes.txt", "w") as f:
    for cls in train_data.class_indices.keys():
        f.write(cls + "\n")

# Save training history
with open(f"{MODEL_DIR}/training_history.json", "w") as f:
    json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)

print("✅  Model saved to model/crop_disease_model.h5")
print("✅  Classes saved to model/classes.txt")