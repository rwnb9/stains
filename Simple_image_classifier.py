import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# === 1. Parameters ===
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 5  # blood, clean, coffee, mold, various_stains

# === 2. Data Preparation ===
DATASET_PATH = r"C:\Users\shadow\PycharmProjects\nikkkkkk\web_app_2\dataset"

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)

print("📂 Classes:", train_generator.class_indices)

# === 3. Build the Model ===
model = keras.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D(),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(NUM_CLASSES, activation='softmax')  # 5 classes
])

# === 4. Compile the Model ===
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# === 5. Train the Model ===
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator
)

# === 6. Save the Model ===
model.save("my_stains_model.h5")
print("✅ Model saved as my_stains_model.h5")

# === 7. Test on a new image ===


# Load the trained model
model = load_model("my_stains_model.h5")

# Your class order — IMPORTANT: Must match train_generator.class_indices
classes = ['blood', 'clean', 'coffee', 'mold', 'various_stains']

# Load and preprocess the image
img_path = "C:\\Users\\shadow\\PycharmProjects\\nikkkkkk\web_app_2\\223.jpg"  # Replace with path to your test image
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0) / 255.0

# Predict
prediction = model.predict(x)
predicted_class = classes[np.argmax(prediction)]
confidence = np.max(prediction)

print(f"🧠 Predicted: {predicted_class} ({confidence:.2%})")
