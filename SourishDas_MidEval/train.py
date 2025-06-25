import os
import numpy as np
import matplotlib.pyplot as plt
import subprocess
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Download dataset using Kaggle API if not already present
if not os.path.exists('dataset/annotations') and not os.path.exists('dataset/images'):
    os.makedirs('dataset', exist_ok=True)
    subprocess.run([
        'kaggle', 'datasets', 'download',
        '-d', 'andrewmvd/face-mask-detection',
        '-p', 'dataset', '--unzip'
    ], check=True)
    print("✅ Dataset downloaded and extracted")

# Organize images manually into classes based on filename (simplified split)
# This assumes the image folders were unzipped as 'dataset/images'
# You can refine this part using annotations for bounding boxes

img_size = 128
batch_size = 32
epochs = 10

# Use ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./255, 
    validation_split=0.2
)

train = train_datagen.flow_from_directory(
    'dataset/images',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val = train_datagen.flow_from_directory(
    'dataset/images',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(train, validation_data=val, epochs=epochs)

# Save model
model.save("model/mask_cnn_model.h5")

# Plot accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()
