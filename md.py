import os
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import xml.etree.ElementTree as ET

# Set paths
IMAGE_DIR = 'face-mask-detection/images'
ANNOTATION_DIR = 'face-mask-detection/annotations'

# Define class labels
LABELS = ['without_mask', 'with_mask', 'mask_weared_incorrect']

def extract_data():
    data = []
    labels = []
    
    for xml_file in os.listdir(ANNOTATION_DIR):
        if not xml_file.endswith('.xml'):
            continue
        
        tree = ET.parse(os.path.join(ANNOTATION_DIR, xml_file))
        root = tree.getroot()
        
        filename = root.find('filename').text
        img_path = os.path.join(IMAGE_DIR, filename)
        image = cv2.imread(img_path)
        if image is None:
            continue
        image = cv2.resize(image, (100, 100))
        
        label_found = False
        for obj in root.findall('object'):
            label = obj.find('name').text
            if label in LABELS:
                data.append(image)
                labels.append(LABELS.index(label))
                label_found = True
                break  # One label per image (most prominent one)

    return np.array(data), np.array(labels)

# Load dataset
X, y = extract_data()
X = X / 255.0  # Normalize
y_cat = to_categorical(y, num_classes=3)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(100,100,3)),
    MaxPooling2D(2,2),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')
])

# Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.2f}")
