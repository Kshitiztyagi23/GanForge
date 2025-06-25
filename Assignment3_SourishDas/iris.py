# Import necessary libraries
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score


# Task 1 - Data Loading and Preprocessing

# Load the dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Labels

# Split the data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling (standardization)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# One-hot encode the target labels
y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)


# Task 2 - Neural Network Construction

model = Sequential()
model.add(Dense(8, activation='relu', input_shape=(4,)))  # Hidden layer with 8 neurons
model.add(Dense(3, activation='softmax'))  # Output layer for 3 classes


# Task 3 - Model Compilation and Training

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train_encoded, epochs=100, batch_size=5, verbose=0)


# Task 4 - Model Evaluation

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test_encoded, verbose=0)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
