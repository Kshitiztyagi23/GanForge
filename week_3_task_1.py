from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# One-hot encoding
y_train_encoded = to_categorical(y_train, num_classes=3)
y_test_encoded = to_categorical(y_test, num_classes=3)

# Task 2 - Neural Network Construction
model = Sequential()
model.add(Input(shape=(4,)))                # Explicit input layer, no warning
model.add(Dense(8, activation='relu'))      # Hidden layer
model.add(Dense(3, activation='softmax'))   # Output layer
# model = Sequential()

# Input Layer: Corresponding to the 4 input features
# Hidden Layer: 8 neurons with ReLU activation function
# model.add(Dense(8, input_shape=(4,), activation='relu', name='hidden_layer'))

# Output Layer: 3 neurons (one for each Iris species) with softmax activation
# model.add(Dense(3, activation='softmax', name='output_layer'))

print("Neural network architecture:")
print(model.summary())


# Task 3 - Model Compilation and Training
model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_scaled, y_train_encoded, epochs=100, batch_size=5, verbose=1, validation_split=0.1)

# Task 4 - Model Evaluation
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test_encoded, verbose=0)
print(f"\nFinal Test Set Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
