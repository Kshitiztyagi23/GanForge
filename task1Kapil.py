from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ========================== MODULE 1: Data Preparation ==========================

# Load the Iris flower dataset
iris_data = load_iris()
features = iris_data.data
labels = iris_data.target

# Split dataset into training and testing sets (80%-20%)
features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.2, random_state=42
)

# Normalize feature data
norm = StandardScaler()
features_train = norm.fit_transform(features_train)
features_test = norm.transform(features_test)

# Apply one-hot encoding to labels
labels_train = to_categorical(labels_train)
labels_test = to_categorical(labels_test)

# Convert numpy arrays into PyTorch tensors
X_train_tensor = torch.tensor(features_train, dtype=torch.float32)
X_test_tensor = torch.tensor(features_test, dtype=torch.float32)
y_train_tensor = torch.tensor(labels_train, dtype=torch.float32)
y_test_tensor = torch.tensor(labels_test, dtype=torch.float32)

# ========================== MODULE 2: Model Architecture ==========================

class IrisClassifier(nn.Module):
    def __init__(self):
        super(IrisClassifier, self).__init__()
        self.layer1 = nn.Linear(4, 8)
        self.activation1 = nn.ReLU()
        self.layer2 = nn.Linear(8, 8)
        self.activation2 = nn.ReLU()
        self.output_layer = nn.Linear(8, 3)

    def forward(self, x):
        x = self.activation1(self.layer1(x))
        x = self.activation2(self.layer2(x))
        x = self.output_layer(x)
        return x

model = IrisClassifier()

# Combine input and label tensors into a dataset and use a dataloader for batching
training_data = TensorDataset(X_train_tensor, y_train_tensor)
data_loader = DataLoader(training_data, batch_size=5, shuffle=True)

# ========================== MODULE 3: Training Process ==========================

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train the model for 100 epochs
for epoch in range(100):
    for inputs, targets in data_loader:
        predictions = model(inputs)
        loss = loss_function(predictions, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# ========================== MODULE 4: Evaluation ==========================

# Evaluate the trained model on the test set
with torch.no_grad():
    test_output = model(X_test_tensor)
    _, predicted_classes = torch.max(test_output, 1)
    true_classes = y_test_tensor.argmax(dim=1)
    accuracy = (predicted_classes == true_classes).float().mean()
    print(f"\nModel Accuracy on Test Set: {accuracy.item() * 100:.2f}%")

# Generate and display confusion matrix
matrix = confusion_matrix(true_classes, predicted_classes)
display = ConfusionMatrixDisplay(confusion_matrix=matrix)
display.plot(cmap='Blues')
plt.title("Model Confusion Matrix")
plt.show()
