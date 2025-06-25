from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# ============================ TASK 1: Data Preparation ============================

iris = load_iris()
features = iris.data
labels = iris.target

# Split into training and testing sets
X_tr, X_te, y_tr, y_te = train_test_split(features, labels, test_size=0.2, random_state=42)

# Standardize features
std_scaler = StandardScaler()
X_tr = std_scaler.fit_transform(X_tr)
X_te = std_scaler.transform(X_te)

# One-hot encode labels
y_tr_cat = to_categorical(y_tr)
y_te_cat = to_categorical(y_te)

# Convert to torch tensors
X_tr_tensor = torch.tensor(X_tr, dtype=torch.float32)
X_te_tensor = torch.tensor(X_te, dtype=torch.float32)
y_tr_tensor = torch.tensor(y_tr_cat, dtype=torch.float32)
y_te_tensor = torch.tensor(y_te_cat, dtype=torch.float32)

# ============================ TASK 2: Define Model ============================

class IrisClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 3)
        )

    def forward(self, x):
        return self.network(x)

model = IrisClassifier()

train_data = TensorDataset(X_tr_tensor, y_tr_tensor)
train_loader = DataLoader(train_data, batch_size=5, shuffle=True)

# ============================ TASK 3: Training ============================

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

def train_model(model, loader, loss_fn, optimizer, epochs=100):
    model.train()
    for ep in range(epochs):
        for inputs, targets in loader:
            preds = model(inputs)
            loss = loss_fn(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

train_model(model, train_loader, loss_fn, optimizer)

# ============================ TASK 4: Evaluation ============================

model.eval()
with torch.no_grad():
    test_logits = model(X_te_tensor)
    predicted_labels = torch.argmax(test_logits, dim=1)
    true_labels = torch.argmax(y_te_tensor, dim=1)
    accuracy = (predicted_labels == true_labels).float().mean()

print(f"\nTest Accuracy: {accuracy.item() * 100:.2f}%")

# Confusion Matrix
cmatrix = confusion_matrix(true_labels, predicted_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cmatrix)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()
