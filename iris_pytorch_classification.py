from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset
from sklearn.metrics import confusion_matrix , ConfusionMatrixDisplay
import matplotlib.pyplot as plt

#------------------------------------TASK 1----------------------------------------
# loading the dataset
data = load_iris()
X = data.data
y = data.target

# splitting the data into test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# standardizing the dataset
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Using one-hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Making the tensors for the model
# x1 as training X and x2 as testing X similarly for y
x1 = torch.tensor(X_train, dtype=torch.float32)
x2 = torch.tensor(X_test, dtype=torch.float32)
y1 = torch.tensor(y_train, dtype=torch.float32)
y2 = torch.tensor(y_test, dtype=torch.float32)

#---------------------------------------TASK 2--------------------------------------------
#MODEL
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(4, 8)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(8, 8)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(8, 3)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x

model = Net()

train_dataset = TensorDataset(x1, y1)
train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)

#-------------------------------------------TASK 3----------------------------------------------

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


# Training the data
for epoch in range(100):
    for batch_x , batch_y in train_loader:
        out = model(batch_x)      # the output of the model
        loss = criterion(out, batch_y)  # loss due to difference in actual and expected result
        optimizer.zero_grad()     # clears the previous gradients
        loss.backward()   # calculates the new gradients
        optimizer.step()  # Updates the weights based on the gradient


#-------------------------------------------TASK 4-----------------------------------------------

# Testing the data
with torch.no_grad():
    output = model(x2)
    _, predicted = torch.max(output, 1)
    y_true = y2.argmax(dim=1)
    acc = (predicted==y_true).float().mean()
    print(f"\nTest Accuracy: {acc.item() * 100:.2f}%")

cm = confusion_matrix(y_true, predicted)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()