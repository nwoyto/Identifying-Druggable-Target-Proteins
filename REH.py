import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

df = pd.read_csv('DataFiles/ImportantFeatures.csv')
df.head()

X = df.iloc[:, 1:]
y = df['DTP'] #DTP is 0s and 1s

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


# Convert the training set and test set to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train.values).float()
X_test_tensor = torch.from_numpy(X_test.values).float()
y_train_tensor = torch.from_numpy(y_train.values).long()
y_test_tensor = torch.from_numpy(y_test.values).long()



# Convert the training set and test set to PyTorch datasets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)


# Create data loaders for the training set and test set
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, hidden_size_3, 
                                hidden_size_4, hidden_size_5, output_size):
        super(MLP, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.fc3 = nn.Linear(hidden_size_2, hidden_size_3)
        self.fc4 = nn.Linear(hidden_size_3, hidden_size_4)
        self.fc5 = nn.Linear(hidden_size_4, hidden_size_5)
        self.fc6 = nn.Linear(hidden_size_5, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.fc6(x)
        return x

# Initialize the model
input_size = 59
hidden_size_1 = 128
hidden_size_2 = 64
hidden_size_3 = 32
hidden_size_4 = 16
hidden_size_5 = 8
output_size = 1
model = MLP(input_size, hidden_size_1, hidden_size_2, hidden_size_3, hidden_size_4, hidden_size_5, output_size)

# Define the loss function and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())

def train_mlp(train_dataloader, test_dataloader, input_size, output_size, hidden_sizes, n_epochs):
    # Define the model architecture
    layers = []
    layers.append(nn.Linear(input_size, hidden_sizes[0]))
    layers.append(nn.ReLU())
    for i in range(1, len(hidden_sizes)):
        layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(hidden_sizes[-1], output_size))
    layers.append(nn.LogSoftmax(dim=1))
    model = nn.Sequential(*layers)
    # Define the loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    for epoch in range(n_epochs):
        for inputs, labels in train_dataloader:
            optimizer.zero_grad()
            log_ps = model(inputs)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
   
    return model

n_epochs = 10
hidden_sizes = [128, 64, 32, 16, 8]
trained_model = train_mlp(train_dataloader, test_dataloader, input_size, output_size, hidden_sizes, n_epochs)