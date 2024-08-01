import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import glob
import logging
from datetime import datetime

# Function to augment data
def augment_data(features, labels):
    augmented_features = []
    augmented_labels = []

    for i in range(features.shape[0]):
        feature = features[i, :]
        label = labels[i]
        
        # Original
        augmented_features.append(feature)
        augmented_labels.append(label)
        
        # Brightness adjustment
        for adjustment in [0.9, 1.1]:
            augmented_features.append(feature * adjustment)
            augmented_labels.append(label)
        
        # Noise addition
        noise = np.random.normal(0, 0.01, feature.shape)
        augmented_features.append(feature + noise)
        augmented_labels.append(label)
    
    return np.array(augmented_features), np.array(augmented_labels)

# Start time for logging
start_time = datetime.now()

# Set up logging
logging.basicConfig(filename='training_log.txt', level=logging.INFO, format='%(asctime)s %(message)s')
logging.info('Script started')

# Define the path to the training data
data_directory = r'C:\Users\Headwall\Desktop\PestClassifier\data\processed'
data_files = glob.glob(data_directory + '/*.csv')

# Combine all CSV files into one DataFrame
full_dataset = pd.DataFrame()
for file in data_files:
    temp_df = pd.read_csv(file, header=None)
    full_dataset = pd.concat([full_dataset, temp_df], axis=0, ignore_index=True)

# Separate the dataset into features and labels
features = full_dataset.iloc[:, :-1].values
labels = full_dataset.iloc[:, -1].values

# Downsample the background pixels if necessary
background_limit = 5000
background_data = full_dataset[full_dataset.iloc[:, -1] == -101]  # Assuming -101 is the background code
beetle_data = full_dataset[full_dataset.iloc[:, -1] != -101]

if len(background_data) > background_limit:
    background_data = background_data.sample(n=background_limit, random_state=42)

# Combine downsampled background data with beetle data
combined_data = pd.concat([background_data, beetle_data], ignore_index=True)
features = combined_data.iloc[:, :-1].values
labels = combined_data.iloc[:, -1].values

# Augment the data
features, labels = augment_data(features, labels)

# Convert data to PyTorch tensors and create a dataset
features = torch.tensor(features.reshape(-1, 1, 13, 13).astype(np.float32))
labels = torch.tensor((labels + 106).astype(np.int64))

dataset = TensorDataset(features, labels)

# Split the dataset into training and testing sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 3 * 3, 64)
        self.fc2 = nn.Linear(64, 6)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 3 * 3)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNN()

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f'Test accuracy: {correct / total}')

# Save the model
model_save_path = r'C:\Users\Headwall\Desktop\PestClassifier\models\CNN_BeetleClassifier.pth'
torch.save(model.state_dict(), model_save_path)
logging.info(f'Model saved to {model_save_path}')

# Log the end time
end_time = datetime.now()
logging.info(f'Script completed in {end_time - start_time}')
