import pandas as pd
import numpy as np

# Load the features and labels
features_path = r'C:\Users\Headwall\Desktop\BeetleClassifier\trainingData\features.csv'
labels_path = r'C:\Users\Headwall\Desktop\BeetleClassifier\trainingData\labels.csv'

# Read the data
features = pd.read_csv(features_path, header=None)  # Keep features transposed as columns
labels = pd.read_csv(labels_path, header=None, names=['label'])

# Find the minimum class count
min_count = labels['label'].value_counts().min()

# Sample each class to have the same number as the minimum class count
indices_to_keep = []
for label in labels['label'].unique():
    indices_of_class = labels[labels['label'] == label].index
    sampled_indices = np.random.choice(indices_of_class, min_count, replace=False)
    indices_to_keep.extend(sampled_indices)

# Sort the indices to maintain order
indices_to_keep = sorted(indices_to_keep)

# Use the indices to select the balanced data from both features and labels
balanced_features = features.iloc[:, indices_to_keep]
balanced_labels = labels.iloc[indices_to_keep, :]

# Save the balanced data to new CSV files
balanced_features.to_csv(r'C:\Users\Headwall\Desktop\BeetleClassifier\trainingData\balanced_features.csv', index=False, header=False)
balanced_labels.to_csv(r'C:\Users\Headwall\Desktop\BeetleClassifier\trainingData\balanced_labels.csv', index=False, header=False)

print('Balanced dataset saved.')