import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import glob
import logging
from datetime import datetime

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
features = full_dataset.iloc[:, :-1]
labels = full_dataset.iloc[:, -1]

# Downsample the background pixels if necessary
background_limit = 5000
background_data = full_dataset[full_dataset.iloc[:, -1] == -101]  # Assuming -101 is the background code
beetle_data = full_dataset[full_dataset.iloc[:, -1] != -101]

if len(background_data) > background_limit:
    background_data = background_data.sample(n=background_limit, random_state=42)

# Combine downsampled background data with beetle data
combined_data = pd.concat([background_data, beetle_data], ignore_index=True)
features = combined_data.iloc[:, :-1]
labels = combined_data.iloc[:, -1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Initialize the Canonical Discriminant Analysis
clf = LinearDiscriminantAnalysis()

# Fit the CDA model to the training data
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

logging.info(f'Accuracy: {accuracy}')
logging.info('Classification Report:\n' + report)

print(f'Accuracy: {accuracy}')
print('Classification Report:\n', report)

# Save the trained model
model_save_path = r'C:\Users\Headwall\Desktop\PestClassifier\models\LDA_BeetleClassifierV1_with_CBB.pkl'
joblib.dump(clf, model_save_path)
logging.info(f'Model saved to {model_save_path}')

# Log the end time
end_time = datetime.now()
logging.info(f'Script completed in {end_time - start_time}')
