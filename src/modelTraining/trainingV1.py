import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(filename='training_log.txt', level=logging.INFO, format='%(asctime)s %(message)s')

# Log the start time
start_time = datetime.now()
logging.info('Script started')

# Load the features and labels
logging.info('Loading features and labels...')
features_path = r'C:\Users\Headwall\Desktop\BeetleClassifier\labeledData\features.csv'
labels_path = r'C:\Users\Headwall\Desktop\BeetleClassifier\labeledData\labels.csv'

features = pd.read_csv(features_path, header=None).transpose()
labels = pd.read_csv(labels_path, header=None).values.ravel()

logging.info('Features and labels loaded')

# Split the data into training and testing sets
logging.info('Splitting data into training and testing sets...')
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
logging.info('Data split completed')

# Initialize the Canonical Discriminant Analysis
clf = LinearDiscriminantAnalysis()

# Fit the CDA model to the training data
logging.info('Starting model fitting...')
clf.fit(X_train, y_train)
logging.info('Model fitting completed')

# Predict on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Background', 'CRYPH', 'ERUD', 'AND', 'TNB'])

logging.info(f'Accuracy: {accuracy}')
logging.info('Classification Report:')
logging.info(report)

print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)

# Save the trained model in a specific directory
model_save_path = r'C:\Users\Headwall\Desktop\PestClassifier\models\CDA_BeetleClassifierV2.pkl'
joblib.dump(clf, model_save_path)
logging.info(f'Model saved to {model_save_path}')

# Log the end time
end_time = datetime.now()
logging.info(f'Script completed in {end_time - start_time}')