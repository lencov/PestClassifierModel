import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
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
features_path = r'C:\Users\Headwall\Desktop\PestClassifier\data\labeledData\features.csv'
labels_path = r'C:\Users\Headwall\Desktop\PestClassifier\data\labeledData\labels.csv'

features = pd.read_csv(features_path, header=None).transpose().values
labels = pd.read_csv(labels_path, header=None).values.ravel()

logging.info('Features and labels loaded')

# Downsample the background pixels if necessary
background_limit = 5000
background_indices = np.where(labels == -101)[0]
beetle_indices = np.where(labels != -101)[0]

logging.info(f'Number of background pixels: {len(background_indices)}')
logging.info(f'Number of beetle pixels: {len(beetle_indices)}')

if len(background_indices) > background_limit:
    sampled_background_indices = np.random.choice(background_indices, background_limit, replace=False)
    logging.info(f'Sampled {background_limit} background pixels')
else:
    sampled_background_indices = background_indices

# Combine downsampled background data with beetle data
combined_indices = np.concatenate((sampled_background_indices, beetle_indices))
logging.info(f'Total combined indices: {len(combined_indices)}')

# Apply combined indices to features and labels
features = features[combined_indices, :]
labels = labels[combined_indices]

logging.info(f'Shape of features after indexing: {features.shape}')
logging.info(f'Shape of labels after indexing: {labels.shape}')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)

logging.info(f'Training set size: {X_train.shape[0]}')
logging.info(f'Testing set size: {X_test.shape[0]}')

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'solver': ['lsqr', 'eigen'],
    'shrinkage': ['auto', None, 0.1, 0.5, 0.9]  # Adding different shrinkage values for tuning
}

# Initialize the Linear Discriminant Analysis model
lda = LinearDiscriminantAnalysis()

# Use Stratified K-Fold cross-validation to evaluate the model
cv = StratifiedKFold(n_splits=5)

# Set up Grid Search
grid_search = GridSearchCV(estimator=lda, param_grid=param_grid, cv=cv, n_jobs=-1, verbose=2, scoring='accuracy')

# Fit the model
logging.info('Starting grid search...')
grid_search.fit(X_train, y_train)
logging.info('Grid search completed')

# Evaluate the best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Background', 'CRYPH', 'ERUD', 'AND', 'TNB', 'CBB'])

logging.info(f'Best Parameters: {grid_search.best_params_}')
logging.info(f'Test Accuracy: {accuracy}')
logging.info('Test Classification Report:\n' + report)

print(f'Best Parameters: {grid_search.best_params_}')
print(f'Test Accuracy: {accuracy}')
print('Test Classification Report:\n', report)

# Save the trained model
model_save_path = r'C:\Users\Headwall\Desktop\PestClassifier\models\CDA_BeetleClassifierV5.pkl'
joblib.dump(best_model, model_save_path)
logging.info(f'Model saved to {model_save_path}')

# Log the end time
end_time = datetime.now()
logging.info(f'Script completed in {end_time - start_time}')
