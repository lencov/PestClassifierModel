import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the features and labels
features_path = r'C:\Users\Headwall\Desktop\BeetleClassifier\features.csv'
labels_path = r'C:\Users\Headwall\Desktop\BeetleClassifier\labels.csv'

features = pd.read_csv(features_path, header=None).transpose()
labels = pd.read_csv(labels_path, header=None).values.ravel()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train the Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Background', 'CRYPH', 'ERUD', 'AND', 'TNB'])

print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)

# Save the trained model in a specific directory
model_save_path = r'C:\Users\Headwall\Desktop\BeetleClassifier\random_forest_model.pkl'
joblib.dump(clf, model_save_path)