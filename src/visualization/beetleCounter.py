import numpy as np
import matplotlib.pyplot as plt
from spectral import open_image
import joblib
from scipy.ndimage import label
from scipy.stats import mode

# Load the trained model
model_save_path = r'C:\Users\Headwall\Desktop\BeetleClassifier\models\Beetle_Classifier_CDA.pkl'
model = joblib.load(model_save_path)

# Load the hyperspectral image
hdr_path = r'C:\Users\Headwall\Desktop\BeetleClassifier\data\TNB\newImage3\data.hdr'
img = open_image(hdr_path)
data = img.load()

# Reshape the data for prediction
num_bands = data.shape[2]
num_pixels = data.shape[0] * data.shape[1]
reshaped_data = data.reshape((num_pixels, num_bands))

# Predict labels for each pixel
predicted_labels = model.predict(reshaped_data)
predicted_labels = predicted_labels.reshape((data.shape[0], data.shape[1]))

# Visualize the predicted labels
plt.imshow(predicted_labels, cmap='jet')
plt.colorbar()
plt.title('Predicted Labels')
plt.show()

# Label connected components
structure = np.ones((3, 3), dtype=np.int32)  # Define connectivity
labeled, num_features = label(predicted_labels != -101, structure=structure)

# Initialize counts
beetle_counts = {
    'CRYPH': 0,
    'ERUD': 0,
    'AND': 0,
    'TNB': 0
}

# Mapping from label to beetle name
label_to_beetle = {
    -102: 'CRYPH',
    -103: 'ERUD',
    -104: 'AND',
    -105: 'TNB'
}

# Process each cluster
for i in range(1, num_features + 1):
    coordinates = np.where(labeled == i)
    if coordinates[0].size > 6:  # Check size of the cluster
        cluster_labels = predicted_labels[coordinates]
        if cluster_labels.size > 0:
            cluster_mode = mode(cluster_labels)
            most_common = cluster_mode.mode if cluster_mode.mode.size > 0 else cluster_mode.mode
            if isinstance(most_common, np.ndarray):
                most_common = most_common.item()  # Convert numpy array to Python scalar if necessary
            count = cluster_mode.count if cluster_mode.count.size > 0 else 'N/A'
            print(f"Cluster {i}: Most common label is {most_common} with count {count}")
            if most_common is not None and most_common in label_to_beetle:
                beetle_name = label_to_beetle[most_common]
                beetle_counts[beetle_name] += 1
        else:
            print(f"Cluster {i}: Cluster labels are empty")

print("Beetle counts:", beetle_counts)

# Save the predicted labels if needed
np.savetxt('predicted_labels.csv', predicted_labels, delimiter=',')
