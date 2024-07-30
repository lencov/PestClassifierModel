import numpy as np
import matplotlib.pyplot as plt
from spectral import open_image
import joblib
from scipy.ndimage import label
from scipy.stats import mode
import pandas as pd
import os

def update_spreadsheet(file_path, new_entry):
    """
    Update the spreadsheet with the new beetle counts and file path.

    :param file_path: Path to the spreadsheet
    :param new_entry: Dictionary containing the beetle counts and file path
    """
    # Check if the spreadsheet exists
    if os.path.exists(file_path):
        df = pd.read_excel(file_path, index_col=0)
    else:
        # Create a new DataFrame if the spreadsheet does not exist
        columns = ['CRYPH', 'ERUD', 'AND', 'TNB', 'CBB', 'File Path']
        df = pd.DataFrame(columns=columns)

    # Append the new entry to the DataFrame
    new_entry_df = pd.DataFrame([new_entry])
    df = pd.concat([df, new_entry_df], ignore_index=True)

    # Calculate the total counts
    total_counts = df.sum(numeric_only=True)
    total_counts['File Path'] = 'Total'

    # Insert the total counts at the top of the DataFrame
    df = pd.concat([total_counts.to_frame().T, df], ignore_index=True)

    # Save the updated DataFrame to the spreadsheet
    df.to_excel(file_path)

# Load the trained model
model_save_path = r'C:\Users\Headwall\Desktop\PestClassifier\models\LDA_BeetleClassifierV1_with_CBB.pkl'
model = joblib.load(model_save_path)

# Load the hyperspectral image. Make sure that within the same directory, there exists the data binary files. The other files might be required as well so just make sure all the files you get from taking the image are present
hdr_path = r'C:\Users\Headwall\Desktop\PestClassifier\data\raw\CBB\data.hdr'
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
    'TNB': 0,
    'CBB': 0
}

# Mapping from label to beetle name
label_to_beetle = {
    -102: 'CRYPH',
    -103: 'ERUD',
    -104: 'AND',
    -105: 'TNB',
    -106: 'CBB'
}

# Process each cluster
for i in range(1, num_features + 1):
    coordinates = np.where(labeled == i)
    if coordinates[0].size > 10:  # Check size of the cluster. Adjust this number to change the minimum pixel cluster size
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

# Prepare the new entry for the spreadsheet
new_entry = beetle_counts.copy()
new_entry['File Path'] = hdr_path

# Update the spreadsheet
spreadsheet_path = r'C:\Users\Headwall\Desktop\PestClassifier\beetle_counts.xlsx'
update_spreadsheet(spreadsheet_path, new_entry)
