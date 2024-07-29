import numpy as np
from spectral import open_image
import joblib
import pandas as pd

def save_image_data(image_path, model_path, output_path, beetle_class_code):
    """
    Process a hyperspectral image to classify pixels, filter by specific class,
    and save the spectral data with labels to a CSV file.

    :param image_path: Path to the hyperspectral image (.hdr file)
    :param model_path: Path to the trained model (.pkl file)
    :param output_path: Path to save the output CSV file
    :param beetle_class_code: The numeric code for the beetle class to retain
    """
    # Load the trained model
    model = joblib.load(model_path)

    # Load the hyperspectral image
    img = open_image(image_path)

    # Read the hyperspectral data
    data = img.load()

    # Reshape the data for prediction
    num_bands = data.shape[2]
    num_pixels = data.shape[0] * data.shape[1]
    reshaped_data = data.reshape((num_pixels, num_bands))

    # Predict labels for each pixel
    predicted_labels = model.predict(reshaped_data)

    # Filter out all classes except for background and the specified beetle class
    indices = (predicted_labels == -101) | (predicted_labels == beetle_class_code)
    filtered_data = reshaped_data[indices, :]
    filtered_labels = predicted_labels[indices]

    # Save the filtered data and labels to a CSV file
    combined_data = np.column_stack((filtered_data, filtered_labels))
    pd.DataFrame(combined_data).to_csv(output_path, header=False, index=False)

    print(f'Data saved successfully to {output_path}')

# Example usage
image_path = r'C:\Users\Headwall\Desktop\BeetleClassifier\data\CBB\data.hdr'
model_path = r'C:\Users\Headwall\Desktop\BeetleClassifier\models\Beetle_Classifier_CDA.pkl'
output_path = r'C:\Users\Headwall\Desktop\BeetleClassifier\trainingData\CBB_data.csv'
beetle_class_code = -106

save_image_data(image_path, model_path, output_path, beetle_class_code)