import numpy as np
from spectral import open_image
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize

def save_image_data(image_path, model_path, output_path, beetle_class_code, max_background_points=5000):
    """
    Process a hyperspectral image to classify pixels, filter by specific class,
    and save the spectral data with labels to a CSV file.

    :param image_path: Path to the hyperspectral image (.hdr file)
    :param model_path: Path to the trained model (.pkl file)
    :param output_path: Path to save the output CSV file
    :param beetle_class_code: The numeric code for the beetle class to retain
    :param max_background_points: Maximum number of background points to retain
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

    # Debug print for predicted labels
    print("Predicted labels (flattened):", np.unique(predicted_labels, return_counts=True))

    # Classify non-background pixels as the specified beetle class
    non_background_indices = predicted_labels != -101
    print(f"Number of non-background pixels: {np.sum(non_background_indices)}")
    
    predicted_labels[non_background_indices] = beetle_class_code

    # Debug print for modified predicted labels
    print("Modified predicted labels (flattened):", np.unique(predicted_labels, return_counts=True))

    # Visualize the predicted labels
    visualize_predictions(predicted_labels, data.shape[0], data.shape[1])

    # Filter out all classes except for background and the specified beetle class
    background_indices = np.where(predicted_labels == -101)[0]
    beetle_indices = np.where(predicted_labels == beetle_class_code)[0]

    # Limit the number of background data points
    if background_indices.size > max_background_points:
        background_indices = np.random.choice(background_indices, max_background_points, replace=False)

    # Combine the beetle and sampled background indices
    combined_indices = np.concatenate((background_indices, beetle_indices))

    # Filter the data and labels
    filtered_data = reshaped_data[combined_indices, :]
    filtered_labels = predicted_labels[combined_indices]

    # Debug print for filtered data and labels
    print("Filtered data shape:", filtered_data.shape)
    print("Filtered labels:", np.unique(filtered_labels, return_counts=True))

    # Save the filtered data and labels to a CSV file
    combined_data = np.column_stack((filtered_data, filtered_labels))
    pd.DataFrame(combined_data).to_csv(output_path, header=False, index=False)

    print(f'Data saved successfully to {output_path}')

def visualize_predictions(predicted_labels, height, width):
    """
    Visualize the predicted labels using a color map.

    :param predicted_labels: Array of predicted labels
    :param height: Height of the original image
    :param width: Width of the original image
    """
    # Reshape the predicted labels to match the original image dimensions
    predicted_labels = predicted_labels.reshape((height, width))

    # Define the color map
    cmap = ListedColormap(['red', 'white'])  # Colors for Beetle and Background
    bounds = [(beetle_class_code - .5), -101.5]
    norm = Normalize(vmin=-106, vmax=-101)

    # Create the plot
    plt.figure(figsize=(10, 8))
    plt.imshow(predicted_labels, cmap=cmap, norm=norm)
    plt.colorbar(ticks=[beetle_class_code, -101], format=lambda x, pos: { -101: 'Background', beetle_class_code: 'Beetle'}.get(x, ''))
    plt.title('Predicted Labels')
    plt.show()

# change model, image, csv file paths and corresponding beetle class code
# -101: 'Background', -102: 'CRYPH', -103: 'ERUD', -104: 'AND', -105: 'TNB', -106: 'CBB'

image_path = r'C:\Users\Headwall\Desktop\BeetleClassifier\data\CRYPH\newImage3\data.hdr'
model_path = r'C:\Users\Headwall\Desktop\BeetleClassifier\models\Beetle_Classifier_CDA.pkl'
output_path = r'C:\Users\Headwall\Desktop\PestClassifier\data\processed\CRYPH_data.csv'
beetle_class_code = -102

save_image_data(image_path, model_path, output_path, beetle_class_code)
