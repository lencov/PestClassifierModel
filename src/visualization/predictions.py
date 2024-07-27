import numpy as np
from spectral import open_image
import joblib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
import matplotlib.patches as mpatches

# Load the trained model
model_save_path = r'C:\Users\Headwall\Desktop\BeetleClassifier\models\New_Beetle_Classifier_CDA.pkl'
model = joblib.load(model_save_path)

# Load the hyperspectral image
hdr_path = r'C:\Users\Headwall\Desktop\BeetleClassifier\data\AND\newImage4\data.hdr'
img = open_image(hdr_path)

# Read the hyperspectral data
data = img.load()

# Reshape the data for prediction
num_bands = data.shape[2]
num_pixels = data.shape[0] * data.shape[1]
reshaped_data = data.reshape((num_pixels, num_bands))

# Predict labels for each pixel
predicted_labels = model.predict(reshaped_data)

# Reshape the predicted labels to match the original image dimensions
predicted_labels = predicted_labels.reshape((data.shape[0], data.shape[1]))

# Visualize the predicted labels using a color map
cmap = ListedColormap(['orange','purple', 'red', 'blue', 'green', 'white'])  # Colors for Background, CRYPH, ERUD, AND, TNB
bounds = [-106.5, -105.5, -104.5, -103.5, -102.5, -101.5]
norm = Normalize(vmin=-106, vmax=-101)

plt.imshow(predicted_labels, cmap=cmap, norm=norm)
plt.colorbar(ticks=[-106, -105, -104, -103, -102, -101], format=lambda x, pos: { -101: 'Background', -102: 'CRYPH', -103: 'ERUD', -104: 'AND', -105: 'TNB', -106: 'CBB'}.get(x, ''))
plt.title('Predicted Labels')
plt.show()

# Save the predicted labels if needed
np.savetxt('predicted_labels.csv', predicted_labels, delimiter=',')
