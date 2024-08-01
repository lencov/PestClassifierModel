# PestClassifierModel
Machine Learning for classifying beetle species in near-infrared hyperspectral images

Beetle Counting Workflow:

1. Use Headwall to take images of beetles. Make sure to spread them out and try not to let them be touching each other.
2. Move the image folders into your desired root folder
3. Run MultiImageBeetleCounter.py and make sure to check that the root folder path and xlsx file path are correct(also make sure that the xlsx file isn't open in another application as that will cause an access denied error)
4. For each image, a new image with the pixel classifications will appear. Visually check the image to make sure it looks correct and close the image to move onto the next. If there was a mistake in the settings when taking the image, there could be lines of beetle identified pixels running down the image that will cause miscounts.
5. Once you have finished closing all the images, the beetle counts should appear in an xlsx file.

New Model Training Workflow:
1. Use Headwall to take images of desired objects and label the pixels in perClassMira. It is important to accurately label the pixels and to label a sufficient number of pixels. Refer to the BeetleClassifierV2 project in perClassMira to see how labeling should be done. 'Background' should be class 1.
2. Navigate to File > Export  and click on Export Labeled Data to Matlab. Also remember to save the project in PerClassMira.
3. Use the convertMatToCSV script in MatLab to convert the data into Features and Labels csv files which will be used to train the model. Make sure to change the file paths in the script if necessary. If the csv file already exists, then it will not delete any existing data and will just add on to it which could potentially cause issues in the model training.
4. Run trainingV1.py and make sure to change the file paths of the features and labels csv files and the model path if necessary. Also, remember to change the target names to the classes you created in PerClassMira.
5. If necessary, change the class names in the beetle counter scripts to the classes you created in PerClassMira. The labels the models are trained on correspond to the number of the class in PerClassMira. For example, Background is class 1, so its label is -101. TNB is class 2, so its label is -102. So if you change the classes in your training data, you need to change the spreadsheet column labels and the following which are on lines 67-83 in beetleCounter.py and lines 70-86 in MultiImageBeetleCounter.py

# Initialize counts
beetle_counts = {
    'CRYPH': 0,
    'AND': 0,
    'ERUD': 0,
    'TNB': 0,
    'CBB': 0
}

# Mapping from label to beetle name
label_to_beetle = {
    -102: 'TNB',
    -103: 'AND',
    -104: 'ERUD',
    -105: 'CRYPH',
    -106: 'CBB'
}

Possible work to be done:

Some progress was made on eliminating the need for manually labeling data by using the previously trained model to separate the background pixels from an image and classifying every other pixel in that image as the desired class. It appears to work well but the model trained from this data appears to be have some overfitting problems and only worked on the images it was trained on and did not work at all on test images. This could possibly be fixed but some troubleshooting needs to be done to figure out the solution.

Workflow for this method:
1. Take images using Headwall
2. Run extractData.py, changing the class code and file paths if necessary(the files should all be saved in the same directory). Repeat for all desired training images.
3. Run trainingV2.py, changing the data directory and model save paths if necessary
