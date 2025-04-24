# ML-Emotion-Recognition-Project
Program for emotion recognition through convolutional neural networks

Emotion Recognition using Computer Vision by John Li


Description:
Program to identify facial emotions from images or videos using Convoluted Neural Networks
Desired output is accurate depiction of emotion and ability to differentiate between different facial expressions


Datasets:
Face expression recognition dataset from Kaggle by Jonathan Oheix
https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset/data 


Model and Method Overview
2 convolutional layers each followed by BatchNormalization, MaxPooling2D, and Dropout. Each layer has 128 filters, kernel size of 3x3 ReLU activation, and surface convolutional layer has input shape of 48x48x1. Pool sizes for MaxPooling2D layers is 2x2.

following convlutional layers is a flatten layer and 2 Dense layers. Surface Dense layer has 512 units and hidden Dense layer has 256 units. Both layers have ReLU activation and are coupled with a Dropout layer. 

All Dropout layers for convolutional and dense layers are at 0.5.

the output layer is a Dense layer with 7 units and softmax activation.

model is compiled with adam optimizer, categorical_crossentropy loss, and accuracy metrics.

The model is fitted with x_train and y_train data with batch_size of 128 and 100 epochs. x_test and y_test are used for validation.


Instructions:
1. install opendatasets using pip install
2. import opendataset as od, and download the Kaggle dataset listed above using od.download(link_to_kaggle)
3. downloading the dataset requires a username and key, if you already possess a kaggle key skip to step 8
4. log into kaggle or create new account
5. navigate to the settings page by clicking the profile tab on the top right
6. scroll down to API section and click create new token
7. this will download a file containing the username and key, to access open the file using a text file like notepad
8. when prompted enter the username and key from Kaggle
9. import shutil and delete extra redundant folder using shutil.rmtree. The location for the redundant folder is /content/face-expression-recognition-dataset/images/images
10. establish training set and testing set by storing the corresponding paths into variables train_set and test_set. Corresponding folders are listed below in 10.1

10.1. training set: /content/face-expression-recognition-dataset/images/train
      testing set: /content/face-expression-recognition-dataset/images/validation

11. initiate dataframe function
12. import pandas as pd and os
13. establish training and testing dataframes
14. import numpy as np, cv2, and tqdm from tqdm
15. determine max_class_size using .value_counts()
16. initiate translate_image function, which will translate images in the training set such that each class has equal quantity
17. rerun dataframe function on training and testing set to update it
18. from tqdm.notebook import tqdm and from keras.preprocessing.image import load_img
19. initiate the extract_features function
20. using the extract_features function, extract features from training and testing dataframes
21. import necessary components for neural network, list of all requirements are shown below

    from keras.utils import to_categorical

    from keras.preprocessing.image import load_img

    from keras.models import Sequential

    from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

    from keras.layers import BatchNormalization

23. normalize the data by dividing the training and testing features by max pixel density value
24. import LabelEncoder from sklearn.preprocessing and utilize to fit labels for training set
25. transform label for training and testing set using Encoder.transform
26. convert data to numerical matrix using to_categorical
27. initiate convolutional neural network
28. compile the model
29. fit the training and testing data to the model


Results and Findings

It is possible the validation set does not possess enough data since the output accuracy of training and validation data demonstrated overfitting. While the training set is able to reach accuracy of 90%, validation set is only able to reach a max accuracy of 60%. This overfitting could potentially be resolved by physical collection of more unaugmented validation data.
