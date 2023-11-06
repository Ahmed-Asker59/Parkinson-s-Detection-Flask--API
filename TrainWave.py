from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from skimage import feature
from imutils import paths
import numpy as np
import cv2
import os
import pickle

def ExtractFeatures(image):
	# compute the histogram of oriented gradients feature vector for
	# the input image
	features = feature.hog(image, orientations=9,
		pixels_per_cell=(10, 10), cells_per_block=(2, 2),
		transform_sqrt=True, block_norm="L1")
	# return the feature vector
	return features

def InitializeDataAndLabels(path):
	imagePaths = list(paths.list_images(path))
	data = []
	labels = []
	for imagePath in imagePaths:
		label = imagePath.split(os.path.sep)[-2]
		image = cv2.imread(imagePath)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		image = cv2.resize(image, (200, 200))
		image = cv2.threshold(image, 0, 255,
			cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
		features = ExtractFeatures(image)
		data.append(features)
		labels.append(label)
	# return the data and labels
	return (np.array(data), np.array(labels))

# define the path to the training and testing directories
trainingPath = r"D:\dataset\wave\training"

#initialize data and labels
'''
(trainX, trainY) = InitializeDataAndLabels(trainingPath)


le = LabelEncoder()
trainY = le.fit_transform(trainY)

file = open('object.pkl', 'wb')
pickle.dump(le, file)
file.close()
# encode the labels as integers


'''




def Train():
    for i in range(0,5):
        model = RandomForestClassifier(n_estimators=100)
        model.fit(trainX,trainY)
        return model
        




# WaveModelTrained = Train()


'''
WaveModelFile = 'finalized_model.sav'
pickle.dump(WaveModelTrained, open(WaveModelFile, 'wb'))
'''