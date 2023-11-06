from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from skimage import feature
from imutils import paths
import numpy as np
import cv2
import os
import pickle
from TrainWave import ExtractFeatures, InitializeDataAndLabels

def Train():
    for i in range(0,5):
        model = RandomForestClassifier(n_estimators=100)
        model.fit(trainX,trainY)
        return model


trainingPath = r"D:\dataset\spiral\training"

(trainX, trainY) = InitializeDataAndLabels(trainingPath)


le2 = LabelEncoder()
trainY = le2.fit_transform(trainY)
#file to store the label encoders of spiral
file = open('spiral_label_encoder_object.pkl', 'wb')
pickle.dump(le2, file)
file.close()


SpiralModelTrained = Train()



SpiralModelFile = 'finalized_model2.sav'
pickle.dump(SpiralModelTrained, open(SpiralModelFile, 'wb'))


