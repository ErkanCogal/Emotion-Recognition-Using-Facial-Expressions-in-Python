import cv2
import glob
import random
import math
import numpy as np
import dlib
import itertools
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

classifier = KNeighborsClassifier(n_neighbors=10)

emotions = ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sadness", "surprise"] #Emotion list
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"C:\Users\barba\.spyder-py3\shape_predictor_68_face_landmarks.dat") #Or set this to whatever you named the downloaded file

data = {} #Make dictionary for all values
#data['landmarks'] = []
def get_files(emotion): #Define function to get file list, randomly shuffle it and split 80/20
    files = glob.glob("dataset\\%s\\*" %emotion)
    random.shuffle(files)
    training = files[:int(len(files)*0.8)] #get first 80% of file list
    prediction = files[-int(len(files)*0.2):] #get last 20% of file list
    return training, prediction
def get_landmarks(image):
    detections = detector(image, 1)
    for k,d in enumerate(detections): #For all detected face instances individually
        shape = predictor(image, d) #Draw Facial Landmarks with the predictor class
        xlist = []
        ylist = []
        landmarks = []
        for i in range(1,68): #Store X and Y coordinates in two lists
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))
        for x, y in zip(xlist, ylist): #Store all landmarks in one list in the format x1,y1,x2,y2,etc.
            landmarks.append(x)
            landmarks.append(y)
    data['landmarks'] = landmarks        
    if len(detections) > 0:
        return landmarks
    else: #If no faces are detected, return error message to other function to handle
        landmarks = "error"
        return landmarks

min_max_scaler = preprocessing.MinMaxScaler() 
def make_sets():
    training_data = []
    trainingx_data = []
    training_labels = []
    prediction_data = []
    predictionx_data = []
    prediction_labels = []
    for emotion in emotions:
        #print(" working on %s" %emotion)
        training, prediction = get_files(emotion)
        #Append data to training and prediction list, and generate labels 0-7
        for item in training:
            image = cv2.imread(item) #open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
            clahe_image = clahe.apply(gray)
            get_landmarks(clahe_image)
            if data['landmarks'] == "error":
                print("no face detected on this one")
            else:
                trainingx_data.append(data['landmarks']) #append image array to training data list
                training_data = min_max_scaler.fit_transform(trainingx_data)
                training_labels.append(emotions.index(emotion))
        for item in prediction:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe_image = clahe.apply(gray)
            get_landmarks(clahe_image)
            if data['landmarks'] == "error":
                print("no face detected on this one")
            else:
                predictionx_data.append(data['landmarks'])
                prediction_data = min_max_scaler.transform(predictionx_data)
                prediction_labels.append(emotions.index(emotion))
    return training_data, training_labels, prediction_data, prediction_labels
accur_lin = []

for i in range(1,11):
    print("\n\n---------------------------------------------------------------------------")
    print("\n Set %s for Dlib + kNN \n\n" %i) #Make sets by random sampling 80/20%
    training_data, training_labels, prediction_data, prediction_labels = make_sets()
    npar_train = np.array(training_data) #Turn the training set into a numpy array for the classifier
    npar_trainlabs = np.array(training_labels)
  
    
    classifier.fit(training_data, training_labels)
    y_pred = classifier.predict(prediction_data)
    print("Confusion Matrix for set %s" %i)
    print(" \n\n  AngConDisFeaHapNeuSadSur ")
    print(confusion_matrix(prediction_labels, y_pred))
    print("\n\nclassification report\n")
    print(classification_report(prediction_labels, y_pred))
    
    #print("getting accuracies %s" %i) #Use score() function to get accuracy
    npar_pred = np.array(prediction_data)
    pred_lin = classifier.score(npar_pred, prediction_labels)
    #print ("linear: ", pred_lin)
    #print(clf.predict_proba(prediction_data)[0])
    accur_lin.append(pred_lin) #Store accuracy in a list
print("Mean accuracy of kNN: %s" %np.mean(accur_lin)) #Mean accuracy of the 10 runs
