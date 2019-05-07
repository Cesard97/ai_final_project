#!/usr/bin/env python

import rospy
from std_msgs.msg import Float32MultiArray, Int16
import glob
import numpy as np
import scipy.io as sc
from sklearn import svm


charac_vector = []
nombresLandMark = []
mediaP = []
vectorClassifier = []
matrixVectors = []
y_angry = [0]
y_disgust = [0]
y_fear = [0]
y_happy = [0]
y_neutral = [0]
y_sad = [0]


def features_callback(data):
    global charac_vector
    charac_vector = data.data


def svm_classifier():
    rospy.init_node('svm_classifier', anonymous=False)
    rospy.Subscriber('features', Float32MultiArray, features_callback)
    pub = rospy.Publisher('SVM_response', Int16, queue_size=10)
    rate = rospy.Rate(10)
    x, y = training_data_prep()
    svm_angry_model = svm_model(x, y_angry)
    svm_disgust_model = svm_model(x, y_disgust)
    svm_fear_model = svm_model(x, y_fear)
    svm_happy_model = svm_model(x, y_happy)
    svm_neutral_model = svm_model(x, y_neutral)
    svm_sad_model = svm_model(x, y_sad)
    while not rospy.is_shutdown():
        answer = 0
        if not (charac_vector == []):
            emotion = [0]
            distance = [0]
            angry = svm_angry_model.predict(charac_vector)
            disgust = svm_disgust_model.predict(charac_vector)
            fear = svm_fear_model.predict(charac_vector)
            happy = svm_happy_model.predict(charac_vector)
            neutral = svm_neutral_model.predict(charac_vector)
            sad = svm_sad_model.predict(charac_vector)

            print("Angry: " + str(angry))
            print("Disgust: " + str(disgust))
            print("Fear: " + str(fear))
            print("Happy: " + str(happy))
            print("Neutral: " + str(neutral))
            print("Sad: " + str(sad))

            emotion.append(angry)
            emotion.append(disgust)
            emotion.append(fear)
            emotion.append(happy)
            emotion.append(neutral)
            emotion.append(sad)

            distance.append(plane_distance(svm_angry_model.coef_, charac_vector))
            distance.append(plane_distance(svm_disgust_model.coef_, charac_vector))
            distance.append(plane_distance(svm_fear_model.coef_, charac_vector))
            distance.append(plane_distance(svm_happy_model.coef_, charac_vector))
            distance.append(plane_distance(svm_neutral_model.coef_, charac_vector))
            distance.append(plane_distance(svm_sad_model.coef_, charac_vector))

            count = emotion.count(1)
            if count > 1:
                indexes = []
                posibleEmotionDistance = np.zeros((7, 1))
                for i in range(0, len(emotion)):
                    if emotion[i] == 1:
                        indexes.append(i)
                        posibleEmotionDistance[i] = distance[i]

                answer = np.argmax(posibleEmotionDistance)
            print(answer)
            pub.publish(answer)
        rate.sleep()


def mean():
    global S, mediaP, nombresLandMark
    S = 0
    for file in glob.glob("training_data/happy/*.mat"):
        nombresLandMark.append(file)
    for file in glob.glob("training_data/angry/*.mat"):
        nombresLandMark.append(file)
    for file in glob.glob("training_data/sad/*.mat"):
        nombresLandMark.append(file)
    for file in glob.glob("training_data/fear/*.mat"):
        nombresLandMark.append(file)
    for file in glob.glob("training_data/disgust/*.mat"):
        nombresLandMark.append(file)
    for file in glob.glob("training_data/neutral/*.mat"):
        nombresLandMark.append(file)

    for element in nombresLandMark:
        landmark = sc.loadmat(element)
        vTem = landmark['faceCoordinatesUnwarped']
        S = S + np.outer(vTem, vTem.conjugate())

    [values, vectors] = np.linalg.eig(S)
    index = np.argmax(values)
    mediaP = vectors[:, index]
    mediaP = mediaP.real


def training_data_prep():
    global nombresLandMark, matrixVectors, vectorClassifier, mediaP, y_angry, y_disgust, y_fear, y_happy, y_sad, y_neutral
    mean()
    nombresLandMark = []

    for file in glob.glob("training_data/happy/*.mat"):
        nombresLandMark.append(file)
    for file in glob.glob("training_data/angry/*.mat"):
        nombresLandMark.append(file)
    for file in glob.glob("training_data/sad/*.mat"):
        nombresLandMark.append(file)
    for file in glob.glob("training_data/fear/*.mat"):
        nombresLandMark.append(file)
    for file in glob.glob("training_data/disgust/*.mat"):
        nombresLandMark.append(file)
    for file in glob.glob("training_data/neutral/*.mat"):
        nombresLandMark.append(file)

    for element in nombresLandMark:

        landmark = sc.loadmat(element)
        vTem = landmark['faceCoordinatesUnwarped']
        if vTem.shape == (136, 1):
            vTem = vTem.T
        aux1 = np.dot(vTem, mediaP.T)
        aux2 = (np.dot(vTem, vTem.T))
        aux = aux1 / aux2
        vTem = vTem * aux
        vector = vTem[0] - mediaP
        #print(vector)
        matrixVectors.append(vector)

        name = element.split("/")

        if name[1] == 'angry':
            vectorClassifier.append(1)
            #print(1)
        elif name[1] == 'disgust':
            vectorClassifier.append(2)
            #print(2)
        elif name[1] == 'fear':
            vectorClassifier.append(3)
            #print(3)
        elif name[1] == 'happy':
            vectorClassifier.append(4)
            #print(4)
        elif name[1] == 'neutral':
            vectorClassifier.append(5)
            #print(5)
        elif name[1] == 'sad':
            vectorClassifier.append(6)
            #print(6)
        else:
            vectorClassifier.append(0)

    matrixVectors = np.asanyarray(matrixVectors)
    vectorClassifier = np.asanyarray(vectorClassifier)

    y_angry = np.zeros(vectorClassifier.shape)
    y_disgust = np.zeros(vectorClassifier.shape)
    y_fear = np.zeros(vectorClassifier.shape)
    y_happy = np.zeros(vectorClassifier.shape)
    y_neutral = np.zeros(vectorClassifier.shape)
    y_sad = np.zeros(vectorClassifier.shape)

    for i in range(0, len(vectorClassifier)):
        if not(vectorClassifier[i] == 1):
            y_angry[i] = 0
        else:
            y_angry[i] = 1

    for i in range(0, len(vectorClassifier)):
        if not (vectorClassifier[i] == 2):
            y_disgust[i] = 0
        else:
            y_disgust[i] = 1

    for i in range(0, len(vectorClassifier)):
        if not(vectorClassifier[i] == 3):
            y_fear[i] = 0
        else:
            y_fear[i] = 1

    for i in range(0, len(vectorClassifier)):
        if not(vectorClassifier[i] == 4):
            y_happy[i] = 0
        else:
            y_happy[i] = 1

    for i in range(0, len(vectorClassifier)):
        if not(vectorClassifier[i] == 5):
            y_neutral[i] = 0
        else:
            y_neutral[i] = 1

    for i in range(0, len(vectorClassifier)):
        if not(vectorClassifier[i] == 5):
            y_sad[i] = 0
        else:
            y_sad[i] = 1

    return matrixVectors, vectorClassifier


def svm_model(x, y):
    model = svm.LinearSVC(random_state=0, tol=1e-5)
    model.fit(x, y)

    return model


def plane_distance(plane, vector):
    return np.linalg.norm(plane - vector)


if __name__ == '__main__':
    try:
        svm_classifier()
    except rospy.ROSInterruptException:
        pass
