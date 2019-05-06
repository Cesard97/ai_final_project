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


def features_callback(data):
    global charac_vector
    charac_vector = data.data


def svm_classifier():
    rospy.init_node('svm_classifier', anonymous=False)
    rospy.Subscriber('features', Float32MultiArray, features_callback)
    pub = rospy.Publisher('SVM_response', Int16, queue_size=1)
    rate = rospy.Rate(10)
    x, y = training_data_prep()
    svm_angry_model = svm_model(1, x, y)
    svm_disgust_model = svm_model(2, x, y)
    svm_fear_model = svm_model(3, x, y)
    svm_happy_model = svm_model(4, x, y)
    svm_neutral_model = svm_model(5, x, y)
    svm_sad_model = svm_model(6, x, y)
    while not rospy.is_shutdown():

        if not (charac_vector == []):
            emotion = [0]
            distance = [0]

            emotion.append(svm_angry_model.predict(charac_vector))
            emotion.append(svm_disgust_model.predict(charac_vector))
            emotion.append(svm_fear_model.predict(charac_vector))
            emotion.append(svm_happy_model.predict(charac_vector))
            emotion.append(svm_neutral_model.predict(charac_vector))
            emotion.append(svm_sad_model.predict(charac_vector))

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

                emotion = np.argmax(posibleEmotionDistance)
            print(emotion)
            pub.publish(emotion)
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
    global nombresLandMark, matrixVectors, vectorClassifier, mediaP
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
        matrixVectors.append(vector)

        name = element.split("/")

        if name[1] == 'angry':
            vectorClassifier.append(1)
        elif name[1] == 'disgust':
            vectorClassifier.append(2)
        elif name[1] == 'fear':
            vectorClassifier.append(3)
        elif name[1] == 'happy':
            vectorClassifier.append(4)
        elif name[1] == 'neutral':
            vectorClassifier.append(5)
        elif name[1] == 'sad':
            vectorClassifier.append(6)
        else:
            vectorClassifier.append(0)
    matrixVectors = np.asanyarray(matrixVectors)
    vectorClassifier = np.asanyarray(vectorClassifier)
    return matrixVectors, vectorClassifier


def svm_model(num, x, y):
    y1 = np.zeros(len(y))
    for i in range(0, len(y)):
        if not(y[i] == num):
            y1[i] = 0
        else:
            y1[i] = 1
    model = svm.LinearSVC(random_state=0, tol=1e-5)
    model.fit(x, y1)

    return model


def plane_distance(plane, vector):
    return np.linalg.norm(plane - vector)


if __name__ == '__main__':
    try:
        svm_classifier()
    except rospy.ROSInterruptException:
        pass
