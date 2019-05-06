#!/usr/bin/env python

import rospy
from sklearn.neighbors import KNeighborsClassifier
from std_msgs.msg import Float32MultiArray, Int16
import glob
import numpy as np
import scipy.io as sc

global nombresLandMark
charac_vector = []
nombresLandMark = []
mediaP = []
vectorClassifier = []
matrixVectors = []
clf = 0
svm_angry_model = None
svm_disgust_model = None
svm_fear_model = None
svm_happy_model = None
svm_neutral_model = None
svm_sad_model = None


def features_callback(data):
    global charac_vector
    charac_vector = data.data


def k_neighbors():
    rospy.init_node('k_neighbors', anonymous=False)
    rospy.Subscriber('features', Float32MultiArray, features_callback)
    pub = rospy.Publisher('KN_response', Int16, queue_size=10)
    rate = rospy.Rate(10)
    x, y = training_data_prep()
    neigh = KNeighborsClassifier(n_neighbors=10)
    neigh.fit(x, np.transpose(y))
    while not rospy.is_shutdown():
        if not (charac_vector == []):
            emotion = neigh.predict(charac_vector)
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
        aux = (np.dot(vTem[0], mediaP.T) / (np.dot(vTem, vTem.T)))
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


if __name__ == '__main__':
    try:
        k_neighbors()
    except rospy.ROSInterruptException:
        pass
