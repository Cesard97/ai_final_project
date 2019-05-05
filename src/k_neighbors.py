#!/usr/bin/env python

import rospy
from sklearn.neighbors import KNeighborsClassifier
from std_msgs.msg import Float32MultiArray, Int16
import glob
import numpy as np
import scipy.io as sc

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
    neigh = KNeighborsClassifier(n_neighbors=2)
    neigh.fit(x, np.transpose(y))
    while not rospy.is_shutdown():

        if not (charac_vector == []):
            print(len(charac_vector))
            print(charac_vector)
            emotion = neigh.predict(charac_vector)
            print(emotion)
            pub.publish(emotion)
        rate.sleep()


def mean():
    global S, mediaP
    S = 0

    for file in glob.glob("ArchivosSoporteTarea7/ArchivosPunto1/markings/*.mat"):
        nombresLandMark.append(file)

    for element in nombresLandMark:
        landmark = sc.loadmat(element)
        vTem = landmark['faceCoordinatesUnwarped']
        media1 = np.mean(vTem[:, 0])
        media2 = np.mean(vTem[:, 1])
        vTem[:, 0] = vTem[:, 0] - media1
        vTem[:, 1] = vTem[:, 1] - media2
        vTem = np.vectorize(complex)(vTem[:, 0], vTem[:, 1])
        vTem = vTem / np.linalg.norm(vTem)
        S = S + np.outer(vTem, vTem.conjugate())

    [values, vectors] = np.linalg.eig(S)
    index = np.argmax(values)
    mediaP = vectors[:, index]


def training_data_prep():
    global nombresLandMark, matrixVectors, vectorClassifier, mediaP
    mean()
    nombresLandMark = []

    for file in glob.glob("ArchivosSoporteTarea7/ArchivosPunto1/markings/*.mat"):
        nombresLandMark.append(file)

    for element in nombresLandMark:
        landmark = sc.loadmat(element)
        vTem = landmark['faceCoordinatesUnwarped']
        media1 = np.mean(vTem[:, 0])
        media2 = np.mean(vTem[:, 1])
        vTem[:, 0] = vTem[:, 0] - media1
        vTem[:, 1] = vTem[:, 1] - media2
        vTem = np.vectorize(complex)(vTem[:, 0], vTem[:, 1])
        vTem = vTem / np.linalg.norm(vTem)
        vTem = vTem * (np.dot(vTem.conjugate().T, mediaP) / (np.dot(vTem.conjugate().T, vTem)))
        real = np.array([vTem.real - mediaP.real])
        imag = np.array([vTem.imag - mediaP.imag])
        vector = np.concatenate((real.T, imag.T), axis=0)
        matrixVectors.append(vector)
        name = element.split("/")
        clas = name[3].split("_")
        if clas[3] == 'a':
            vectorClassifier.append(1)
        elif clas[3] == 'd':
            vectorClassifier.append(2)
        elif clas[3] == 'f':
            vectorClassifier.append(3)
        elif clas[3] == 'h':
            vectorClassifier.append(4)
        elif clas[3] == 'n':
            vectorClassifier.append(5)
        elif clas[3] == 's':
            vectorClassifier.append(6)
        else:
            vectorClassifier.append(0)

    matrixVectors = np.asanyarray(matrixVectors)
    vectorClassifier = np.asanyarray(vectorClassifier)
    return matrixVectors[:, :, 0], vectorClassifier


if __name__ == '__main__':
    try:
        k_neighbors()
    except rospy.ROSInterruptException:
        pass
