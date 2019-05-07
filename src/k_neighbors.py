#!/usr/bin/env python

import rospy
from sklearn.neighbors import KNeighborsClassifier
from std_msgs.msg import Float32MultiArray, Int16
import glob
import numpy as np
import scipy.io as sc


class KNeighbors:

    def __init__(self):
        self.charac_vector = []
        self.nombresLandMark = []
        self.mediaP = None
        self.vectorClassifier = []
        self.matrixVectors = []

    def features_callback(self, data):
        self.charac_vector = data.data

    def k_neighbors(self):
        rospy.init_node('k_neighbors', anonymous=False)
        rospy.Subscriber('features', Float32MultiArray, self.features_callback)
        pub = rospy.Publisher('KN_response', Int16, queue_size=1)
        rate = rospy.Rate(1)
        x, y = self.training_data_prep()
        neigh = self.model_creation(x, y)
        while not rospy.is_shutdown():
            if np.sum(self.charac_vector) == 0:
                emotion = 0
            else:
                emotion = neigh.predict(self.charac_vector)
            string = ""
            if emotion == 1:
                string = "angry"
            elif emotion == 2:
                string = "disgust"
            elif emotion == 3:
                string = "fear"
            elif emotion == 4:
                string = "happy"
            elif emotion == 5:
                string = "neutral"
            elif emotion == 6:
                string = "sad"
            elif emotion == 0:
                string= "nada prros"
            print(string)
            pub.publish(emotion)
            rate.sleep()

    def model_creation(self, x, y):
        neigh = KNeighborsClassifier(n_neighbors=150)
        neigh.fit(x, np.transpose(y))
        return neigh

    def mean(self):
        global S, mediaP, nombresLandMark
        S = 0
        for file in glob.glob("training_data/happy/*.mat"):
            self.nombresLandMark.append(file)
        for file in glob.glob("training_data/angry/*.mat"):
            self.nombresLandMark.append(file)
        for file in glob.glob("training_data/sad/*.mat"):
            self.nombresLandMark.append(file)
        for file in glob.glob("training_data/fear/*.mat"):
            self.nombresLandMark.append(file)
        for file in glob.glob("training_data/disgust/*.mat"):
            self.nombresLandMark.append(file)
        for file in glob.glob("training_data/neutral/*.mat"):
            self.nombresLandMark.append(file)

        for element in self.nombresLandMark:
            landmark = sc.loadmat(element)
            vTem = landmark['faceCoordinatesUnwarped']
            S = S + np.outer(vTem, vTem.conjugate())

        [values, vectors] = np.linalg.eig(S)
        index = np.argmax(values)
        mediaP = vectors[:, index]
        self.mediaP = mediaP.real

    def training_data_prep(self):
        self.mean()
        self.nombresLandMark = []

        for file in glob.glob("training_data/happy/*.mat"):
            self.nombresLandMark.append(file)
        for file in glob.glob("training_data/angry/*.mat"):
            self.nombresLandMark.append(file)
        for file in glob.glob("training_data/sad/*.mat"):
            self.nombresLandMark.append(file)
        for file in glob.glob("training_data/fear/*.mat"):
            self.nombresLandMark.append(file)
        for file in glob.glob("training_data/disgust/*.mat"):
            self.nombresLandMark.append(file)
        for file in glob.glob("training_data/neutral/*.mat"):
            self.nombresLandMark.append(file)

        for element in self.nombresLandMark:

            landmark = sc.loadmat(element)
            vTem = landmark['faceCoordinatesUnwarped']
            if vTem.shape == (136, 1):
                vTem = vTem.T
            aux = (np.dot(vTem[0], mediaP.T) / (np.dot(vTem, vTem.T)))
            vTem = vTem * aux
            vector = vTem[0] - mediaP
            self.matrixVectors.append(vector)

            name = element.split("/")

            if name[1] == 'angry':
                self.vectorClassifier.append(1)
            elif name[1] == 'disgust':
                self.vectorClassifier.append(2)
            elif name[1] == 'fear':
                self.vectorClassifier.append(3)
            elif name[1] == 'happy':
                self.vectorClassifier.append(4)
            elif name[1] == 'neutral':
                self.vectorClassifier.append(5)
            elif name[1] == 'sad':
                self.vectorClassifier.append(6)
            else:
                self.vectorClassifier.append(0)

        self.matrixVectors = np.asanyarray(self.matrixVectors)
        self.vectorClassifier = np.asanyarray(self.vectorClassifier)
        return self.matrixVectors, self.vectorClassifier


if __name__ == '__main__':
    try:
        classifier = KNeighbors()
        classifier.k_neighbors()
    except rospy.ROSInterruptException:
        pass
