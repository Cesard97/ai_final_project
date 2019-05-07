#!/usr/bin/env python

import rospy
from std_msgs.msg import Float32MultiArray, Int16
import glob
import numpy as np
import scipy.io as sc
from sklearn import svm


class SvmClassifier:

    def __init__(self):
        self.charac_vector = []
        self.nombresLandMark = []
        self.mediaP = None
        self.vectorClassifier = []
        self.matrixVectors = []
        self.y_angry = [0]
        self.y_disgust = [0]
        self.y_fear = [0]
        self.y_happy = [0]
        self.y_neutral = [0]
        self.y_sad = [0]
        self.svm_angry_model = None
        self.svm_disgust_model = None
        self.svm_fear_model = None
        self.svm_happy_model = None
        self.svm_neutral_model = None
        self.svm_sad_model = None

    def features_callback(self, data):
        self.charac_vector = data.data

    def svm_classifier(self):
        rospy.init_node('svm_classifier', anonymous=False)
        rospy.Subscriber('features', Float32MultiArray, self.features_callback)
        pub = rospy.Publisher('SVM_response', Int16, queue_size=10)
        rate = rospy.Rate(10)
        x, y = self.training_data_prep()
        self.svm_angry_model = self.svm_model(x, self.y_angry)
        self.svm_disgust_model = self.svm_model(x, self.y_disgust)
        self.svm_fear_model = self.svm_model(x, self.y_fear)
        self.svm_happy_model = self.svm_model(x, self.y_happy)
        self.svm_neutral_model = self.svm_model(x, self.y_neutral)
        self.svm_sad_model = self.svm_model(x, self.y_sad)
        while not rospy.is_shutdown():
            answer = 0
            if not (np.sum(self.charac_vector) == 0):
                emotion = [0]
                distance = [0]
                angry = self.svm_angry_model.predict(self.charac_vector)
                disgust = self.svm_disgust_model.predict(self.charac_vector)
                fear = self.svm_fear_model.predict(self.charac_vector)
                happy = self.svm_happy_model.predict(self.charac_vector)
                neutral = self.svm_neutral_model.predict(self.charac_vector)
                sad = self.svm_sad_model.predict(self.charac_vector)

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

                distance.append(self.plane_distance(self.svm_angry_model.coef_, self.charac_vector))
                distance.append(self.plane_distance(self.svm_disgust_model.coef_, self.charac_vector))
                distance.append(self.plane_distance(self.svm_fear_model.coef_, self.charac_vector))
                distance.append(self.plane_distance(self.svm_happy_model.coef_, self.charac_vector))
                distance.append(self.plane_distance(self.svm_neutral_model.coef_, self.charac_vector))
                distance.append(self.plane_distance(self.svm_sad_model.coef_, self.charac_vector))

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

    def mean(self):
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
            mediaP = self.mediaP
            landmark = sc.loadmat(element)
            vTem = landmark['faceCoordinatesUnwarped']
            if vTem.shape == (136, 1):
                vTem = vTem.T
            aux1 = np.dot(vTem, mediaP.T)
            aux2 = (np.dot(vTem, vTem.T))
            aux = aux1 / aux2
            vTem = vTem * aux
            vector = vTem[0] - self.mediaP
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

        self.y_angry = np.zeros(self.vectorClassifier.shape)
        self.y_disgust = np.zeros(self.vectorClassifier.shape)
        self.y_fear = np.zeros(self.vectorClassifier.shape)
        self.y_happy = np.zeros(self.vectorClassifier.shape)
        self.y_neutral = np.zeros(self.vectorClassifier.shape)
        self.y_sad = np.zeros(self.vectorClassifier.shape)

        for i in range(0, len(self.vectorClassifier)):
            if not(self.vectorClassifier[i] == 1):
                self.y_angry[i] = 0
            else:
                self.y_angry[i] = 1

        for i in range(0, len(self.vectorClassifier)):
            if not (self.vectorClassifier[i] == 2):
                self.y_disgust[i] = 0
            else:
                self.y_disgust[i] = 1

        for i in range(0, len(self.vectorClassifier)):
            if not(self.vectorClassifier[i] == 3):
                self.y_fear[i] = 0
            else:
                self.y_fear[i] = 1

        for i in range(0, len(self.vectorClassifier)):
            if not(self.vectorClassifier[i] == 4):
                self.y_happy[i] = 0
            else:
                self.y_happy[i] = 1

        for i in range(0, len(self.vectorClassifier)):
            if not(self.vectorClassifier[i] == 5):
                self.y_neutral[i] = 0
            else:
                self.y_neutral[i] = 1

        for i in range(0, len(self.vectorClassifier)):
            if not(self.vectorClassifier[i] == 5):
                self.y_sad[i] = 0
            else:
                self.y_sad[i] = 1

        return self.matrixVectors, self.vectorClassifier

    def svm_model(self, x, y):
        model = svm.LinearSVC(random_state=0, tol=1e-5)
        model.fit(x, y)
        return model

    def plane_distance(self, plane, vector):
        return np.linalg.norm(plane - vector)


if __name__ == '__main__':
    try:
        classifier = SvmClassifier()
        classifier.svm_classifier()
    except rospy.ROSInterruptException:
        pass
