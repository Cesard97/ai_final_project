import numpy as np
import glob
from svm_classifier import SvmClassifier
from k_neighbors import KNeighbors
import scipy.io as sc

class Main:

    def __init__(self):
        self.dataFiles = []
        self.kn = None
        self.kn_model = None
        self.svm_angry_model = None
        self.svm_disgust_model = None
        self.svm_fear_model = None
        self.svm_happy_model = None
        self.svm_neutral_model = None
        self.svm_sad_model = None

        self.true_classes = []
        self.dataMatrix = []

        self.svm_prediction = None
        self.kn_prediction = None
        self.kn = KNeighbors()
        self.svm = SvmClassifier()

    def data_reading(self):
        for file in glob.glob("validation/happy/*.mat"):
            self.dataFiles.append(file)
            self.true_classes.append(4)
        for file in glob.glob("validation/angry/*.mat"):
            self.dataFiles.append(file)
            self.true_classes.append(1)
        for file in glob.glob("validation/sad/*.mat"):
            self.dataFiles.append(file)
            self.true_classes.append(6)
        for file in glob.glob("validation/fear/*.mat"):
            self.dataFiles.append(file)
            self.true_classes.append(3)
        for file in glob.glob("validation/disgust/*.mat"):
            self.dataFiles.append(file)
            self.true_classes.append(2)
        for file in glob.glob("validation/neutral/*.mat"):
            self.dataFiles.append(file)
            self.true_classes.append(5)
        for i in self.dataFiles:
            landmark = sc.loadmat(i)
            v_tem = landmark['faceCoordinatesUnwarped']
            self.dataMatrix.append(v_tem)

    def instance_creation(self):
        x, y = self.kn.training_data_prep()
        self.kn_model = self.kn.model_creation(x, y)
        x, y = self.svm.training_data_prep()
        self.svm_angry_model = self.svm.svm_model(x, self.svm.y_angry)
        self.svm_disgust_model = self.svm.svm_model(x, self.svm.y_disgust)
        self.svm_fear_model = self.svm.svm_model(x, self.svm.y_fear)
        self.svm_happy_model = self.svm.svm_model(x, self.svm.y_happy)
        self.svm_neutral_model = self.svm.svm_model(x, self.svm.y_neutral)
        self.svm_sad_model = self.svm.svm_model(x, self.svm.y_sad)

    def test_kn(self):
        for i in self.dataMatrix:
            self.kn_prediction[self.dataMatrix[i]] = self.kn_model.predict(i)
    def test_svm(self):
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




if __name__ == '__main__':
    main = Main()
    main.instance_creation()