import glob
from svm_classifier import SvmClassifier
from k_neighbors import KNeighbors
import scipy.io as sc

class Main:

    def __init__(self):
        self.dataFiles = []
        self.kn = None
        self.kn_model = None

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
        self.svm.model_creation(x)

    def test_kn(self):
        for i in self.dataMatrix:
            self.kn_prediction[self.dataMatrix[i]] = self.kn_model.predict(i)

    def test_svm(self):
        for i in self.dataMatrix:
            self.kn_prediction[self.dataMatrix[i]] = self.svm.compute_emotion(i)

    def error_percentage(self):
        kn_counter = 0
        svm_counter = 0
        for i in range(0, len(self.true_classes)):
            if self.true_classes[i] == self.kn_prediction[i]:
                kn_counter = kn_counter + 1
            if self.true_classes[i] == self.svm_prediction[i]:
                svm_counter = svm_counter + 1
        kn_percentage = kn_counter/len(self.true_classes)
        svm_percentage = svm_counter / len(self.true_classes)

    






if __name__ == '__main__':
    main = Main()
    main.instance_creation()