import numpy as np
import glob
from svm_classifier import SvmClassifier
from k_neighbors import KNeighbors


class Main:

    def __init__(self):
        self.dataFiles = []
        self.kn = None
        self.svm_angry_model = None
        self.svm_disgust_model = None
        self.svm_fear_model = None
        self.svm_happy_model = None
        self.svm_neutral_model = None
        self.svm_sad_model = None

        self.true_classes = []

        self.svm_prediction = None
        self.kn_prediction = None

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

    def instance_creation(self):
        x, y = kn.training_data_prep()
        self.kn = kn.model_creation(x, y)
        x, y = svm.training_data_prep()
        self.svm_angry_model = svm.svm_model(x, svm.y_angry)
        self.svm_disgust_model = svm.svm_model(x, svm.y_disgust)
        self.svm_fear_model = svm.svm_model(x, svm.y_fear)
        self.svm_happy_model = svm.svm_model(x, svm.y_happy)
        self.svm_neutral_model = svm.svm_model(x, svm.y_neutral)
        self.svm_sad_model = svm.svm_model(x, svm.y_sad)

    def test_kn(self):
        for i in



if __name__ == '__main__':
    main = Main()
    main.instance_creation()
    kn = KNeighbors()
    svm = SvmClassifier()