import glob
from svm_classifier import SvmClassifier
from k_neighbors import KNeighbors
import matplotlib.pyplot as plt
import scipy.io as sc
import numpy as np


class Main:

    def __init__(self):
        self.dataFiles = []
        self.trainingDataFiles = []
        #self.kn = None
        self.kn_model = None
        self.kn_model_traning = None
        self.svm_model = None

        self.true_classes = []
        self.dataMatrix = []

        self.true_training_classes = []
        self.dataTrainingMatrix = []

        self.svm_prediction = None
        self.kn_prediction = None
        self.kn_prediction_training = None
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

    def training_data_reading(self):
        for file in glob.glob("training_data/happy/*.mat"):
            self.trainingDataFiles.append(file)
            self.true_training_classes.append(4)
        for file in glob.glob("training_data/angry/*.mat"):
            self.trainingDataFiles.append(file)
            self.true_training_classes.append(1)
        for file in glob.glob("training_data/sad/*.mat"):
            self.trainingDataFiles.append(file)
            self.true_training_classes.append(6)
        for file in glob.glob("training_data/fear/*.mat"):
            self.trainingDataFiles.append(file)
            self.true_training_classes.append(3)
        for file in glob.glob("training_data/disgust/*.mat"):
            self.trainingDataFiles.append(file)
            self.true_training_classes.append(2)
        for file in glob.glob("training_data/neutral/*.mat"):
            self.trainingDataFiles.append(file)
            self.true_training_classes.append(5)

        for i in self.trainingDataFiles:
            landmark = sc.loadmat(i)
            v_tem = landmark['faceCoordinatesUnwarped']
            self.dataTrainingMatrix.append(v_tem)


    def instance_creation(self, n=1):
        x, y = self.kn.training_data_prep()
        self.kn_model = self.kn.model_creation(x, y, n)
        x, y = self.svm.training_data_prep()
        self.svm.model_creation(x)

    def instance_creation_traning(self, n=1):
        x, y = self.kn.training_data_prep()
        self.kn_model_traning = self.kn.model_creation(x, y, n)
        x, y = self.svm.training_data_prep()
        self.svm.model_creation(x)

    def test_kn(self):
        self.kn_prediction = np.zeros(len(self.true_classes))
        for i in range(0, len(self.dataMatrix)):
            aux = self.dataMatrix[i].T
            if aux.shape == (136, 1):
                aux = aux.T
            self.kn_prediction[i] = int(self.kn_model.predict(aux))

    def test_training_kn(self):
        self.kn_prediction_training = np.zeros(len(self.true_training_classes))
        for i in range(0, len(self.dataTrainingMatrix)):
            aux = self.dataTrainingMatrix[i].T
            if aux.shape == (136, 1):
                aux = aux.T
            self.kn_prediction_training[i] = int(self.kn_model_traning.predict(aux))

    def test_svm(self):
        self.svm_prediction = np.zeros(len(self.true_classes))
        for i in range(0, len(self.dataMatrix)):
            aux = self.dataMatrix[i].T
            if aux.shape == (136, 1):
                aux = aux.T
            self.svm_prediction[i] = self.svm.compute_emotion(aux)

    def error_percentage(self):
        kn_counter = 0
        svm_counter = 0
        kn_percentage = 0.0
        svm_percentage = 0.0

        for i in range(0, len(self.true_classes)):
            #print(self.true_classes[i])
            #print(self.kn_prediction[i])
            #print(self.svm_prediction[i])
            if self.true_classes[i] == self.kn_prediction[i]:
                kn_counter = kn_counter + 1
            if self.true_classes[i] == self.svm_prediction[i]:
                svm_counter = svm_counter + 1

        kn_percentage = 100 - kn_counter*100/len(self.true_classes)
        svm_percentage = 100 - svm_counter*100/len(self.true_classes)

        #print(kn_percentage)
        #print(svm_percentage)
        return kn_percentage

    def error_percentage_training(self):
        kn_counter = 0
        svm_counter = 0
        kn_percentage = 0.0
        svm_percentage = 0.0

        for i in range(0, len(self.true_training_classes)):
            # print(self.true_classes[i])
            # print(self.kn_prediction[i])
            # print(self.svm_prediction[i])
            if self.true_training_classes[i] == self.kn_prediction_training[i]:
                kn_counter = kn_counter + 1
            #if self.true_classes[i] == self.svm_prediction[i]:
            #    svm_counter = svm_counter + 1

        kn_percentage = 100 - kn_counter * 100 / len(self.true_training_classes)
        #svm_percentage = 100 - svm_counter * 100 / len(self.true_classes)

        #print(kn_percentage)
        #print(svm_percentage)
        return kn_percentage

    def confusion_matrix(self):
        kn_confusion = np.zeros((7, 7))
        svm_confusion = np.zeros((7, 7))
        for i in range(0, len(self.true_classes)):
            kn_confusion[self.true_classes[i]][self.kn_prediction[i]] = kn_confusion[self.true_classes[i]][self.kn_prediction[i]] + 1
            svm_confusion[self.true_classes[i]][self.svm_prediction[i]] = svm_confusion[self.true_classes[i]][self.svm_prediction[i]] + 1

        print(kn_confusion)
        print(svm_confusion)

    def optimizeKneig(self):
        # Error de entrenamiento
        minError = 100
        kOptimo = 0
        error_vector = np.zeros((500))
        error_vector_training = np.zeros((500))
        k_vector = np.zeros((500))
        self.training_data_reading()
        for k in range(1, 500):
            self.kn = KNeighbors()
            self.svm = SvmClassifier()
            self.instance_creation_traning(k)
            self.test_training_kn()
            self.test_svm()
            error = self.error_percentage_training()
            error_vector_training[k] = error
            if error < minError:
                minError = error
                kOptimo = k
            print(k)
        print(kOptimo)
        print(minError)

        # Error de Validacion
        minError = 100
        kOptimo = 0
        self.data_reading()
        for k in range(1, 500):
            self.kn = KNeighbors()
            self.svm = SvmClassifier()
            self.instance_creation(k)
            self.test_kn()
            self.test_svm()
            error = self.error_percentage()
            k_vector[k] = k
            error_vector[k] = error
            if error < minError:
                minError = error
                kOptimo = k
            print(k)
        print(kOptimo)
        print(minError)

        #self.confusion_matrix()
        plt.plot(k_vector, error_vector)
        plt.plot(k_vector, error_vector_training)
        plt.title("Errores de validacion y de entrenamiento contra parametro K")
        plt.show()




if __name__ == '__main__':
    main = Main()
    #main.data_reading()
    #main.instance_creation()
    #main.test_kn()
    #main.test_svm()
    #main.error_percentage()
    #main.error_percentage_training()
    #main.confusion_matrix()
    main.optimizeKneig()