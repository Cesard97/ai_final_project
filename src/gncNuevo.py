#!/usr/bin/env python
import numpy as np
import scipy.io as sc
import glob
import rospy
from std_msgs.msg import Float32, Float32MultiArray, Int16
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


class gnc:

    def __init__(self):
        self.vector = 0
        self.nombresLandMarkHappy = []
        self.nombresLandMarkAngry = []
        self.nombresLandMarkNeutral = []
        self.nombresLandMarkSad = []
        self.nombresLandMarkFear = []
        self.nombresLandMarkDisgust = []
        self.nombresLandMark = []
        self.nombresValidation = []

        for file in glob.glob("training_data/happy/*.mat"):
            self.nombresLandMarkHappy.append(file)
            self.nombresLandMark.append(file)

        for file in glob.glob("training_data/angry/*.mat"):
            self.nombresLandMarkAngry.append(file)
            self.nombresLandMark.append(file)

        for file in glob.glob("training_data/neutral/*.mat"):
            self.nombresLandMarkNeutral.append(file)
            self.nombresLandMark.append(file)

        for file in glob.glob("training_data/sad/*.mat"):
            self.nombresLandMarkSad.append(file)
            self.nombresLandMark.append(file)

        for file in glob.glob("training_data/fear/*.mat"):
            self.nombresLandMarkFear.append(file)
            self.nombresLandMark.append(file)

        for file in glob.glob("training_data/disgust/*.mat"):
            self.nombresLandMarkDisgust.append(file)
            self.nombresLandMark.append(file)


        for file in glob.glob("validation_data/happy/*.mat"):
            self.nombresValidation.append(file)

        for file in glob.glob("validation_data/angry/*.mat"):
            self.nombresValidation.append(file)

        for file in glob.glob("validation_data/neutral/*.mat"):
            self.nombresValidation.append(file)

        for file in glob.glob("validation_data/sad/*.mat"):
            self.nombresValidation.append(file)

        for file in glob.glob("validation_data/fear/*.mat"):
            self.nombresValidation.append(file)

        for file in glob.glob("validation_data/disgust/*.mat"):
            self.nombresValidation.append(file)

        self.mediaP = 0
        self.pPrueba = 0
        self.pHappy = 0
        self.pSad = 0
        self.pAngry = 0
        self.pFear = 0
        self.pNeutral = 0
        self.pDisgusted = 0

        self.numberTotalOfValidation = 89

    def faturesCallBack(self, msg):
        self.vector = msg.data
        self.vector = np.asmatrix(self.vector)
        self.vector = self.vector/2
        pass

    def graficarLandMarks(self):

        for element in self.nombresLandMark:
            landmark = sc.loadmat(element)
            vTem = landmark['faceCoordinatesUnwarped']
            c = np.sqrt((20 * np.random.rand(68)) ** 2)
            plt.scatter(vTem, marker='o', c=c)

        plt.title('Imagen Land Marks Originales')
        # plt.xticks([]), plt.yticks([])
        plt.grid(True)
        plt.show()

    def hallarMediaProcrustes(self):

        S = 0

        for element in self.nombresLandMark:
            landmark = sc.loadmat(element)
            vTem = landmark['faceCoordinatesUnwarped']
            if vTem.shape == (1, 136):
                vTem = vTem.T
            S = S + np.outer(vTem, vTem)
        [values, vectors] = np.linalg.eig(S)
        index = np.argmax(values)
        self.mediaP = vectors[:, index]
        self.mediaP = self.mediaP.real
        print('Encontro media')

    def training(self, cosa):

        happy = []
        sad = []
        angry = []
        fear = []
        neutral = []
        disgusted = []

        print('Va a comenzar a trainear')

        for element in self.nombresLandMark:

            landmark = sc.loadmat(element)
            vTem = landmark['faceCoordinatesUnwarped']
            #print(self.mediaP.shape)
            if vTem.shape == (1,136):
                vTem = vTem.T
            #print(vTem[:,0].shape)
            vTem = vTem[:,0] * (np.dot(vTem[:,0].T, self.mediaP) / (np.dot(vTem[:,0].T, vTem)))
            vector = np.array([vTem - self.mediaP])

            ### Divide by name each landmark

            name = element.split("/")

            #### Clasificacion de los vectores de caracteristicas
            if name[1] == 'happy':
                happy.append(vector)
            elif name[1] == 'sad':
                sad.append(vector)
            elif name[1] == 'angry':
                angry.append(vector)
            elif name[1] == 'fear':
                fear.append(vector)
            elif name[1] == 'neutral':
                neutral.append(vector)
            elif name[1] == 'disgust':
                disgusted.append(vector)

        #### Estimacion de la media y matriz de covarianza para cada clase

        meanHappy = 0
        CxHappy = 0
        meanSad = 0
        CxSad = 0
        meanAngry = 0
        CxAngry = 0
        meanFear = 0
        CxFear = 0
        meanNeutral = 0
        CxNeutral = 0
        meanDisgusted = 0
        CxDisgusted = 0

        I = np.eye(136)

        print('Ya clasifico datos')
        print('longitud de Happy '+str(len(happy)))
        print('longitud de sad ' + str(len(sad)))
        print('longitud de angry ' + str(len(angry)))
        print('longitud de fear ' + str(len(fear)))
        print('longitud de neutral ' + str(len(neutral)))
        print('longitud de disgust ' + str(len(disgusted)))

        ######################## Happy class ##########################

        for i in range(0, len(happy)):
            meanHappy = meanHappy + happy[i]
        meanHappy = meanHappy / len(happy)

        for i in range(0, len(happy)):
            CxHappy = CxHappy + np.outer(happy[i] - meanHappy, happy[i] - meanHappy)
        CxHappy = CxHappy / (len(happy))
        CxHappy = CxHappy + cosa * I
        #CxHappy2 = np.cov(happy[:,0,:])

        detHappy = np.linalg.det(CxHappy)
        print(detHappy)
        invHappy = np.linalg.inv(CxHappy)
        invHappy = np.asmatrix(invHappy)
        vectorPrueba = -np.ones((1,136))/2
        
        self.pHappy = lambda x: (1/np.sqrt(detHappy))*(1 / ((2 * np.pi))) * np.exp(-0.5 * (x - meanHappy) * invHappy * (x - meanHappy).T)
        

        ######################## Sad class ##########################

        for i in range(0, len(sad)):
            meanSad = meanSad + sad[i]
        meanSad = meanSad / len(sad)

        for i in range(0, len(sad)):
            CxSad = CxSad + np.outer(sad[i] - meanSad, sad[i] - meanSad)
        CxSad = CxSad / (len(sad) - 1)
        CxSad = CxSad + cosa * I

        detSad = np.linalg.det(CxSad)
        invSad = np.linalg.inv(CxSad)
        invSad = np.asmatrix(invSad)
        self.pSad = lambda x: (1/np.sqrt(detSad))*(1 / ((2 * np.pi))) * np.exp(-0.5 * (x - meanSad) * invSad * (x - meanSad).T)


        ######################## Angry class ##########################

        for i in range(0, len(angry)):
            meanAngry = meanAngry + angry[i]
        meanAngry = meanAngry / len(angry)

        for i in range(0, len(angry)):
            CxAngry = CxAngry + np.outer(angry[i] - meanAngry, angry[i] - meanAngry)
        CxAngry = CxAngry / (len(angry) - 1)
        CxAngry = CxAngry + cosa * I

        detAngry = np.linalg.det(CxAngry)
        invAngry = np.linalg.inv(CxAngry)
        invAngry = np.asmatrix(invAngry)
        self.pAngry = lambda x: (1/np.sqrt(detAngry))*(1 / ((2 * np.pi))) * np.exp(-0.5 * (x - meanAngry) * invAngry * (x - meanAngry).T)


        ######################## Fear class ##########################

        for i in range(0, len(fear)):
            meanFear = meanFear + fear[i]
        meanFear = meanFear / len(fear)

        for i in range(0, len(fear)):
            CxFear = CxFear + np.outer(fear[i] - meanFear, fear[i] - meanFear)
        CxFear = CxFear / (len(fear) - 1)
        CxFear = CxFear + cosa * I

        detFear = np.linalg.det(CxFear)
        invFear = np.linalg.inv(CxFear)
        invFear = np.asmatrix(invFear)
        self.pFear = lambda x: (1/np.sqrt(detFear))*(1 / ((2 * np.pi))) * np.exp(-0.5 * (x - meanFear) * invFear * (x - meanFear).T)


        ######################## Neutral class ##########################

        for i in range(0, len(neutral)):
            meanNeutral = meanNeutral + neutral[i]
        meanNeutral = meanNeutral / len(neutral)

        for i in range(0, len(neutral)):
            CxNeutral = CxNeutral + np.outer(neutral[i] - meanNeutral, neutral[i] - meanNeutral)
        CxNeutral = CxNeutral / (len(neutral) - 1)
        CxNeutral = CxNeutral + cosa * I

        detNeutral = np.linalg.det(CxNeutral)
        invNeutral = np.linalg.inv(CxNeutral)
        invNeutral = np.asmatrix(invNeutral)
        self.pNeutral = lambda x: (1/np.sqrt(detNeutral))*(1 / ((2 * np.pi))) * np.exp(-0.5 * (x - meanNeutral) * invNeutral * (x - meanNeutral).T)

        ######################## Disgusted class ##########################

        for i in range(0, len(disgusted)):
            meanDisgusted = meanDisgusted + disgusted[i]
        meanDisgusted = meanDisgusted / len(disgusted)

        for i in range(0, len(disgusted)):
            CxDisgusted = CxDisgusted + np.outer(disgusted[i] - meanDisgusted, disgusted[i] - meanDisgusted)
        CxDisgusted = CxDisgusted / (len(disgusted) - 1)
        CxDisgusted = CxDisgusted + cosa * I

        detDisgusted = np.linalg.det(CxDisgusted)
        invDisgusted = np.linalg.inv(CxDisgusted)
        invDisgusted = np.asmatrix(invDisgusted)
        self.pDisgusted = lambda x: (1/np.sqrt(detDisgusted))*(1 / ((2 * np.pi))) * np.exp(-0.5 * (x - meanDisgusted) * invDisgusted * (x - meanDisgusted).T)

        print('Termino training')

    def validation(self, element):

        emocion = 0
        #### Clasificacion de los vectores de caracteristicas
        vectorPrueba = np.ones((1,136))/2
        probaClass = 0.18
        probaHappy = self.pHappy(element) * probaClass
        print('Probabilidad feliz'+str(probaHappy))
        probaSad = self.pSad(element) * probaClass
        print('Probabilidad triste'+str(probaSad))
        probaAngry = self.pAngry(element) * probaClass
        print('Probabilidad bravo'+str(probaAngry))
        probaFear = self.pFear(element) * probaClass
        print('Probabilidad miedo'+str(probaFear))
        probaNeutral = self.pNeutral(element) * probaClass
        print('Probabilidad neutral'+str(probaNeutral))
        probaDisgusted = self.pDisgusted(element) * probaClass
        print('Probabilidad disgustado'+str(probaDisgusted))

        if probaHappy > probaSad and probaHappy > probaAngry and probaHappy > probaFear and probaHappy > probaNeutral and probaHappy > probaDisgusted:
            print('Clase correspondinte: Happy')
            print('Probabilidad Happy' + str(probaHappy))
            emocion = 4
        elif probaSad > probaHappy and probaSad > probaAngry and probaSad > probaFear and probaSad > probaNeutral and probaSad > probaDisgusted:
            print('Clase correspondinte: Sad')
            print('Probabilidad Sad' + str(probaSad))
            emocion = 6
        elif probaAngry > probaHappy and probaAngry > probaSad and probaAngry > probaFear and probaAngry > probaNeutral and probaAngry > probaDisgusted:
            print('Clase correspondinte: Angry')
            print('Probabilidad Angry' + str(probaAngry))
            emocion = 1
        elif probaFear > probaHappy and probaFear > probaSad and probaFear > probaAngry and probaFear > probaNeutral and probaFear > probaDisgusted:
            print('Clase correspondinte: Fear')
            print('Probabilidad Fear' + str(probaFear))
            emocion = 3
        elif probaNeutral > probaHappy and probaNeutral > probaSad and probaNeutral > probaAngry and probaNeutral > probaFear and probaNeutral > probaDisgusted:
            print('Clase correspondinte: Neutral')
            print('Probabilidad Neutral' + str(probaNeutral))
            emocion = 5
        elif probaDisgusted > probaHappy and probaDisgusted > probaSad and probaDisgusted > probaAngry and probaDisgusted > probaFear and probaDisgusted > probaNeutral:
            print('Clase correspondinte: Disgust')
            print('Probabilidad disgust' + str(probaDisgusted))
            emocion = 2

        return emocion

    def matrizConfusion(self):
        confusion = np.zeros((6,6))
        for element in self.nombresValidation:
            landmark = sc.loadmat(element)
            vTem = landmark['faceCoordinatesUnwarped']
            #print(self.mediaP.shape)
            if vTem.shape == (1,136):
                vTem = vTem.T
            vTem = vTem[:,0] * (np.dot(vTem[:,0].T, self.mediaP) / (np.dot(vTem[:,0].T, vTem)))
            vector = np.array([vTem - self.mediaP])

            name = element.split("/")

            #### Clasificacion de los vectores de caracteristicas

            probaClass = 0.18


            probaHappy = self.pHappy(vector)*probaClass
            print('Probabilidad'+str(probaHappy))
            probaSad = self.pSad(vector)*probaClass
            probaAngry = self.pAngry(vector)*probaClass
            probaFear = self.pFear(vector)*probaClass
            probaNeutral = self.pNeutral(vector)*probaClass
            probaDisgusted = self.pDisgusted(vector)*probaClass

            if name[1] == 'happy':
                if probaHappy > probaSad and probaHappy > probaAngry and probaHappy > probaFear and probaHappy > probaNeutral and probaHappy > probaDisgusted:
                    print('Correcto')
                    print('Clase correspondinte: Happy')
                    print('Probabilidad Happy'+str(probaHappy))
                    correcto = correcto + 1
                    confusion[0, 0] = confusion[0, 0] + 1
                elif probaSad > probaHappy and probaSad > probaAngry and probaSad > probaFear and probaSad > probaNeutral and probaSad > probaDisgusted:
                    print('Incorrecto')
                    print('Clase correspondinte: Sad')
                    print('Probabilidad Happy ' + str(probaHappy))
                    print('Probabilidad Sad '+str(probaSad))
                    incorrecto = incorrecto+1
                    confusion[0, 1] = confusion[0, 1] + 1
                elif probaAngry > probaHappy and probaAngry > probaSad and probaAngry > probaFear and probaAngry > probaNeutral and probaAngry > probaDisgusted:
                    print('Incorrecto')
                    print('Clase correspondinte: Angry')
                    print('Probabilidad Happy ' + str(probaHappy))
                    print('Probabilidad Angry ' + str(probaAngry))
                    incorrecto = incorrecto + 1
                    confusion[0, 2] = confusion[0, 2] + 1
                elif probaFear > probaHappy and probaFear > probaSad and probaFear > probaAngry and probaFear > probaNeutral and probaFear > probaDisgusted:
                    print('Incorrecto')
                    print('Clase correspondinte: Fear')
                    print('Probabilidad Happy ' + str(probaHappy))
                    print('Probabilidad Fear ' + str(probaFear))
                    incorrecto = incorrecto + 1
                    confusion[0, 3] = confusion[0, 3] + 1
                elif probaNeutral > probaHappy and probaNeutral > probaSad and probaNeutral > probaAngry and probaNeutral > probaFear and probaNeutral > probaDisgusted:
                    print('Incorrecto')
                    print('Clase correspondinte: Neutral')
                    print('Probabilidad Happy ' + str(probaHappy))
                    print('Probabilidad Neutral ' + str(probaNeutral))
                    incorrecto = incorrecto + 1
                    confusion[0, 4] = confusion[0, 4] + 1
                elif probaDisgusted > probaHappy and probaDisgusted > probaSad and probaDisgusted > probaAngry and probaDisgusted > probaFear and probaDisgusted > probaNeutral:
                    print('Incorrecto')
                    print('Clase correspondinte: Disgusted')
                    print('Probabilidad Happy ' + str(probaHappy))
                    print('Probabilidad Disgusted ' + str(probaDisgusted))
                    incorrecto = incorrecto + 1
                    confusion[0, 5] = confusion[0, 5] + 1


            if name[1] == 'sad':
                if probaHappy > probaSad and probaHappy > probaAngry and probaHappy > probaFear and probaHappy > probaNeutral and probaHappy > probaDisgusted:
                    print('Incorrecto')
                    print('Clase correspondinte: Happy')
                    print('Probabilidad Happy ' + str(probaHappy))
                    print('Probabilidad Sad ' + str(probaSad))
                    incorrecto = incorrecto + 1
                    confusion[1, 0] = confusion[1, 0] + 1
                elif probaSad > probaHappy and probaSad > probaAngry and probaSad > probaFear and probaSad > probaNeutral and probaSad > probaDisgusted:
                    print('Correcto')
                    print('Clase correspondinte: Sad')
                    print('Probabilidad Sad' + str(probaSad))
                    correcto = correcto + 1
                    confusion[1, 1] = confusion[1, 1] + 1
                elif probaAngry > probaHappy and probaAngry > probaSad and probaAngry > probaFear and probaAngry > probaNeutral and probaAngry > probaDisgusted:
                    print('Incorrecto')
                    print('Clase correspondinte: Angry')
                    print('Probabilidad Sad ' + str(probaSad))
                    print('Probabilidad Angry ' + str(probaAngry))
                    incorrecto = incorrecto + 1
                    confusion[1, 2] = confusion[1, 2] + 1
                elif probaFear > probaHappy and probaFear > probaSad and probaFear > probaAngry and probaFear > probaNeutral and probaFear > probaDisgusted:
                    print('Incorrecto')
                    print('Clase correspondinte: Fear')
                    print('Probabilidad Sad ' + str(probaSad))
                    print('Probabilidad Fear ' + str(probaFear))
                    incorrecto = incorrecto + 1
                    confusion[1, 3] = confusion[1, 3] + 1
                elif probaNeutral > probaHappy and probaNeutral > probaSad and probaNeutral > probaAngry and probaNeutral > probaFear and probaNeutral > probaDisgusted:
                    print('Incorrecto')
                    print('Clase correspondinte: Neutral')
                    print('Probabilidad Sad ' + str(probaSad))
                    print('Probabilidad Neutral ' + str(probaNeutral))
                    incorrecto = incorrecto + 1
                    confusion[1, 4] = confusion[1, 4] + 1
                elif probaDisgusted > probaHappy and probaDisgusted > probaSad and probaDisgusted > probaAngry and probaDisgusted > probaFear and probaDisgusted > probaNeutral:
                    print('Incorrecto')
                    print('Clase correspondinte: Disgusted')
                    print('Probabilidad Sad ' + str(probaSad))
                    print('Probabilidad Disgusted ' + str(probaDisgusted))
                    incorrecto = incorrecto + 1
                    confusion[1, 5] = confusion[1, 5] + 1


            if name[1] == 'angry':
                if probaHappy > probaSad and probaHappy > probaAngry and probaHappy > probaFear and probaHappy > probaNeutral and probaHappy > probaDisgusted:
                    print('Incorrecto')
                    print('Clase correspondinte: Happy')
                    print('Probabilidad Happy ' + str(probaHappy))
                    print('Probabilidad Angry ' + str(probaAngry))
                    incorrecto = incorrecto + 1
                    confusion[2, 0] = confusion[2, 0] + 1
                elif probaSad > probaHappy and probaSad > probaAngry and probaSad > probaFear and probaSad > probaNeutral and probaSad > probaDisgusted:
                    print('Incorrecto')
                    print('Clase correspondinte: Sad')
                    print('Probabilidad Happy ' + str(probaHappy))
                    print('Probabilidad Angry '+str(probaAngry))
                    incorrecto = incorrecto + 1
                    confusion[2, 1] = confusion[2, 1] + 1
                elif probaAngry > probaHappy and probaAngry > probaSad and probaAngry > probaFear and probaAngry > probaNeutral and probaAngry > probaDisgusted:
                    print('Correcto')
                    print('Clase correspondinte: Angry')
                    print('Probabilidad Angry' + str(probaAngry))
                    correcto = correcto + 1
                    confusion[2, 2] = confusion[2, 2] + 1
                elif probaFear > probaHappy and probaFear > probaSad and probaFear > probaAngry and probaFear > probaNeutral and probaFear > probaDisgusted:
                    print('Incorrecto')
                    print('Clase correspondinte: Fear')
                    print('Probabilidad Angry ' + str(probaAngry))
                    print('Probabilidad Fear ' + str(probaFear))
                    incorrecto = incorrecto + 1
                    confusion[2, 3] = confusion[2, 3] + 1
                elif probaNeutral > probaHappy and probaNeutral > probaSad and probaNeutral > probaAngry and probaNeutral > probaFear and probaNeutral > probaDisgusted:
                    print('Incorrecto')
                    print('Clase correspondinte: Neutral')
                    print('Probabilidad Angry ' + str(probaAngry))
                    print('Probabilidad Neutral ' + str(probaNeutral))
                    incorrecto = incorrecto + 1
                    confusion[2, 4] = confusion[2, 4] + 1
                elif probaDisgusted > probaHappy and probaDisgusted > probaSad and probaDisgusted > probaAngry and probaDisgusted > probaFear and probaDisgusted > probaNeutral:
                    print('Incorrecto')
                    print('Clase correspondinte: Disgusted')
                    print('Probabilidad Angry ' + str(probaAngry))
                    print('Probabilidad Disgusted ' + str(probaDisgusted))
                    incorrecto = incorrecto + 1
                    confusion[2, 5] = confusion[2, 5] + 1



            if name[1] == 'fear':
                if probaHappy > probaSad and probaHappy > probaAngry and probaHappy > probaFear and probaHappy > probaNeutral and probaHappy > probaDisgusted:
                    print('Incorrecto')
                    print('Clase correspondinte: Happy')
                    print('Probabilidad Happy ' + str(probaHappy))
                    print('Probabilidad Fear ' + str(probaFear))
                    incorrecto = incorrecto + 1
                    confusion[3, 0] = confusion[3, 0] + 1
                elif probaSad > probaHappy and probaSad > probaAngry and probaSad > probaFear and probaSad > probaNeutral and probaSad > probaDisgusted:
                    print('Incorrecto')
                    print('Clase correspondinte: Sad')
                    print('Probabilidad Fear ' + str(probaFear))
                    print('Probabilidad Sad '+str(probaSad))
                    incorrecto = incorrecto + 1
                    confusion[3, 1] = confusion[3, 1] + 1
                elif probaAngry > probaHappy and probaAngry > probaSad and probaAngry > probaFear and probaAngry > probaNeutral and probaAngry > probaDisgusted:
                    print('Incorrecto')
                    print('Clase correspondinte: Angry')
                    print('Probabilidad Fear ' + str(probaFear))
                    print('Probabilidad Angry ' + str(probaAngry))
                    incorrecto = incorrecto + 1
                    confusion[3, 2] = confusion[3, 2] + 1
                elif probaFear > probaHappy and probaFear > probaSad and probaFear > probaAngry and probaFear > probaNeutral and probaFear > probaDisgusted:
                    print('Correcto')
                    print('Clase correspondinte: Fear')
                    print('Probabilidad Fear' + str(probaFear))
                    correcto = correcto + 1
                    confusion[3, 3] = confusion[3, 3] + 1
                elif probaNeutral > probaHappy and probaNeutral > probaSad and probaNeutral > probaAngry and probaNeutral > probaFear and probaNeutral > probaDisgusted:
                    print('Incorrecto')
                    print('Clase correspondinte: Neutral')
                    print('Probabilidad Fear ' + str(probaFear))
                    print('Probabilidad Neutral ' + str(probaNeutral))
                    incorrecto = incorrecto + 1
                    confusion[3, 4] = confusion[3, 4] + 1
                elif probaDisgusted > probaHappy and probaDisgusted > probaSad and probaDisgusted > probaAngry and probaDisgusted > probaFear and probaDisgusted > probaNeutral:
                    print('Incorrecto')
                    print('Clase correspondinte: Disgusted')
                    print('Probabilidad Fear ' + str(probaFear))
                    print('Probabilidad Disgusted ' + str(probaDisgusted))
                    incorrecto = incorrecto + 1
                    confusion[3, 5] = confusion[3, 5] + 1


            if name[1] == 'neutral':
                if probaHappy > probaSad and probaHappy > probaAngry and probaHappy > probaFear and probaHappy > probaNeutral and probaHappy > probaDisgusted:
                    print('Incorrecto')
                    print('Clase correspondinte: Happy')
                    print('Probabilidad Happy ' + str(probaHappy))
                    print('Probabilidad Neutral ' + str(probaNeutral))
                    incorrecto = incorrecto + 1
                    confusion[4, 0] = confusion[4, 0] + 1
                elif probaSad > probaHappy and probaSad > probaAngry and probaSad > probaFear and probaSad > probaNeutral and probaSad > probaDisgusted:
                    print('Incorrecto')
                    print('Clase correspondinte: Sad')
                    print('Probabilidad Neutral ' + str(probaNeutral))
                    print('Probabilidad Sad '+str(probaSad))
                    incorrecto = incorrecto + 1
                    confusion[4, 1] = confusion[4, 1] + 1
                elif probaAngry > probaHappy and probaAngry > probaSad and probaAngry > probaFear and probaAngry > probaNeutral and probaAngry > probaDisgusted:
                    print('Incorrecto')
                    print('Clase correspondinte: Angry')
                    print('Probabilidad Neutral ' + str(probaNeutral))
                    print('Probabilidad Angry ' + str(probaAngry))
                    incorrecto = incorrecto + 1
                    confusion[4, 2] = confusion[4, 2] + 1
                elif probaFear > probaHappy and probaFear > probaSad and probaFear > probaAngry and probaFear > probaNeutral and probaFear > probaDisgusted:
                    print('Incorrecto')
                    print('Clase correspondinte: Fear')
                    print('Probabilidad Neutral ' + str(probaNeutral))
                    print('Probabilidad Fear ' + str(probaFear))
                    incorrecto = incorrecto + 1
                    confusion[4, 3] = confusion[4, 3] + 1
                elif probaNeutral > probaHappy and probaNeutral > probaSad and probaNeutral > probaAngry and probaNeutral > probaFear and probaNeutral > probaDisgusted:
                    print('Correcto')
                    print('Clase correspondinte: Neutral')
                    print('Probabilidad Fear' + str(probaNeutral))
                    correcto = correcto + 1
                    confusion[4, 4] = confusion[4, 4] + 1
                elif probaDisgusted > probaHappy and probaDisgusted > probaSad and probaDisgusted > probaAngry and probaDisgusted > probaFear and probaDisgusted > probaNeutral:
                    print('Incorrecto')
                    print('Clase correspondinte: Disgusted')
                    print('Probabilidad Neutral ' + str(probaNeutral))
                    print('Probabilidad Disgusted ' + str(probaDisgusted))
                    incorrecto = incorrecto + 1
                    confusion[4, 5] = confusion[4, 5] + 1


            if name[1] == 'disgust':
                if probaHappy > probaSad and probaHappy > probaAngry and probaHappy > probaFear and probaHappy > probaNeutral and probaHappy > probaDisgusted:
                    print('Incorrecto')
                    print('Clase correspondinte: Happy')
                    print('Probabilidad Happy ' + str(probaHappy))
                    print('Probabilidad Disgusted ' + str(probaDisgusted))
                    incorrecto = incorrecto + 1
                    confusion[5, 0] = confusion[5, 0] + 1
                elif probaSad > probaHappy and probaSad > probaAngry and probaSad > probaFear and probaSad > probaNeutral and probaSad > probaDisgusted:
                    print('Incorrecto')
                    print('Clase correspondinte: Sad')
                    print('Probabilidad Disgusted ' + str(probaDisgusted))
                    print('Probabilidad Sad '+str(probaSad))
                    incorrecto = incorrecto + 1
                    confusion[5, 1] = confusion[5, 1] + 1
                elif probaAngry > probaHappy and probaAngry > probaSad and probaAngry > probaFear and probaAngry > probaNeutral and probaAngry > probaDisgusted:
                    print('Incorrecto')
                    print('Clase correspondinte: Angry')
                    print('Probabilidad Disgusted ' + str(probaDisgusted))
                    print('Probabilidad Angry ' + str(probaAngry))
                    incorrecto = incorrecto + 1
                    confusion[5, 2] = confusion[5, 2] + 1
                elif probaFear > probaHappy and probaFear > probaSad and probaFear > probaAngry and probaFear > probaNeutral and probaFear > probaDisgusted:
                    print('Incorrecto')
                    print('Clase correspondinte: Fear')
                    print('Probabilidad Disgusted ' + str(probaDisgusted))
                    print('Probabilidad Fear ' + str(probaFear))
                    incorrecto = incorrecto + 1
                    confusion[5, 3] = confusion[5, 3] + 1
                elif probaNeutral > probaHappy and probaNeutral > probaSad and probaNeutral > probaAngry and probaNeutral > probaFear and probaNeutral > probaDisgusted:
                    print('Incorrecto')
                    print('Clase correspondinte: Neutral')
                    print('Probabilidad Disgusted ' + str(probaDisgusted))
                    print('Probabilidad Neutral ' + str(probaNeutral))
                    incorrecto = incorrecto + 1
                    confusion[5, 4] = confusion[5, 4] + 1
                elif probaDisgusted > probaHappy and probaDisgusted > probaSad and probaDisgusted > probaAngry and probaDisgusted > probaFear and probaDisgusted > probaNeutral:
                    print('Correcto')
                    print('Clase correspondinte: Disgusted')
                    print('Probabilidad Disgusted' + str(probaDisgusted))
                    correcto = correcto + 1
                    confusion[5, 5] = confusion[5, 5] + 1
        print(confusion)




    def main(self):
        rospy.init_node('gnc_classifier', anonymous=False)
        rospy.Subscriber('features', Float32MultiArray, self.faturesCallBack)
        pubEmocion = rospy.Publisher('gnc_response', Int16, queue_size=10)
        rate = rospy.Rate(10)
        self.hallarMediaProcrustes()
        self.training(0.1)
        rate.sleep()
        while not rospy.is_shutdown():
            self.graficarLandMarks
            print('Va a comenzar a validar')
            #emocion = self.validation(self.vector)
            self.matrizConfusion()
            #print('La emocion es' + str(emocion))
            msg = Int16()
            msg.data = emocion
            pubEmocion.publish(msg)
            rate.sleep()


if __name__ == '__main__':
    clasificador = gnc()
    clasificador.main()
