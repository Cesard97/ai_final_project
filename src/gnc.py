#!usr/bin/env python
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

        for file in glob.glob("training_data/disgusted/*.mat"):
            self.nombresLandMarkDisgust.append(file)
            self.nombresLandMark.append(file)

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
		print('vectorIn')

    def graficarLandMarks(self):

        for element in self.nombresLandMark:
            landmark = sc.loadmat(element)
            vTem = landmark['faceCoordinatesUnwarped']
            c = np.sqrt((20 * np.random.rand(68)) ** 2)
            plt.scatter(vTem, marker='o',c = c)

        plt.title('Imagen Land Marks Originales')
        #plt.xticks([]), plt.yticks([])
        plt.grid(True)
        plt.show()


    def hallarMediaProcrustes(self):

        S = 0

        for element in self.nombresLandMark:
            landmark = sc.loadmat(element)
            vTem = landmark['faceCoordinatesUnwarped']
            if vTem.shape == (1,136):
            	vTem = vTem.T
            S = S + np.outer(vTem, vTem)
        [values, vectors] = np.linalg.eig(S)
        index = np.argmax(values)
        self.mediaP = vectors[:, index]
        self.mediaP = self.mediaP.real
        print('Hayó media')


    def training(self, alpha):

        happy = []
        sad = []
        angry = []
        fear = []
        neutral = []
        disgusted = []

        for element in self.nombresLandMark:

            landmark = sc.loadmat(element)
            vTem = landmark['faceCoordinatesUnwarped']
            if vTem.shape == (1,136):
            	vTem = vTem.T
            vTem = vTem * (np.dot(vTem, self.mediaP.T) / (np.dot(vTem, vTem.T)))
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

        ######################## Happy class ##########################

        for i in range(0, len(happy)):
            meanHappy = meanHappy + happy[i]
        meanHappy = meanHappy/len(happy)

        for i in range(0, len(happy)):
            CxHappy = CxHappy + np.outer(happy[i]-meanHappy, happy[i]-meanHappy)
        CxHappy = CxHappy / (len(happy))
        CxHappy = CxHappy + alpha*I


        detHappy = np.linalg.det(CxHappy)
        invHappy = np.linalg.inv(CxHappy)
        invHappy = np.asmatrix(invHappy)
        self.pHappy = lambda x: (1/(np.sqrt(detHappy)))*(1/((2*np.pi)))*np.exp(-0.5*(x - meanHappy).T*invHappy*(x - meanHappy))

        ######################## Sad class ##########################

        for i in range(0,len(sad)):
            meanSad = meanSad + sad[i]
        meanSad = meanSad/len(sad)

        for i in range(0, len(sad)):
            CxSad = CxSad + np.outer(sad[i]-meanSad, sad[i]-meanSad)
        CxSad = CxSad / (len(sad)-1)
        CxSad = CxSad + alpha*I

        detSad = np.linalg.det(CxSad)
        invSad = np.linalg.inv(CxSad)
        invSad = np.asmatrix(invSad)
        self.pSad = lambda x: (1 / (np.sqrt(detSad))) * (1 / ((2 * np.pi))) * np.exp(-0.5 * (x - meanSad).T * invSad * (x - meanSad))

        ######################## Angry class ##########################

        for i in range(0,len(angry)):
            meanAngry = meanAngry + angry[i]
        meanAngry = meanAngry/len(angry)

        for i in range(0, len(angry)):
            CxAngry = CxAngry + np.outer(angry[i]-meanAngry,angry[i]-meanAngry)
        CxAngry = CxAngry / (len(angry)-1)
        CxAngry = CxAngry + alpha*I

        detAngry = np.linalg.det(CxAngry)
        invAngry = np.linalg.inv(CxAngry)
        invAngry = np.asmatrix(invAngry)
        self.pAngry = lambda x: (1 / (np.sqrt(detAngry))) * (1 / ((2 * np.pi))) * np.exp(-0.5 * (x - meanAngry).T * invAngry * (x - meanAngry))

        ######################## Fear class ##########################

        for i in range(0,len(fear)):
            meanFear = meanFear + fear[i]
        meanFear = meanFear/len(fear)

        for i in range(0, len(fear)):
            CxFear = CxFear + np.outer(fear[i]-meanFear,fear[i]-meanFear)
        CxFear = CxFear / (len(fear)-1)
        CxFear = CxFear + alpha*I


        detFear = np.linalg.det(CxFear)
        invFear = np.linalg.inv(CxFear)
        invFear = np.asmatrix(invFear)
        self.pFear = lambda x: (1 / (np.sqrt(detFear))) * (1 / ((2 * np.pi))) * np.exp(-0.5 * (x - meanFear).T * invFear * (x - meanFear))

        ######################## Neutral class ##########################

        for i in range(0,len(neutral)):
            meanNeutral = meanNeutral + neutral[i]
        meanNeutral = meanNeutral/len(neutral)

        for i in range(0, len(neutral)):
            CxNeutral = CxNeutral + np.outer(neutral[i]-meanNeutral,neutral[i]-meanNeutral)
        CxNeutral = CxNeutral / (len(neutral)-1)
        CxNeutral = CxNeutral + alpha*I

        detNeutral = np.linalg.det(CxNeutral)
        invNeutral = np.linalg.inv(CxNeutral)
        invNeutral = np.asmatrix(invNeutral)
        self.pNeutral = lambda x: (1 / (np.sqrt(detNeutral))) * (1 / ((2 * np.pi))) * np.exp(-0.5 * (x - meanNeutral).T * invNeutral * (x - meanNeutral))

        ######################## Disgusted class ##########################

        for i in range(0,len(disgusted)):
            meanDisgusted = meanDisgusted + disgusted[i]
        meanDisgusted = meanDisgusted/len(disgusted)

        for i in range(0, len(disgusted)):
            CxDisgusted = CxDisgusted + np.outer(disgusted[i]-meanDisgusted,disgusted[i]-meanDisgusted)
        CxDisgusted = CxDisgusted / (len(disgusted)-1)
        CxDisgusted = CxDisgusted + alpha*I


        detDisgusted = np.linalg.det(CxDisgusted)
        invDisgusted = np.linalg.inv(CxDisgusted)
        invDisgusted = np.asmatrix(invDisgusted)
        self.pDisgusted = lambda x: (1 / (np.sqrt(detDisgusted))) * (1 / ((2 * np.pi))) * np.exp(-0.5 * (x - meanDisgusted).T * invDisgusted * (x - meanDisgusted))

    	print('Termino training')

    def validation(self):
    	element = 'training_data/happy/happy_vector_0_.mat'
    	landmark = sc.loadmat(element)
    	self.vector = landmark['faceCoordinatesUnwarped']
    	emocion = 0

    	if np.sum(self.vector) == 0:
    		return emocion
    	else: 

        #### Clasificacion de los vectores de caracteristicas

	        probaClass = 1/6


	        probaHappy = self.pHappy(self.vector)*probaClass
	        probaSad = self.pSad(self.vector)*probaClass
	        probaAngry = self.pAngry(self.vector)*probaClass
	        probaFear = self.pFear(self.vector)*probaClass
	        probaNeutral = self.pNeutral(self.vector)*probaClass
	        probaDisgusted = self.pDisgusted(self.vector)*probaClass

	        if probaHappy > probaSad and probaHappy > probaAngry and probaHappy > probaFear and probaHappy > probaNeutral and probaHappy > probaDisgusted:
	            print('Clase correspondinte: Happy')
	            print('Probabilidad Happy'+str(probaHappy))
	            emocion = 4
	        elif probaSad > probaHappy and probaSad > probaAngry and probaSad > probaFear and probaSad > probaNeutral and probaSad > probaDisgusted:
	            print('Clase correspondinte: Sad')
	            print('Probabilidad Sad'+str(probaSad))
	            emocion = 6
	        elif probaAngry > probaHappy and probaAngry > probaSad and probaAngry > probaFear and probaAngry > probaNeutral and probaAngry > probaDisgusted:
	            print('Clase correspondinte: Angry')
	            print('Probabilidad Angry'+str(probaAngry))
	            emocion = 1
	        elif probaFear > probaHappy and probaFear > probaSad and probaFear > probaAngry and probaFear > probaNeutral and probaFear > probaDisgusted:
	            print('Clase correspondinte: Fear')
	            print('Probabilidad Fear'+str(probaFear))
	            emocion = 3
	        elif probaNeutral > probaHappy and probaNeutral > probaSad and probaNeutral > probaAngry and probaNeutral > probaFear and probaNeutral > probaDisgusted:
	            print('Clase correspondinte: Neutral')
	            print('Probabilidad Neutral'+str(probaNeutral))
	            emocion = 5
	        elif probaDisgusted > probaHappy and probaDisgusted > probaSad and probaDisgusted > probaAngry and probaDisgusted > probaFear and probaDisgusted > probaNeutral:
	            print('Clase correspondinte: Disgust')
	            print('Probabilidad disgust'+str(probaDisgusted))
	            emocion = 2

	        return emocion


    def main(self):
    	rospy.init_node('gnc_classifier', anonymous = True)
    	rospy.Subscriber('features', Float32MultiArray, self.faturesCallBack)
    	pubEmocion = rospy.Publisher('gnc_emotion', Int16, queue_size = 1)
    	rate = rospy.Rate(10)
    	self.hallarMediaProcrustes()
    	self.training(1)
    	while not rospy.is_shutdown():
    		self.graficarLandMarks
    		print('Va a comenzar a validar')
    		emocion = self.validation()
    		print('La emocion es'+str(emocion))
    		pubEmocion.publish(emocion)
    		rate.sleep()



if __name__ == '__main__':
    clasificador = gnc()
    clasificador.main()
    
