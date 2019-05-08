#!/usr/bin/env python
import numpy as np
import scipy.io as sc
import glob
import rospy
from std_msgs.msg import Float32, Float32MultiArray, Int16


from std_msgs.msg import Float32, Float32MultiArray, Int16
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.optimizers import SGD, RMSprop
global featuresCallBack
global vector
global emocion
global model
global training_data
global target_data

class neuralNetwork:

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

        for file in glob.glob("training_data/disgust/*.mat"):
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
        self.vector = np.asmatrix(self.vector)
        self.vector = self.vector/2
        pass

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

    def training(self):

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
            print(vector.shape)

            #### Clasificacion de los vectores de caracteristicas
            if name[1] == 'happy':
                if happy == []:
                    happy = vector
                else: 
                    happy=np.concatenate((happy,vector))
            elif name[1] == 'sad':
                if sad == []:
                    sad = vector
                else: 
                    sad=np.concatenate((sad,vector))
            elif name[1] == 'angry':
                if angry == []:
                    angry = vector
                else: 
                    angry=np.concatenate((angry,vector))
            elif name[1] == 'fear':
                if fear == []:
                    fear = vector
                else: 
                    fear=np.concatenate((fear,vector))
            elif name[1] == 'neutral':
                if neutral == []:
                    neutral = vector
                else: 
                    neutral=np.concatenate((neutral,vector))
            elif name[1] == 'disgust':
                if disgusted == []:
                    disgusted = vector
                else: 
                    disgusted=np.concatenate((disgusted,vector))
        #datos de entrada
        ones_angry = np.ones(len(angry))
        ones_angry = np.transpose(ones_angry)
        ones_disgust = np.ones(len(disgusted))
        ones_disgust = ones_disgust*2
        ones_disgust = np.transpose(ones_disgust)
        ones_fear = np.ones(len(fear))
        ones_fear = ones_fear*3
        ones_fear = np.transpose(ones_fear)
        ones_happy = np.ones(len(happy))
        ones_happy = ones_happy*4
        ones_happy = np.transpose(ones_happy)
        ones_neutral = np.ones(len(neutral))
        ones_neutral = ones_neutral*5
        ones_neutral = np.transpose(ones_neutral)
        ones_sad = np.ones(len(sad))
        ones_sad = ones_sad*6
        ones_sad = np.transpose(ones_sad)

        print(happy.shape)
        #happy = np.asmatrix(happy)
        #sad = np.asmatrix(sad)
        #fear = np.asmatrix(fear)
        #neutral = np.asmatrix(neutral)
        #angry = np.asmatrix(angry)
        #disgust = np.asmatrix(disgusted)

        training_data= np.concatenate((happy, sad, fear, neutral, disgusted, angry))
        target_data= np.concatenate((ones_happy, ones_sad, ones_fear, ones_neutral, ones_disgust, ones_angry))

        model=Sequential()

        model.add(Dense(50, input_dim=136, activation='relu')) #one hiddel layer
        model.add(Dense(50, input_dim=136, activation='relu'))
        model.add(Dense(50, input_dim=136, activation='relu'))
        model.add(Dense(50, input_dim=136, activation='relu'))
        model.add(Dense(50, input_dim=136, activation='relu'))
        model.add(Dense(50, input_dim=136, activation='relu'))


        model.add(Dense(1, activation='linear'))

        model.summary()

        opt = SGD (lr=0.0001)

        model.compile(optimizer=opt, loss='mean_squared_error',metrics=['accuracy'])
        model.fit(training_data, target_data, epochs=1000, shuffle=False)


    def validation(self, element):

        emocion=0

        return emocion

    def main(self):
        rospy.init_node('neural_network_validation', anonymous=False)
        rospy.Subscriber('features', Float32MultiArray, self.faturesCallBack)
        pubEmocion = rospy.Publisher('NN_response', Int16, queue_size=1)
        rate = rospy.Rate(10)
        self.hallarMediaProcrustes()
        self.training()
        while not rospy.is_shutdown():
            print('Va a comenzar a validar')
            emocion = self.validation(self.vector)
            print('La emocion es' + str(emocion))
            pubEmocion.publish(emocion)
            rate.sleep()


if __name__ == '__main__':
    clasificador = neuralNetwork()
    clasificador.main()
