import numpy as np
import rospy
import scipy.io as sc
import glob
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers import Activation
from keras.optimizers import Nadam
from keras.optimizers import SGD, RMSprop

nombresLandMark = []
for file in glob.glob("Tarea6/archivosSoporteTarea6/TrainingData/marks/*.mat"):
    nombresLandMark.append(file)

happy = []
sad = []
angry = []
fear = []
neutral = []
disgusted = []

for element in nombresLandMark:

    landmark = sc.loadmat(element)
    vTem = landmark['faceCoordinatesUnwarped']
    media1 = np.mean(vTem[:, 0])
    media2 = np.mean(vTem[:, 1])
    vTem[:, 0] = vTem[:, 0] - media1
    vTem[:, 1] = vTem[:, 1] - media2
    vTem = np.vectorize(complex)(vTem[:, 0], vTem[:, 1])
    vTem = vTem / np.linalg.norm(vTem)
    real = np.array([vTem.real])
    imag = np.array([vTem.imag])
           ### Vector de caracteristicas

    vector = np.concatenate((real.T, imag.T), axis=0)


            ### Divide by name each landmark

    name = element.split("/")
    clas = name[4].split("_")
    print(clas)

            #### Clasificacion de los vectores de caracteristicas

    if clas[3] == 'h':
        happy.append(vector)
    elif clas[3] == 's':
        sad.append(vector)
    elif clas[3] == 'a':
        angry.append(vector)
    elif clas[3] == 'f':
        fear.append(vector)
    elif clas[3] == 'n':
        neutral.append(vector)
    elif clas[3] == 'd':
        disgusted.append(vector)


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


training_data = np.concatenate((angry, disgusted, fear , happy, neutral, sad))

#resultados
target_data = np.concatenate((ones_angry, ones_disgust, ones_fear, ones_happy, ones_neutral, ones_sad))

model = Sequential()

model.add(Dense(20, input_dim=134, activation='relu')) #one hiddel layer
model.add(Dense(20, input_dim= 134, activation='relu'))
model.add(Dense(20, input_dim= 134, activation='relu'))

model.add(Dense(1, activation='linear'))

model.summary()

opt = RMSprop (lr=0.001)

model.compile(optimizer=opt, loss='mean_squared_error',metrics=['accuracy'])
model.fit(training_data[:, :, 0], target_data, batch_size=300, epochs=100000, shuffle=False)

scores = model.evaluate(training_data[:, :, 0], target_data)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print('Clase a la que pertenece el dato 1')
print(target_data[0])
print('Clase predecida por el modelo')
print(training_data[0, :, 0].shape)
print(model.predict(training_data[:, :, 0]).round())


# def neural_network_py():
#   rospy.init.node('neural_network', anonymous= True)
  #  rospy.Subscriber('features',Float32MultiArray, featuresCallBack)


