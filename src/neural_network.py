import numpy as np
import rospy
import scipy.io as sc
import glob
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.optimizers import Adam

nombresLandMark=[]
for file in glob.glob("archivosSoporteTarea6/TrainingData/marks/*.mat"):
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

    vector = np.concatenate((real.T,imag.T), axis=0)


            ### Divide by name each landmark

    name = element.split("/")
    clas = name[3].split("_")

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
ones_angry=np.transpose(ones_angry)
ones_disgust = np.ones(len(disgusted))
ones_disgust=ones_disgust*2
ones_disgust=np.transpose(ones_disgust)
ones_fear = np.ones(len(fear))
ones_fear=ones_fear*3
ones_fear=np.transpose(ones_fear)
ones_happy = np.ones(len(happy))
ones_happy=ones_happy*4
ones_happy=np.transpose(ones_happy)
ones_neutral = np.ones(len(neutral))
ones_neutral=ones_neutral*5
ones_neutral=np.transpose(ones_neutral)
ones_sad = np.ones(len(sad))
ones_sad=ones_sad*6
ones_sad=np.transpose(ones_sad)

#angry = np.asmatrix(angry)

#disgust = np.asmatrix(disgusted)

#fear = np.asmatrix(fear)

#sad = np.asmatrix(sad)

#happy = np.asmatrix(happy)

#neutral = np.asmatrix(neutral)




training_data = np.concatenate((angry, disgusted, fear, happy, neutral, sad))

#resultados
target_data = np.concatenate((ones_angry, ones_disgust, ones_fear, ones_happy, ones_neutral, ones_sad))

model = Sequential()

model.add(Dense(16, input_dim=134, activation='relu')) #one hiddel layer

model.add(Dense(6, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['binary_accuracy'])
model.fit(training_data[:, :, 0], target_data, epochs=1000)

scores = model.evaluate(training_data[:, :, 0], target_data)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print(model.predict(training_data[:, :, 0]).round())


# def neural_network_py():
#   rospy.init.node('neural_network', anonymous= True)
  #  rospy.Subscriber('features',Float32MultiArray, featuresCallBack)


