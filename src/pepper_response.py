#!/usr/bin/env python

import rospy
from random import randint
import numpy as np
from keyboard import Keyreader
from std_msgs.msg import Int16, String

gnc = 0
nn = 0
kn = 0
svm = 0
w_gnc = 2
w_nn = 2
w_svm = 1
w_kn = 2
say_angry = ["Oh! Veo que estas bravo... Comete una snickers",
             'No estes de mal genio... La vida es una sola...',
             'No te pongas bravo! se te arrugara la frente']
say_disgust = ['Uh... yo tambien siento un poco de asco',
             'Si! Tambien creo que huele mal',
             'Iiuu. Sabes de donde viene ese olor tan feo?']
say_fear = ['No tengas miedo! Yo te protejo',
             'Por que tienes miedo? Soy un robot muy amable',
             'No te asustes. Nadie te hara danio']
say_happy = ['Yo tambien estoy muy feliz hoy!',
             'Oh! Si! La vida es una y hay que ser felices',
             'Me alegra que estes feliz! Seamos amigos']
say_neutral = ['Estas muy serio. Te contare un chiste. Sabes por que Arnold Shuazeneger quiere ser rodilla? Por que silvester es talon'
             'Por que estas tan serio??',
             'no estes tan serio! Sonriele a la vida']
say_sad = ['No estes triste. Te daria un abrazo, pero no control bien mis movimientos',
             'Como dijo Clea Cruz: Ay! No hay que llorar, que la vida es un carnaval... Y es mas bello vivir cantando',
             'Creo que estes triste. Te contare un chiste: Sabes cual es el colmo de un robot? Tener nervios de acero.']


def gnc_callback(data):
    global gnc
    gnc = data.data


def nn_callback(data):
    global nn
    nn = data.data


def kn_callback(data):
    global kn
    kn = data.data


def svm_callback(data):
    global svm
    svm = data.data


def compute_votes():
    response = []
    for i in range(0, w_gnc):
        response.append(gnc)
    for i in range(0, w_nn):
        response.append(nn)
    for i in range(0, w_kn):
        response.append(kn)
    for i in range(0, w_svm):
        response.append(svm)

    angry = response.count(1)
    disgust = response.count(2)
    fear = response.count(3)
    happy = response.count(4)
    neutral = response.count(5)
    sad = response.count(6)
    votes = [angry, disgust, fear, happy, neutral, sad]
    emotion = np.argmax(votes) + 1

    print("Angry: " + str(angry))
    print("Disgust: " + str(disgust))
    print("Fear: " + str(fear))
    print("Happy: " + str(happy))
    print("Neutral: " + str(neutral))
    print("Sad: " + str(sad))

    print(response)
    return emotion


def pepper_node():
    rospy.init_node('pepper_response', anonymous=True)
    rospy.Subscriber('gnc_response', Int16, gnc_callback)
    rospy.Subscriber('NN_response', Int16, nn_callback)
    rospy.Subscriber('KN_response', Int16, kn_callback)
    rospy.Subscriber('SVM_response', Int16, svm_callback)
    pub = rospy.Publisher('speech', String, queue_size=10)
    rate = rospy.Rate(1)
    key = Keyreader()
    key.start()
    while not rospy.is_shutdown():
        pepper_say = ""
        intended = key.getNumber()
        if intended == 0:
            emotion = compute_votes()
            pepper_say = pepper_response(emotion)
        pub.publish(pepper_say)
        print(pepper_say)
        rate.sleep()


def pepper_response(emotion):
    num = randint(0, 2)
    pepper_say = 'No he reconocido tu emocion'
    if emotion == 1:
        pepper_say = say_angry[num]
    if emotion == 2:
        pepper_say = say_disgust[num]
    if emotion == 3:
        pepper_say = say_fear[num]
    if emotion == 4:
        pepper_say = say_happy[num]
    if emotion == 5:
        pepper_say = say_neutral[num]
    if emotion == 6:
        pepper_say = say_sad[num]
    return pepper_say


if __name__ == '__main__':
    try:
        pepper_node()
    except rospy.ROSInterruptException:
        pass
