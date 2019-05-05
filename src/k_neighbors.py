#!/usr/bin/env python

import rospy
from sklearn.neighbors import KNeighborsClassifier
from std_msgs.msg import Float32MultiArray, Int16

charac_vector = []


def features_callback(data):
    global charac_vector
    charac_vector = data


def k_neighbors():
    rospy.init_node('k_neighbors', anonymous=False)
    rospy.Subscriber('features', Float32MultiArray, features_callback)
    pub = rospy.Publisher('KN_neighbors', Int16, queue_size=10)
    rate = rospy.Rate(10)
    neigh = KNeighborsClassifier(n_neighbors=2)
    neigh.fit(X, y)

    while not rospy.is_shutdown():
        neigh.predict(charac_vector)
        rate.sleep()


if __name__ == '__main__':
    try:
        k_neighbors()
    except rospy.ROSInterruptException:
        pass
