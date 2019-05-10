#!/usr/bin/env python
# USAGE
# python facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg

# import the necessary packages
from imutils import face_utils
from std_msgs.msg import Float32, Float32MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from sklearn import preprocessing
import numpy as np
import rospy
import dlib
import cv2

class preProcesing:

    def __init__(self):
        self.image = np.zeros((500, 500, 3), np.uint8)
        self.bridge = CvBridge()
        self.landmarks = np.zeros((68,2))

    def cameraCallback(self,img):
        self.image = self.bridge.imgmsg_to_cv2(img, "bgr8")
        #self.image = imutils.resize(self.image, width=500)

    def main(self):
        # initialize dlib's face detector (HOG-based) and then create
        # the facial landmark predictor
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('shape_predictor.dat')
        # Node init
        rospy.init_node('pre_proccesing', anonymous=False)
        # Pub and Subs
        pubFeatures = rospy.Publisher('features', Float32MultiArray, queue_size=100)
        rospy.Subscriber("/naoqi_driver/camera/front/image_raw", Image, self.cameraCallback)
        # Vector
        featureVector = Float32MultiArray()
        preProcessLandmarks = np.zeros((136,1))
        # Rate
        rate = rospy.Rate(10)  # 10hz

        while not rospy.is_shutdown():

            featureVector.data = np.zeros((136,1))
            print("No hay nadie en la camara")

            #print(featureVector)
            pubFeatures.publish(featureVector)

            # show the output image with the face detections + facial landmarks
            cv2.imshow("Output", self.image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            rate.sleep()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        pre = preProcesing()
        pre.main()
    except rospy.ROSInterruptException:
        pass