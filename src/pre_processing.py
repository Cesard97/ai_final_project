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
import imutils
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
        rospy.init_node('preProccesing', anonymous=True)
        # Pub and Subs
        pubFeatures = rospy.Publisher('features', Float32MultiArray, queue_size=10)
        rospy.Subscriber("/naoqi_driver/camera/bottom/image_raw", Image, self.cameraCallback)
        # Vector
        featureVector = Float32MultiArray()
        preProcessLandmarks = np.zeros((136,1))
        # Rate
        rate = rospy.Rate(10)  # 10hz

        while not rospy.is_shutdown():
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            # detect faces in the grayscale image
            rects = detector(gray, 1)
            # loop over the face detections
            for (i, rect) in enumerate(rects):
                # determine the facial landmarks for the face region, then
                # convert the facial landmark (x, y)-coordinates to a NumPy
                # array
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                self.landmarks = shape
                # convert dlib's rectangle to a OpenCV-style bounding box
                # [i.e., (x, y, w, h)], then draw the face bounding box
                (x, y, w, h) = face_utils.rect_to_bb(rect)
                cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # show the face number
                cv2.putText(self.image, "Face #{}".format(i + 1), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                # loop over the (x, y)-coordinates for the facial landmarks
                # and draw them on the image
                for (x, y) in shape:
                    cv2.circle(self.image, (x, y), 1, (0, 255, 0), -1)

            # Pre-processing
            for i in range(0,67):
                preProcessLandmarks[i] = (self.landmarks[i,0])
            for i in range(68,136):
                preProcessLandmarks[i] = (self.landmarks[i-68,1])

            preProcessLandmarks = preprocessing.scale(preProcessLandmarks)

            print(preProcessLandmarks)
            featureVector.data = preProcessLandmarks
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