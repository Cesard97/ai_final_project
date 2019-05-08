#!/usr/bin/env python
# license removed for brevity
# import the necessary packages
from imutils import face_utils
from std_msgs.msg import Float32, Float32MultiArray
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import scipy.io as sio
import numpy as np
import rospy
import imutils
import dlib
import cv2


class dataTraining:

	def __init__(self):
		self.image = np.zeros((500, 500, 3), np.uint8)
		self.vector = 0
		self.clase = "angry"
		self.bridge = CvBridge()

	def camaraCallBack(self, img):
		self.image = self.bridge.imgmsg_to_cv2(img, "bgr8")
		#self.image = imutils.resize(self.image, width=500)
		print('imagenIn')

	def faturesCallBack(self, msg):
		self.vector = msg.data
		print('vectorIn')

	def showLandMarks(self):
		id = 0
		detector = dlib.get_frontal_face_detector()
		print('noshape')
		predictor = dlib.shape_predictor('shape_predictor.dat')
		print('shape')
		while True:
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

			# show the output image with the face detections + facial landmarks
			cv2.imshow("Output", self.image)

			key = cv2.waitKey(1) & 0xFF
			if key == ord("t"):
				dic = {'faceCoordinatesUnwarped':self.vector}
				sio.savemat('training_data/'+self.clase+'/'+self.clase +'_vector_'+str(id)+'_.mat', dic)
				id = id + 1
				print('Guardando vector')

			if key == ord("q"):
				break
		cv2.destroyAllWindows()

	def data_training_py(self):
		rospy.init_node('data_training', anonymous = True)
		rospy.Subscriber('/naoqi_driver/camera/front/image_raw', Image, self.camaraCallBack)
		rospy.Subscriber('features', Float32MultiArray, self.faturesCallBack)
		rate = rospy.Rate(10)
		print('no ha entrado')
		while not rospy.is_shutdown():
			self.showLandMarks()
			rate.sleep()

	
if __name__ == '__main__':
	data = dataTraining()
	data.data_training_py()
