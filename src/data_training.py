# USAGE
# python facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg 

# import the necessary packages
from imutils import face_utils
from sensor_msgs import Image
from std_msgs import Float32MultiArray
import mat4py
import rospy
import imutils
import dlib
import cv2


class dataTraining:

	def __init__(self):
		self.image = 0
		self.vector = 0
		self.clase = "happy"

	def camaraCallBack(msg):
		global image
		image = msg.data
		pass

	def faturesCallBack(msg):
		global vector
		vector = msg.data
		pass

	def showLandMarks():
		global image
		global vector
		global clase
		id = 0
		# initialize dlib's face detector (HOG-based) and then create
		# the facial landmark predictor
		detector = dlib.get_frontal_face_detector()
		predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
		camara = cv2.VideoCapture(0)
		# load the input image, resize it, and convert it to grayscale

		while True:
			(grabbed, image) = camara.read()
			image = imutils.resize(image, width=500)
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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
				cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

				# show the face number
				cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

				# loop over the (x, y)-coordinates for the facial landmarks
				# and draw them on the image
				for (x, y) in shape:
					cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

			# show the output image with the face detections + facial landmarks
			cv2.imshow("Output", image)
			key = cv2.waitKey(1) & 0xFF
			if key == ord("t"):
				mat4py.savemat(clase+'_vector_'+str(id)'_.mat', vector)
				id = id + 1
			if key == ord("q"):
				break

		camara.release()
		cv2.destroyAllWindows()


	def data_training_py(self):
		rospy.init_node('data_training', anonymous = True)
		rospy.Subscriber('/naoqi_driver/camera/front/image_raw', Image, self.camaraCallBack)
		rospy.Subscriber('features', Float32MultiArray, self.faturesCallBack)
		self.showLandMarks
	
if __name__ == '__main__':
	data_training_py()
