import subprocess
import numpy as np
import cv2
from pyimagesearch import imutils

# define the upper and lower boundaries of the HSV pixel
# intensities to be considered 'skin'
lower = np.array([0, 48, 80], dtype = "uint8")
upper = np.array([20, 255, 255], dtype = "uint8")

# Face Detection
def detect_face(frame, gray):

	# Get current directory path
	pwd = subprocess.check_output("pwd", shell=True).rstrip()

	# Face Detection
	face_cascade = cv2.CascadeClassifier(pwd+'/haarcascades/haarcascade_frontalface_default.xml')
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	face_roi = 0
	face_detected = False
	for (x,y,w,h) in faces:
		face_detected = True
		face_roi = frame[y:y+h,x:x+w]

	# Return the ROI
	return face_detected, face_roi


# Skin Detection
def detect_skin(frame, img_width):
	# Source: http://www.pyimagesearch.com/2014/08/18/skin-detection-step-step-example-using-python-opencv/
	# resize the frame, convert it to the HSV color space,
	# and determine the HSV pixel intensities that fall into
	# the speicifed upper and lower boundaries
	frame = imutils.resize(frame, width = img_width)
	converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	skinMask = cv2.inRange(converted, lower, upper)

	# apply a series of erosions and dilations to the mask
	# using an elliptical kernel
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
	skinMask = cv2.erode(skinMask, kernel, iterations = 2)
	skinMask = cv2.dilate(skinMask, kernel, iterations = 2)

	# blur the mask to help remove noise, then apply the
	# mask to the frame
	skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
	skin = cv2.bitwise_and(frame, frame, mask = skinMask)

	# Return the frame
	return skin

# Define Main function
def main():
	cap = cv2.VideoCapture(0)

	while(True):
		# Capture frame-by-frame
		ret, frame = cap.read()

		# Our operations on the frame come here
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# Detect face and get the ROI
		face_detected, face_roi = detect_face(frame, gray)
		skin = 0
		if (face_detected is True):
			height, width, depth = face_roi.shape

			# Detect skin
			skin = detect_skin(face_roi, width)

		# Display the resulting frame
		cv2.imshow('face_roi',face_roi)
		cv2.imshow('skin',skin)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()

# Call the main function
main()
