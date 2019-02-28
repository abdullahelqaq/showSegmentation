#-------------IMPORT MODULES-------------#
from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os
#----------------------------------------#

#-------------ARGUMENT PARSING-------------#
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
	help="path to input directory of faces + images")
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-d", "--detection-method", type=str, default="hog",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())
#----------------------------------------#

imagePaths = list(paths.list_images(args["dataset"]))
knownEncodings = []
knownNames = []

for (i, imagePath) in enumerate(imagePaths):
	# extract the person name from the image path
	print("Processing image {}/{}".format(i + 1,
		len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]

	# load the input image and convert it from BGR to RGB
	image = cv2.imread(imagePath)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# detect the (x, y)-coordinates of the bounding boxes
	boxes = face_recognition.face_locations(rgb,
		model=args["detection_method"])

	# compute the facial embedding for the face
	encodings = face_recognition.face_encodings(rgb, boxes)

	# loop over the encodings
	for encoding in encodings:
		knownEncodings.append(encoding)
		knownNames.append(name)

# dump the facial encodings + names to disk
print("Serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open(args["encodings"], "wb")
f.write(pickle.dumps(data))
f.close()