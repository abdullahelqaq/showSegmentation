###############################################################################
#-----------------Facial Recognition Module-----------------------------------#
# Author: Abdullah Elqaq                                                      #
# Task  : Identify faces that appear in each frame of video input             #
#-----------------------------------------------------------------------------#

#-------------IMPORT MODULES-------------#
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2
import csv
#----------------------------------------#

#-------------ARGUMENT PARSING-------------#
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-i", "--input", required=True,
	help="path to input video")
ap.add_argument("-o", "--output", type=str,
	help="path to output as csv")
ap.add_argument("-d", "--detection-method", type=str, default="hog",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())
#----------------------------------------#

#-------------CLASS DECLARATIONS-------------#
class Result:
	def __init__(self, timestamp, faces):
		self.timestamp = timestamp
		self.faces = faces
#----------------------------------------#


#-------------GLOBAL VARIABLES-------------#

data = pickle.loads(open(args["encodings"], "rb").read())

stream = cv2.VideoCapture(args["input"])
results = []
counter = 0
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
if int(major_ver)  < 3:
	fps = stream.get(cv2.cv.CV_CAP_PROP_FPS)
	frames = stream.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
else:
	fps = stream.get(cv2.CAP_PROP_FPS)
	frames = stream.get(cv2.CAP_PROP_FRAME_COUNT)

#----------------------------------------#


while True:
	counter += 1
	#pull next frame from video, break if finished all frames
	(grabbed, frame) = stream.read()
	if not grabbed:
		break #end of video

	# convert the input frame from BGR to RGB then resize it to have
	# a width of 750px (easier processing)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	rgb = imutils.resize(frame, width=750)
	r = frame.shape[1] / float(rgb.shape[1])

	# detect the (x, y)-coordinates of the bounding boxes
	# corresponding to each face in the input frame, then compute
	# the facial embeddings for each face
	boxes = face_recognition.face_locations(rgb,
		model=args["detection_method"])
	encodings = face_recognition.face_encodings(rgb, boxes)
	names = []

	# loop over the facial embeddings
	for encoding in encodings:
		# attempt to match each face in the input image to our known
		# encodings
		matches = face_recognition.compare_faces(data["encodings"],
			encoding)
		name = "Unknown"

		# check to see if we have found a match
		if True in matches:
			# find the indexes of all matched faces then initialize a
			# dictionary to count the total number of times each face
			# was matched
			matchedIdxs = [i for (i, b) in enumerate(matches) if b]
			counts = {}

			# loop over the matched indexes and maintain a count for
			# each recognized face face
			for i in matchedIdxs:
				name = data["names"][i]
				counts[name] = counts.get(name, 0) + 1

			# determine the recognized face with the largest number
			# of votes (note: in the event of an unlikely tie Python
			# will select first entry in the dictionary)
			name = max(counts, key=counts.get)
		
		# update the list of names
		names.append(name)

	nameString = '-'.join(names)
	result = Result(counter/fps, nameString)
	results.append(result)

stream.release()

csvData = [['Timestamp (s)', 'Faces Identified']]
for i in range(len(results)):
	data = [results[i].timestamp, results[i].faces]
	csvData.append(data)

with open(args["output"], 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(csvData)