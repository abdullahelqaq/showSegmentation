###############################################################################
#------------------Central Driving Module-------------------------------------#
# Author: Abdullah Elqaq                                                      #
# Task  : Process sequences of boundary cues from other modules               #
#-----------------------------------------------------------------------------#

#-------------IMPORT MODULES-------------#
import cv2
import time
import imutils
import argparse
import csv
import faceRecog
#----------------------------------------#

#-------------ARGUMENT PARSING-------------#
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-i", "--input", required=True,
	help="path to input video")
ap.add_argument("-s", "--show-metadata", required=True,
	help="path to show metadata csv")
ap.add_argument("-o", "--output", type=str,
	help="path to output boundaries as csv")
ap.add_argument("-d", "--detection-method", type=str, default="hog",
	help="face detection model to use: either `hog` or `cnn`") #use hog if no GPU available
args = vars(ap.parse_args())
#----------------------------------------#

#-------------CLASS DECLARATIONS-------------#
class Cue: #simple struct for tracking appearance of show cues
	def __init__(self, cueType, show):
		self.cueType = cueType
		self.show = show
		self.age = 0

class Show: #simple struct to ID shows for easy reference
	def __init__(self, showID, name, logoPath, musicPath, faces):
		self.id = showID
		self.name = name
		self.logoPath = logoPath
		self.musicPath = musicPath
		self.faces = faces
		self.cues = []

		cueCount = 0
		if len(faces) > 0:
			cueCount += 1
		if not logoPath is None:
			cueCount += 1
		if not musicPath is None:
			cueCount += 1

	def addCue(newCue):
		for i in range(len(self.cues)):
			if self.cues[i].cueType == newCue.cueType:
				self.cues[i] = newCue
				break
			else:
				self.cues.append(newCue)

class Boundary: #simple struct to ID shows for easy reference
	def __init__(self, timestamp, prev, current):
		self.timestamp = timestamp #in seconds
		self.prevID = prev
		self.currentID = current

#----------------------------------------#

#-------------GLOBAL VARIABLES (ONLY CHANGE timeBuffer)-------------#
currentShow = -1
timeBuffer = 30 #seconds
counter = 0 #current frame index
shows = [] #list of all shows
boundaries = []

with open(args["show_metadata"], mode='r') as csv_file:
	csv_reader = csv.DictReader(csv_file)
	index = 0
	for row in csv_reader:
		index += 1
		shows.append(Show(index, row["name"], row["logoPath"], row["musicPath"], row["faces"].split(',')))
		

stream = cv2.VideoCapture(args["input"])
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
if int(major_ver)  < 3:
	fps = stream.get(cv2.cv.CV_CAP_PROP_FPS)
else:
	fps = stream.get(cv2.CAP_PROP_FPS)
#----------------------------------------#

#-------------FUNCTION WRAPPERS-------------#
def getFaces(rgb):
	return faceRecog.main(args["encodings"], rgb, args["detection_method"])

def whatMusic():
	#TODO
	return 0

def whatLogo():
	#TODO
	return 0

#----------------------------------------#

while True:

	#increment cue age so as to only consider cues from past timeBuffer seconds
	#if all cues for a non-current show appear within 30 seconds, boundary found
	for i in range(len(shows)):
		for j in range(len(shows[i].cues) - 1, -1, -1):
			shows[i].cues[j].age += 1
			if shows[i].cues[j].age > (timeBuffer*fps):
				del cues[j]
		if shows[i].cueCount == len(shows.cues):
			if not currentShow == shows[i].id:
				boundaries.append(Boundary(counter*fps, currentShow, shows[i].id))
				currentShow = shows[i].id
				break

	#pull next frame from video, break if finished all frames
	(grabbed, frame) = stream.read()
	if not grabbed:
		break

	# convert the input frame from BGR to RGB then resize it to have
	# a width of 750px (easier processing)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	rgb = imutils.resize(frame, width=750)
	r = frame.shape[1] / float(rgb.shape[1])

	#get faces in frame
	faces = getFaces(rgb)

	for i in range(len(shows)):

		# record each face cue under relevant show w/o cue type repitition
		for j in range(len(faces)):	
			if faces[j] in shows[i].faces:
				newCue = Cue("face", show[i].id)
				show[i].addCue(newCue)
				
		#placeholder function; music analysis will require range of video frames (TODO)
		if whatMusic() == show[i].id:
			newCue = Cue("music", show[i].id)
			show[i].addCue(newCue)

		#placeholder function; need to link to current implementation (TODO)
		if whatLogo() == show[i].id:
			newCue = Cue("logo", show[i].id)
			show[i].addCue(newCue)

	counter += 1

stream.release()

#format data into CSV format and write to specified output
csvData = [['Timestamp (s)', 'From', 'To']]
for i in range(len(boundaries)):
	data = []
	data.append(str(boundaries[i].timestamp))
	for j in range(len(shows)):
		if (shows[j].id == boundaries[i].prevID):
			if len(data) == 2:
				tmp = data[1]
				data[1] = shows[j].name
				data.append(tmp)
				break
			else:
				data.append(shows[j].name)
		elif (shows[j].id == boundaries[i].currentID):
			data.append(shows[j].name)
			if len(data) == 3:
				break

with open(args["output"], 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(csvData)
csvFile.close()