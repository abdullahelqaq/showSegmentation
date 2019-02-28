#-------------IMPORT MODULES-------------#
import face_recognition
import pickle
import cv2
#----------------------------------------#

def main(encodingInput, rgb, detectionMethod): #returns list of names of faces in rgb
	# load the known faces and embeddings
	data = pickle.loads(open(encodingInput, "rb").read())

	# detect the (x, y)-coordinates of the bounding boxes corresponding
	# to each face in the input image, then compute the facial embeddings
	# for each face
	boxes = face_recognition.face_locations(rgb,
		model=detectionMethod)
	encodings = face_recognition.face_encodings(rgb, boxes)

	# initialize the list of names for each face detected
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

			# determine the recognized face with the largest number of
			# votes (note: in the event of an unlikely tie Python will
			# select first entry in the dictionary)
			name = max(counts, key=counts.get)
		
		# update the list of names
		names.append(name)
	return names