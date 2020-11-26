import csv
# from PIL import Image
import cv2
# import dlib
# from imutils import face_utils
# import imutils
from sklearn.utils import shuffle
import face_recognition
import sklearn
import pickle

def createData(csvfile):	
	x = []; y = []
	i = 0
	with open(csvfile, 'r') as f:
		reader = csv.reader(f)

		for row in reader:
			i += 1
			print(i)
			imgpath = row[0]
			clas = row[1]
			y.append(clas)

			image = cv2.imread(imgpath)
			rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)		
			boxes = face_recognition.face_locations(rgb)
			top, right, bottom, left = boxes[0]		
			cropped_gray_image = cv2.cvtColor(rgb[top:bottom, left:right], cv2.COLOR_RGB2GRAY)
			norm_image = cv2.normalize(cropped_gray_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
			resized_image = cv2.resize(norm_image, (32, 32))

			x.append(resized_image)
	
	return(shuffle(x, y))
		# print(len(resized_image), len(resized_image[0]))



def create_and_save_training_data(csvfile):
	print("Creating training data.....")
	X_train, y_train = createData(csvfile)

	with open('trainingpickledata.pkl', 'wb') as tr:
		pickle.dump([X_train, y_train], tr)


def create_and_save_validation_data(csvfile):
	print("Creating validation data.....")
	X_valid_norm, y_valid = createData(csvfile)

	with open('testingpickledata.pkl', 'wb') as te:
		pickle.dump([X_valid_norm, y_valid], te)


# We need to call them only once

# create_and_save_training_data('trainImages.csv')
# create_and_save_validation_data('validationImages.csv')
print("Data created")



