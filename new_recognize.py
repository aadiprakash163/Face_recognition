import os
import csv
import dlib
import scipy.misc
import numpy as np

face_detector = dlib.get_frontal_face_detector()

shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

face_recognition_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

TOLERANCE = 0.6



# def get_face_encodings(path_to_image):
def face_det_info(path_to_image):
    
    image = scipy.misc.imread(path_to_image)
    
    detected_faces = face_detector(image, 1)
    num_faces = len(detected_faces)
    
    # shapes_faces = [shape_predictor(image, face) for face in detected_faces]

    
    # return [np.array(face_recognition_model.compute_face_descriptor(image, face_pose, 1)) for face_pose in shapes_faces]
    return(num_faces)


def generateData():
	subfolders = os.listdir("trainset/")
	fields = ['Path to image', 'Class_index', 'Class', 'Num_faces']
	c_index = 0

	with open('trainImages.csv', 'w') as traincsv, open('testImages.csv', 'w') as testcsv, open('test2Images.csv', 'w') as test2csv:
		trainwriter = csv.writer(traincsv)
		testwriter = csv.writer(testcsv)
		test2writer = csv.writer(test2csv)

		trainwriter.writerow(fields)
		testwriter.writerow(fields)
		test2writer.writerow(fields)


		for subfolder in subfolders:	
			subsubfolders = os.listdir("trainset/" + subfolder)
			for subsubfolder in subsubfolders:
				
				c_index += 1
				images = os.listdir("trainset/" + subfolder+ "/" + subsubfolder)
				onetrain = False
				onetest = False
				for image in images:
					path_to_image = "trainset/" + subfolder + "/" + subsubfolder + "/" + image
					row = [path_to_image, c_index, subsubfolder]
					print(path_to_image)
					try:
						if face_det_info(path_to_image) == 1:							
							if not onetrain:
								trainwriter.writerow(row)
								onetrain = True
							if onetrain and not onetest:
								testwriter.writerow(row)
								onetest = True
							if onetest:
								trainwriter.writerow(row)

						else:
							test2writer.writerow(row)
					except Exception:
						pass

# generateData()

