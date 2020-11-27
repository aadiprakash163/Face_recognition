import pickle
import csv
import cv2
from sklearn.utils import shuffle
import face_recognition
import sklearn
import pickle
# import tensorflow as tf
import tensorflow.compat.v1 as tf
import numpy as np
tf.disable_v2_behavior()


def cropped_gray_multiple_faces(path):
	image = cv2.imread(imgpath)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)		
	boxes = face_recognition.face_locations(rgb)
	detected_faces = []
	for box in boxes:
		top, right, bottom, left = box		
		cropped_gray_image = cv2.cvtColor(rgb[top:bottom, left:right], cv2.COLOR_RGB2GRAY)
		norm_image = cv2.normalize(cropped_gray_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
		resized_image = cv2.resize(norm_image, (32, 32))
		resized_image = np.asarray(resized_image, dtype = np.float32)
		resized_image = np.stack((resized_image, resized_image, resized_image), axis = -1)	
		detected_faces.append(resized_image)

		
	return(detected_faces)

def evaluate(X_data, y_data):
    """Evaluate accuracy of model on given dataset"""
    total_accuracy = 0.0
    n_data = len(X_data)
    sess = tf.get_default_session()
    for start in range(0, n_data, batch_size):
        end = start + batch_size
        batch_x, batch_y = X_data[start:end], y_data[start: end]
        acc = sess.run(accuracy,feed_dict = {x: batch_x, y: batch_y, dropout : 1.0})
        total_accuracy += (acc*len(batch_x))
    return total_accuracy/n_data

# x = tf.placeholder(tf.float32, (None, 32, 32, 3))
# # x = tf.placeholder(tf.float32, (None, 32, 32))
# y = tf.placeholder(tf.int32,[None])
# dropout = tf.placeholder(tf.float32)
# one_hot_y = tf.one_hot(y,depth=1012, on_value = 1., off_value = 0., axis=-1)

# logits = conv_net(x,dropout)
# cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = one_hot_y, logits = logits)
# cost = tf.reduce_mean(cross_entropy)
# optimizer = tf.train.AdamOptimizer(learning_rate = learn_rate).minimize(cost)

# correct_prediction = tf.equal(tf.argmax(logits,1),tf.argmax(one_hot_y,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

# init = tf.global_variables_initializer()
# saver = tf.train.Saver()

# with open('test2Images.csv', 'r') as f:
# 	reader = csv.reader(f)

# 	for row in reader:
# 		imgpath = row[0]
# 		faces_in_img = cropped_gray_multiple_faces(imgpath)
# 		clas = row[1]
# 		images = []
# 		labels = []
# 		for face in faces_in_img:
# 			images.append(face)
# 			labels.append(clas)

# 		save_file = './learned_model/facerec'
# 		with tf.Session() as sess:
# 		    #sess.run(tf.global_variables_initializer())
# 		    saver.restore(sess,save_file)
# 		    test_accuracy = evaluate(images, labels)
# 		    print("Test Accuracy = {:.3f}".format(test_accuracy))
# 		    exit()


with open('test2Images.csv', 'r') as f:
    reader = csv.reader(f)
    grp_images_data = []
    for row in reader:
        imgpath = row[0]
        faces_in_img = cropped_gray_multiple_faces(imgpath)
        clas = row[1]
        images = []
        labels = []
        for face in faces_in_img:
            images.append(face)
            labels.append(clas)
        grp_images_data.append([images, labels])

save_file = './learned_model/facerec'
with tf.Session() as sess:
    #sess.run(tf.global_variables_initializer())
    saver.restore(sess,save_file)
    for pair in grp_images_data:
    	images = pair[0]
    	labels = pair[1]

    test2_accuracy = evaluate(images, labels)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
