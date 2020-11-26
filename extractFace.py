# import os
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
# import tensorflow as tf
import tensorflow.compat.v1 as tf
import numpy as np
tf.disable_v2_behavior()

def cropped_gray_single_face(path):
	image = cv2.imread(imgpath)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)		
	boxes = face_recognition.face_locations(rgb)
	top, right, bottom, left = boxes[0]		
	cropped_gray_image = cv2.cvtColor(rgb[top:bottom, left:right], cv2.COLOR_RGB2GRAY)
	norm_image = cv2.normalize(cropped_gray_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
	resized_image = cv2.resize(norm_image, (32, 32))	
	return(resized_image)

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
		detected_faces.append(resized_image)

		
	return(detected_faces)

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

			resized_image = cropped_gray_single_face(imgpath)

			x.append(resized_image)
	
	return(shuffle(x, y))
		# print(len(resized_image), len(resized_image[0]))
		
		
def conv2d(x,W,b,strides = 1):
    """Convolution layer wraper"""
    conv_net = tf.nn.conv2d(x,W,strides = [1, strides, strides, 1],padding = 'SAME' )
    conv_net = tf.nn.bias_add(conv_net,b)
    return conv_net

def maxpool2d(x, k = 2):
    """Maxpool wraper"""
    mp = tf.nn.max_pool(x,ksize = [1, k, k, 1],strides = [1, k, k, 1],padding = 'SAME')
    return mp

# from tensorflow.contrib.layers import flatten
def conv_net(x,dropout):
    """Convolution network model"""
    cn1 = conv2d(x,weights['wc1'],biases['bc1'])
    cn1 = tf.nn.relu(cn1)
    cn1 = maxpool2d(cn1, k=2)
    

    cn2 = conv2d(cn1,weights['wc2'],biases['bc2'])
    cn2 = tf.nn.relu(cn2)
    cn2 = maxpool2d(cn2, k=2)

    # fc0 = flatten(cn2)
    fc0 = tf.reshape(cn2, [tf.shape(cn2)[0], -1]) 

    fc1 = tf.add(tf.matmul(fc0,weights['wfc1']),biases['bfc1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1,dropout)
    
    fc2 = tf.add(tf.matmul(fc1,weights['wfc2']),biases['bfc2'])
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2,dropout)

    out = tf.add(tf.matmul(fc2,weights['out']),biases['out'])

    return out

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

learn_rate = 0.001
batch_size = 128
n_classes = 1012


weights = {
    'wc1' : tf.Variable(tf.random.truncated_normal([5,5,3,32],mean = 0, stddev = 0.1)),
    # 'wc1' : tf.Variable(tf.random.truncated_normal([5,5,3,32],mean = 0, stddev = 0.1)),
    'wc2' : tf.Variable(tf.random.truncated_normal([5,5,32,64],mean = 0, stddev = 0.1)),
    'wfc1' : tf.Variable(tf.random.truncated_normal([8*8*64,1024],mean = 0, stddev = 0.1)),
    'wfc2' : tf.Variable(tf.random.truncated_normal([1024,400],mean = 0, stddev = 0.1)),
    'out'  : tf.Variable(tf.random.truncated_normal([400, n_classes], mean = 0, stddev = 0.1))
}

biases = {
    'bc1' : tf.Variable(tf.zeros(32)),
    'bc2' : tf.Variable(tf.zeros(64)),
    'bfc1': tf.Variable(tf.zeros(1024)),
    'bfc2': tf.Variable(tf.zeros(400)),
    'out' : tf.Variable(tf.zeros(n_classes))
}

x = tf.placeholder(tf.float32, (None, 32, 32, 3))
# x = tf.placeholder(tf.float32, (None, 32, 32))
y = tf.placeholder(tf.int32,[None])
dropout = tf.placeholder(tf.float32)
one_hot_y = tf.one_hot(y,depth=1012, on_value = 1., off_value = 0., axis=-1)

logits = conv_net(x,dropout)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = one_hot_y, logits = logits)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = learn_rate).minimize(cost)

correct_prediction = tf.equal(tf.argmax(logits,1),tf.argmax(one_hot_y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()


print("Loading the training and validation data from pickle file")
with open('trainingpickledata.pkl', 'rb') as f:
	trdata = pickle.load(f)

[X_train_norm, y_train] = trdata
# X_train_norm = np.asarray(X_train_norm, dtype = np.float32).reshape(len(X_train_norm), 32, 32, 1)
X_train_norm = np.asarray(X_train_norm, dtype = np.float32)
X_train_norm = np.stack((X_train_norm, X_train_norm, X_train_norm), axis = -1)

# print(X_train_norm.shape)
# exit()

with open('testingpickledata.pkl', 'rb') as f2:
	tedata = pickle.load(f2)

[X_valid_norm, y_valid] = tedata
X_valid_norm = np.asarray(X_valid_norm, dtype = np.float32)
X_valid_norm = np.stack((X_valid_norm, X_valid_norm, X_valid_norm), axis = -1)


train_size = len(X_train_norm)

print("Starting learning")
save_file = './lenet'
with tf.Session() as sess:
    sess.run(init)

    print('Training....')
    print()
    epochs = 0
    validation_accuracy = 0.0
    while validation_accuracy < 0.8 :
        epochs += 1
        X_train_norm, y_train = shuffle(X_train_norm,y_train)
        for start in range(0,train_size,batch_size):
            end = start + batch_size
            batch_x, batch_y = X_train_norm[start:end],y_train[start:end]
            sess.run(optimizer, feed_dict = {x: batch_x, y: batch_y, dropout: 0.75})

        
        validation_accuracy = evaluate(X_valid_norm, y_valid)
        print("EPOCH {} ...".format(epochs))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
    
    saver.save(sess, save_file)
    print("Model saved")