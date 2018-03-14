import cv2
import numpy as np
import tensorflow as tf
import scipy
from scipy import io as sio
from scipy import ndimage, misc
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

layers = ['conv1_1_b', 'conv1_1_W', 'conv1_2_b', 'conv1_2_W', 'conv2_1_b', 'conv2_1_W', 'conv2_2_b', 'conv2_2_W', 'conv3_1_b', 'conv3_1_W', 'conv3_2_b', 'conv3_2_W', 'conv3_3_b', 'conv3_3_W', 'conv4_1_b', 'conv4_1_W', 'conv4_2_b', 'conv4_2_W', 'conv4_3_b', 'conv4_3_W', 'conv5_1_b', 'conv5_1_W', 'conv5_2_b', 'conv5_2_W', 'conv5_3_b', 'conv5_3_W', 'fc6_b', 'fc6_W', 'fc7_b', 'fc7_W', 'fc8_b', 'fc8_W']

def read_images(images_path, number_of_images):
	images = []
	for i in range(1, number_of_images + 1):
		path = images_path + '/' + str(i) + '.jpg'
		image = scipy.misc.imread(path)
		images.append(image)
		print np.array(images).shape

#def read_labels(train_labels_path):
	
	

def conv2d(path, previous_layer, layer_no):
	W = np.load(path + '/' + layers[layer_no + 1] + '.npy')
	b = np.load(path + '/' + layers[layer_no] + '.npy')
	layer_name = layers[layer_no][:-2]
	convolution = tf.nn.conv2d(previous_layer, filter = tf.constant(W), strides = [1,1,1,1], padding = 'SAME')
	bias = tf.constant(np.reshape(b, b.size))
	return convolution + bias

def fully_connected(path, previous_layer, layer_no, output_layers):
	W = np.load(path + '/' + layers[layer_no + 1] + '.npy')
	b = np.load(path + '/' + layers[layer_no] + '.npy')
	layer_name = layers[layer_no][:-2]
	fc = tf.contrib.layers.fully_connected(previous_layer, output_layers, activation_fn=tf.nn.relu)
	bias = tf.constant(np.reshape(b, b.size))
	return fc + bias

def generate_model(path, image_height, image_width, channels):
	model = {}
	model['input_image'] = tf.Variable(np.zeros((1, image_height, image_width, channels)), dtype = 'float32')
	print 'input_image  :' + str(model['input_image'].shape)
	model['conv1_1'] = tf.nn.relu(conv2d(path, model['input_image'], 0))
	model['conv1_2'] = tf.nn.relu(conv2d(path, model['conv1_1'], 2))
	model['maxpool1'] = tf.nn.max_pool(model['conv1_2'], ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

	model['conv2_1'] = tf.nn.relu(conv2d(path, model['maxpool1'], 4))
	model['conv2_2'] = tf.nn.relu(conv2d(path, model['conv2_1'], 6))
	model['maxpool2'] = tf.nn.max_pool(model['conv2_2'], ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

	model['conv3_1'] = tf.nn.relu(conv2d(path, model['maxpool2'], 8))
	model['conv3_2'] = tf.nn.relu(conv2d(path, model['conv3_1'], 10))
	model['conv3_3'] = tf.nn.relu(conv2d(path, model['conv3_2'], 12))
	model['maxpool3'] = tf.nn.max_pool(model['conv3_3'], ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

	model['conv4_1'] = tf.nn.relu(conv2d(path, model['maxpool3'], 14))
	model['conv4_2'] = tf.nn.relu(conv2d(path, model['conv4_1'], 16))
	model['conv4_3'] = tf.nn.relu(conv2d(path, model['conv4_2'], 18))
	model['maxpool4'] = tf.nn.max_pool(model['conv4_3'], ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

	model['conv5_1'] = tf.nn.relu(conv2d(path, model['maxpool4'], 20))
	model['conv5_2'] = tf.nn.relu(conv2d(path, model['conv5_1'], 22))
	model['conv5_3'] = tf.nn.relu(conv2d(path, model['conv5_2'], 24))
	model['maxpool5'] = tf.nn.max_pool(model['conv5_3'], ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
	shape = model['maxpool5'].get_shape().as_list()
	model['maxpool5'] = tf.reshape(model['maxpool5'], [1, 1, 1, reduce(lambda x, y: x*y, shape)])
	model['fc6'] = fully_connected(path, model['maxpool5'], 26, 4096)
	model['fc7'] = fully_connected(path, model['fc6'], 28, 4096)
	model['fc8'] = fully_connected(path, model['fc7'], 30, 1000)
	model['softmax'] = tf.nn.softmax(model['fc8'])
	cnt = 0
	for layer_name in layers:
		if cnt%2==0:
			print str(layer_name[:-2]) + '  :' + str(model[layer_name[:-2]].shape)
		cnt+=1
	
	print '\n'
	return model

def train_model(model):
	sess = tf.InteractiveSession()
	sess.run(tf.global_variables_initializer())
	

# Path - VGG16 weights

image_height = 256
image_width = 256
channels = 3
path = '../../pretrained_models/vgg16_pretrained_weights'
model = generate_model(path, image_height, image_width, channels)

train_images_path = '../../data/train'
train_images = read_images(train_images_path, 1888)
train_labels_path = '../../data/test_labels.csv'
train_labels = read_labels(train_labels_path)
train_model(model, train_images, train_labels)

# test_images_path = '../../data/test'
# test_images = read_images(test_images_path, 800)
# test_labels_path = '../../data/test_labels.csv'
# labels = read_labels(test_labels_path)

