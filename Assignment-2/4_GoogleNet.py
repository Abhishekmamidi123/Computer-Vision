import cv2
import csv
import numpy as np
import scipy
import pickle
import keras
import time
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from scipy import io as sio
from scipy import ndimage, misc
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from inception_v3 import InceptionV3
from keras.layers import Input
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense, Dropout,Activation,Flatten
from keras.utils.np_utils import to_categorical
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def read_images(images_path, number_of_images):
	images = []
	for i in range(1, number_of_images + 1):
		path = images_path + '/' + str(i) + '.jpg'
		image = scipy.misc.imread(path)
		image = scipy.misc.imresize(image, [229, 229, 3])
		images.append(image)
	images = np.array(images)
	print images.shape
	return images

def read_labels(labels_path):
	with open(labels_path,'rU') as csvfile:
		csvfile = csv.reader(csvfile, delimiter=',')
		csvdata = list(csvfile)
		labels = map(int, csvdata[0])
		return labels

image_height = 256
image_width = 256
channels = 3
train_images_path = '../../../data/train'
train_images = read_images(train_images_path, 1888)
train_images.shape
train_labels_path = '../../../data/train_labels.csv'
train_labels = read_labels(train_labels_path)
print len(train_labels)
# print train_labels

test_images_path = '../../../data/test'
test_images = read_images(test_images_path, 800)
test_images.shape
test_labels_path = '../../../data/test_labels.csv'
test_labels = read_labels(test_labels_path)
print len(test_labels)


image_input = Input(shape=(229, 229, 3))
num_classes = 9
print image_input
print 'Abhishek'
# require_flatten = True
																																					 input_shape=(229, 229, 3))
print 'Abhishek12'
# model.summary()																																			
last_layer = model.get_layer('avg_pool').output
# x = Flatten(name='flatten')(last_layer)
x = last_layer
out = Dense(num_classes, activation='so																																																																									ftmax', name='output_layer')(x)
custom_resnet_model = Model(inputs=image_input,outputs= out)
custom_resnet_model.summary()

for layer in custom_resnet_model.layers[:-1]:
	layer.trainable = False

custom_resnet_model.layers[-1].trainable=True
custom_resnet_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

train_labels = to_categorical(train_labels)
print train_labels.shape
test_labels = to_categorical(test_labels)
t=time.time()
print train_images.shape
hist = custom_resnet_model.fit(train_images, train_labels, batch_size=32, epochs=12, verbose=1, validation_data=(test_images, test_labels))

print('Training time: %s' % (t - time.time()))
(loss, accuracy) = custom_resnet_model.evaluate(test_images, test_labels, batch_size=10, verbose=1)
print loss
print accuracy
