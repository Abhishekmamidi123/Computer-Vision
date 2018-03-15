import cv2
import csv
import numpy as np
import scipy
import pickle
import keras
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from scipy import io as sio
from scipy import ndimage, misc
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from resnet50 import ResNet50


def read_images(images_path, number_of_images):
	images = []
	for i in range(1, number_of_images + 1):
		path = images_path + '/' + str(i) + '.jpg'
		image = scipy.misc.imread(path)
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
train_labels_path = '/home/abhishek/Desktop/SEM-6/CV/data/test_labels.csv'
train_labels = read_labels(train_labels_path)
print train_labels

test_images_path = '../../../data/test'
test_images = read_images(test_images_path, 800)
test_labels_path = '/home/abhishek/Desktop/SEM-6/CV/data/test_labels.csv'
labels = read_labels(test_labels_path)

model = ResNet50(input_tensor=image_input, include_top=True,weights='imagenet')
model.summary()
last_layer = model.get_layer('avg_pool').output
x= Flatten(name='flatten')(last_layer)
out = Dense(num_classes, activation='softmax', name='output_layer')(x)
custom_resnet_model = Model(inputs=image_input,outputs= out)
custom_resnet_model.summary()

for layer in custom_resnet_model.layers[:-1]:
	layer.trainable = False

custom_resnet_model.layers[-1].trainable

custom_resnet_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

t=time.time()
hist = custom_resnet_model.fit(X_train, y_train, batch_size=32, epochs=12, verbose=1, validation_data=(X_test, y_test))
print('Training time: %s' % (t - time.time()))
(loss, accuracy) = custom_resnet_model.evaluate(X_test, y_test, batch_size=10, verbose=1)
