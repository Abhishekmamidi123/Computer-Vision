import cv2
import csv
import numpy as np
import scipy
import pickle
import keras
from keras.preprocessing import image
from imagenet_utils import preprocess_input
from scipy import io as sio
from scipy import ndimage, misc
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from vgg16 import VGG16


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
train_labels_path = '../../data/test_labels.csv'
train_labels = read_labels(train_labels_path)
print train_labels


test_images_path = '../../data/test'
test_images = read_images(test_images_path, 800)
test_labels_path = '../../data/test_labels.csv'
labels = read_labels(test_labels_path)
