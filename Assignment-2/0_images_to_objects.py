import cv2
import csv
import numpy as np
import tensorflow as tf
import scipy
import pickle
from scipy import io as sio
from scipy import ndimage, misc
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def read_images(images_path, number_of_images):
	images = []
	for i in range(1, number_of_images + 1):
		path = images_path + '/' + str(i) + '.jpg'
		print path
		image = scipy.misc.imread(path)
		images.append(image)
	images = np.array(images)
	print images.shape
	return images
		
train_images_path = '../../data/train'
train_images = read_images(train_images_path, 1888)
print train_images.shape
pickle.dump(train_images, open('../../data/train_images.obj', 'w'))
train_images = pickle.load(open('../../data/train_images.obj', 'r'))
print train_images.shape

test_images_path = '../../data/test'
test_images = read_images(test_images_path, 800)
print test_images.shape
pickle.dump(test_images, open('../../data/test_images.obj', 'w'))
test_images = pickle.load(open('../../data/test_images.obj', 'r'))
print test_images.shape
