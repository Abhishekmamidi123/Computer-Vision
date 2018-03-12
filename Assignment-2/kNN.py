import csv
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def read_sift_descriptors(path, name, number_of_images):
	sift_features_of_all_images = []
	number_of_features_in_each_image = []
	for i in range(1, number_of_images + 1):
		train_file = path + '/' + str(i) + '_' + name + '_sift.csv'
		csvfile = open(train_file, 'rb')
		csvfile = csv.reader(csvfile)
		number_of_features = 0
		for x in csvfile:
			sift_features_of_all_images.append(map(int, x[4:]))
			number_of_features+=1
		number_of_features_in_each_image.append(number_of_features)
	return sift_features_of_all_images, number_of_features_in_each_image
	# print len(sift_features_of_all_images)

def compute_visual_words_k_means(train_sift_features, test_sift_features, k_clusters):
    return 0

def visual_words_representation_of_images(train_sift_features):
    return 0

def read_labels(path):
    return 0

def kNN_classifer(test_features_path, k_NN):
    return 0

def display_confusion_matrix(test_prediction, test_labels):
    return 0

def display_categorization_accuracy(test_prediction, test_labels):
    return 0

# sift features of training set
train_features_path = '../../data/train_sift_features'
train_sift_features, number_of_features_in_each_train_image = read_sift_descriptors(train_features_path, 'train', 1888)
print len(number_of_features_in_each_train_image)

# sift features of test set
test_features_path = '../../data/test_sift_features'
test_sift_features, number_of_features_in_each_test_image = read_sift_descriptors(test_features_path, 'test', 800)
print (number_of_features_in_each_test_image)

# Use K-means to compute visual words # Cluster descriptors
k_clusters = 100
visual_words = compute_visual_words_k_means(train_sift_features, test_sift_features, k_clusters)

# Training
# Represent each image by normalized counts of visual words
train_images = visual_words_representation_of_images(train_sift_features)
train_labels_path = ''
train_labels = read_labels(train_labels_path)

# Testing
# kNN to categorize the test images
k_NN = 0
test_prediction = kNN_classifer(test_features_path, k_NN)
test_labels_path = ''
test_labels = read_labels(test_labels_path)

# Confusion matrix
display_confusion_matrix(test_prediction, test_labels)

# Categorization Accuracy
display_categorization_accuracy(test_prediction, test_labels)

