import csv
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial import distance

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

def compute_visual_words_k_means(train_sift_features, test_sift_features, k_clusters):
	k_means = KMeans(n_clusters = k_clusters)
	k_means.fit(train_sift_features + test_sift_features)
	centroids = k_means.cluster_centers_
	labels = k_means.labels_
	return centroids, labels

def find_similarity(image_sift_feature, visual_words_centroids):
	index = 0
	distances = []
	for feature in visual_words_centroids:
		d = distance.euclidean(feature, image_sift_feature)
		distances.append(d)
	index = distances.index(min(distances))
	return index

def visual_words_representation_of_images(All_sift_features, number_of_features_in_each_image, visual_words_centroids, k_clusters):
	images = []
	count = 0
	i = 0
	image_feature = [0]*k_clusters
	for image_sift_feature in All_sift_features:
		index = find_similarity(image_sift_feature, visual_words_centroids)
		image_feature[index] += 1
		count+=1
		if count == number_of_features_in_each_image[i]:
			images.append(image_feature)
			image_feature = [0]*k_clusters
			count = 0
			i+=1
	return images

def read_labels(path):
	with open(path,'rU') as csvfile:
		csvfile = csv.reader(csvfile, delimiter=',')
		csvdata = list(csvfile)
		return map(int, csvdata[0])

def kNN_classifer(train_images, train_labels, k_NN):
	knn = KNeighborsClassifier(n_neighbors = k_NN)
	knn.fit(train_images, train_labels)
	return knn

def display_confusion_matrix(test_labels, test_prediction):
    print classification_report(test_labels, test_prediction, target_names=['0', '1', '2', '3', '4', '5', '6', '7'])

def display_categorization_accuracy(test_labels, test_prediction):
	categorization_accuracy = accuracy_score(test_labels, test_prediction)
	print categorization_accuracy

# === Main function === #

# sift features of training set
train_features_path = '../../data/train_sift_features'
train_sift_features, number_of_features_in_each_train_image = read_sift_descriptors(train_features_path, 'train', 1888)
print len(number_of_features_in_each_train_image)

# sift features of test set
test_features_path = '../../data/test_sift_features'
test_sift_features, number_of_features_in_each_test_image = read_sift_descriptors(test_features_path, 'test', 800)
print len(number_of_features_in_each_test_image)

# Use K-means to compute visual words # Cluster descriptors
k_clusters = 3
visual_words_centroids, labels = compute_visual_words_k_means(train_sift_features, test_sift_features, k_clusters)
print visual_words_centroids

# Training
# Represent each image by normalized counts of visual words
train_images = visual_words_representation_of_images(train_sift_features, number_of_features_in_each_train_image, visual_words_centroids, k_clusters)
train_labels_path = '../../data/train_labels.csv'
train_labels = read_labels(train_labels_path)

# Train the images using kNN classifer
k_NN = 5
kNN_model = kNN_classifer(train_images, train_labels, k_NN)

# Testing
test_images = visual_words_representation_of_images(test_sift_features, number_of_features_in_each_test_image, visual_words_centroids, k_clusters)
test_prediction = kNN_model.predict(test_images)
test_labels_path = '../../data/test_labels.csv'
test_labels = read_labels(test_labels_path)

# Confusion matrix
display_confusion_matrix(test_labels, test_prediction)

# Categorization Accuracy
display_categorization_accuracy(test_labels, test_prediction)
