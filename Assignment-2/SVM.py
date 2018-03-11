import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.svm import SVC

def read_sift_descriptors(path):
    return 0

def compute_visual_words_k_means(train_sift_features, test_sift_features, k_clusters):
    return 0

def visual_words_representation_of_images(train_sift_features):
    return 0

def read_labels(path):
    return 0

def SVM_classifer(test_features_path):
    return 0

def display_confusion_matrix(test_prediction, test_labels):
    return 0

def display_categorization_accuracy(test_prediction, test_labels):
    return 0

# sift features of training set
train_features_path = ''
train_sift_features = read_sift_descriptors(train_features_path)

# sift features of test set
test_features_path = ''
test_sift_features = read_sift_descriptors(test_features_path)

# Use K-means to compute visual words # Cluster descriptors
k_clusters = 0
visual_words = compute_visual_words_k_means(train_sift_features, test_sift_features, k_clusters)

# Training
# Represent each image by normalized counts of visual words
train_images = visual_words_representation_of_images(train_sift_features)
train_labels_path = ''
train_labels = read_labels(train_labels_path)

# Testing
# SVM to categorize the test images
test_prediction = SVM_classifer(test_features_path)
test_labels_path = ''
test_labels = read_labels(test_labels_path)

# training time and testing time


# Confusion matrix
display_confusion_matrix(test_prediction, test_labels)

# Categorization Accuracy
display_categorization_accuracy(test_prediction, test_labels)

