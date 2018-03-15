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
from resnet50 import ResNet50


model = VGG16(weights='imagenet', include_top=False)

