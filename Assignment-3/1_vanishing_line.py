import cv2
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
# from PIL import Image

def read_image(path):
	image = scipy.misc.imread(path)
	return image

def read_points(n_points):
	plt.imshow(image)
	points = plt.ginput(4)

# Read image
image_path = 'HW3/img1.jpg'
image = read_image(image_path)

print 'Number of lines?'
n_points = input()
points = read_points(n_points)
