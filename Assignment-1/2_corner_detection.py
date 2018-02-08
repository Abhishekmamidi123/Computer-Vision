from matplotlib import pyplot as plt
import numpy.linalg as linalg
from scipy import ndimage,misc
import numpy as np
import cv2

global window_size, alpha, threshold 
alpha = 0.04
window_size = 3
threshold = 0.5

def readImage(imagePath):
    image = cv2.imread(imagePath)
    return image

def displayImage(title, image):
	plt.imshow(image)
	plt.title(title)
	plt.show()
	misc.imsave(str(title)+"___"+str(alpha)+".png",image)

def findGradients(image):
	I_x = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=5)
	I_y = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=5)
	return I_x, I_y

def window_hessian(row, col, center, Ix_square, Iy_square, Ix_Iy):
        Ix2 = Ix_square[row-center:row+center+1, col-center:col+center+1]
        Ixy = Ix_Iy[row-center:row+center+1, col-center:col+center+1]
        Iy2 = Iy_square[row-center:row+center+1, col-center:col+center+1]
        return Ix2.sum(), Ixy.sum(), Iy2.sum()

def gradient_squares(Ix, Iy):
	Ix_square = Ix*Ix
	Iy_square = Iy*Iy
	Ix_Iy = Ix*Iy
	return Ix_square, Iy_square, Ix_Iy

def harris_corners(image):
	global threshold
	center = window_size/2
	numRows = image.shape[0]
	numCols = image.shape[1]
	harris_img = image.copy()
	Ix, Iy = findGradients(image)
	Ix_square, Iy_square, Ix_Iy = gradient_squares(Ix, Iy)
	R= np.minimum(Ix_square, Iy_square)
	threshold = threshold * R.max()
	for row in range(center, numRows-center):
		for col in range(center, numCols-center):
			A, B, C = window_hessian(row, col, center, Ix_square, Iy_square, Ix_Iy)
			Det = A*C -B*B
			Trace = A + C
			f = Det - alpha*(Trace*Trace)
			if f > threshold*100000000:
				harris_img.itemset((row, col, 0), 0)
				harris_img.itemset((row, col, 1), 0)
				harris_img.itemset((row, col, 2), 255)
	return harris_img

def shi_tomasi_corners(image):
	global threshold
	center = window_size/2
	numRows = image.shape[0]
	numCols = image.shape[1]
	st_img = image.copy()
	Ix, Iy = findGradients(image)
	Ix_square, Iy_square, Ix_Iy = gradient_squares(Ix, Iy)
	R= np.minimum(Ix_square, Iy_square)
	threshold = threshold * R.max()
	for row in range(center, numRows-center):
		for col in range(center, numCols-center):
			A, B, C = window_hessian(row, col, center, Ix_square, Iy_square, Ix_Iy)
			H = np.array([[A,B],[B,C]])
			eigen = np.linalg.eigvals(H)
			f = min(eigen)
			if f > threshold:
				st_img.itemset((row, col, 0), 0)
				st_img.itemset((row, col, 1), 0)
				st_img.itemset((row, col, 2), 255)
	return st_img

def lambda_method_corners(image):
	global threshold
	center = window_size/2
	numRows = image.shape[0]
	numCols = image.shape[1]
	lambda_img = image.copy()
	Ix, Iy = findGradients(image)
	Ix_square, Iy_square, Ix_Iy = gradient_squares(Ix, Iy)
	R= np.minimum(Ix_square, Iy_square)
	threshold = threshold * R.max()
	for row in range(center, numRows-center):
		for col in range(center, numCols-center):
			A, B, C = window_hessian(row, col, center, Ix_square, Iy_square, Ix_Iy)
			H = np.array([[A,B],[B,C]])
			eigen = np.linalg.eigvals(H)
			min_eigen = min(eigen)
			Det = eigen[0]*eigen[1]
			Trace = eigen[0]+eigen[1]
			f = Det-(alpha*Trace)
			if f > threshold*100000000:
				lambda_img.itemset((row, col, 0), 0)
				lambda_img.itemset((row, col, 1), 0)
				lambda_img.itemset((row, col, 2), 255)
	return lambda_img

image = readImage('butterfly.jpg')
img = harris_corners(image)
displayImage("Harris", img)
img = shi_tomasi_corners(image)
displayImage("Shi-Tomasi", img)
img = lambda_method_corners(image)
displayImage("lambda", img)
