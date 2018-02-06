import numpy as np
from matplotlib import pyplot as plt
import cv2
import math
import numpy.linalg as linalg

global corner_points,threshold
corner_points=[]
threshold=3

def gray_image(image):
	gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	return gray_image/255.0

def sobel(gray_image):
	kernel = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
	sobelx = cv2.filter2D(gray_image,-1,kernel)
	sobely = cv2.filter2D(gray_image,-1,np.transpose(kernel))
	#print sobely
	#print "---------"
	return sobelx,sobely

def matrix_elements(sobelx,sobely):
	Ix = sobelx
	Iy = sobely
	A = Ix*Ix
	B = abs(Ix*Iy)
	C = np.multiply(Iy,Iy)
	print "\n"
	print C
	return A,B,C
	
def get_points(A,B,C,gray_image):	
	rows,cols = A.shape
	for x in range(rows):
		for y in range(cols):
			H = np.array([[A[x][y],B[x][y]],[B[x][y],C[x][y]]])
			eigen_values,_ = linalg.eig(H)
			if (min(eigen_values!=0)):
				cv2.circle(gray_image,(x,y), 2, (255,255,255), 1)
			
			lambda_min = min(eigen_values)
			print lambda_min
			
			
gray_image = cv2.imread('Image2.jpg', 0)
sobelx,sobely = sobel(gray_image)

A,B,C = matrix_elements(sobelx,sobely)
get_points(A,B,C,gray_image)
print corner_points

cv2.waitKey(0)                 # Waits forever for user to press any key
cv2.destroyAllWindows()
