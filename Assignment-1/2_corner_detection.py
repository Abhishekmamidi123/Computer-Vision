import numpy as np
from matplotlib import pyplot as plt
import cv2
import math
import numpy.linalg as linalg

global corner_points,threshold
corner_points=[]
threshold=900000

def sobel(gray_image):
	kernel = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
	sobelx = cv2.filter2D(gray_image,-1,kernel)
	print sobelx
	sobely = cv2.filter2D(gray_image,-1,np.transpose(kernel))
	return sobelx,sobely

def hessian_matrix_elements(sobelx,sobely):
	#Ix = sobelx.astype(np.uint32)
	#Iy = sobely.astype(np.uint32)
	Ix = sobelx
	Iy = sobely
	print "****"
	print Iy
	A = Ix*Ix
	#B = abs(Ix*Iy)
	C = np.multiply(Iy,Iy)
	print "\n"
	return A,C
	
def get_points(A,C,gray_image):	
	#rows,cols = A.shape
	det = A*C
	print det
	print "--------"
	trace = A+C
	print trace
	print "---------"
	alpha = 0.03
	harris_function = (det - (alpha*trace))
	print harris_function
	#if harris_function>threshold:
		
	rows,cols = harris_function.shape
	for x in range(rows):
		for y in range(cols):
			if harris_function[x][y]>threshold:
				cv2.circle(gray_image,(x,y), 2, (255,255,255), 1)
	cv2.imshow("gray_image",gray_image)
				
	'''for x in range(rows):
		for y in range(cols):
			H = np.array([[A[x][y],B[x][y]],[B[x][y],C[x][y]]])
			print H
			print "-----------"
			eigen_values,_ = linalg.eig(H)
			print eigen_values
			if (min(eigen_values!=0)):
				cv2.circle(gray_image,(x,y), 2, (255,255,255), 1)
			
			lambda_min = min(eigen_values)
			print lambda_min
	cv2.imshow("gray_image",gray_image)'''		
			
gray_image = cv2.imread('Image2.jpg', 0)
print gray_image
sobelx,sobely = sobel(gray_image)

A,C = hessian_matrix_elements(sobelx,sobely)
get_points(A,C,gray_image)
print corner_points

cv2.waitKey(0)                 # Waits forever for user to press any key
cv2.destroyAllWindows()
