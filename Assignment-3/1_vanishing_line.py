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
	points = plt.ginput(n_points)
	return points

def slope(P1, P2):
	return(P2[1] - P1[1]) / (P2[0] - P1[0])

def y_intercept(P1, slope):
	return P1[1] - slope * P1[0]

def line_intersect(m1, b1, m2, b2):
	if m1 == m2:
		print ("These lines are parallel!!!")
		return None
	x = (b2 - b1) / (m1 - m2)
	y = m1 * x + b1
	return x,y

def intersection_point(A1, A2, B1, B2):
	slope_A = slope(A1, A2)
	slope_B = slope(B1, B2)
	y_int_A = y_intercept(A1, slope_A)
	y_int_B = y_intercept(B1, slope_B)
	return line_intersect(slope_A, y_int_A, slope_B, y_int_B)

# Read image
image_path = 'HW3/img1.jpg'
image = read_image(image_path)

# Read points
print 'Number of lines = 4'
n_lines = 4
n_points = 8
points = read_points(n_points)
print points

# Find intersection points
vanishing_points = []
print range(len(points))
for i in range(0, len(points), 4):
	A1 = points[i]
	A2 = points[i+1]
	B1 = points[i+2]
	B2 = points[i+3]
	plt.plot([A1[0], A2[0]], [A1[1], A2[1]], marker = 'o')
	plt.plot([B1[0], B2[0]], [B1[1], B2[1]], marker = 'o')
	p1, p2 = intersection_point(A1, A2, B1, B2)
	vanishing_points.append((p1, p2))
print vanishing_points

# Plot the vanishing points
plt.plot([vanishing_points[0][0], vanishing_points[1][0]], [vanishing_points[0][1], vanishing_points[1][1]], marker = 'o')
plt.show()

# Find Equation
m = slope(vanishing_points[0], vanishing_points[1])
c = y_intercept(vanishing_points[0], m)
print '\n'
print 'Equation:'
equation = 'y = ' + str(m) + '*x + ' + str(c)
print equation
