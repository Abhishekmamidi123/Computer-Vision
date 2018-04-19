import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import warnings

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
# image_path = 'HW3/img1.jpg'
image_path = 'img1.jpg'
image = read_image(image_path)

# Read points
print 'Number of lines = 6'
n_lines = 6
n_points = n_lines*2
points = read_points(n_points)
print points

# Find intersection points
vanishing_points = []
print range(len(points))
for i in range(0, len(points), n_lines):
	A1 = points[i]
	A2 = points[i+1]
	B1 = points[i+2]
	B2 = points[i+3]
	C1 = points[i+4]
	C2 = points[i+5]
	plt.plot([A1[0], A2[0]], [A1[1], A2[1]], marker = 'o')
	plt.plot([B1[0], B2[0]], [B1[1], B2[1]], marker = 'o')
	plt.plot([C1[0], C2[0]], [C1[1], C2[1]], marker = 'o')
	p11, p12 = intersection_point(A1, A2, B1, B2)
	p21, p22 = intersection_point(B1, B2, C1, C2)
	p31, p32 = intersection_point(C1, C2, A1, A2)
	p1 = float(p11 + p21 + p31)/3.0
	p2 = float(p12 + p22 + p32)/3.0
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
