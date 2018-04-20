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
	
def plot_vanishing_line():
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
	
	# Plot the vanishing points
	plt.plot([vanishing_points[0][0], vanishing_points[1][0]], [vanishing_points[0][1], vanishing_points[1][1]], marker = 'o')
	
	# Find Equation
	m = slope(vanishing_points[0], vanishing_points[1])
	c = y_intercept(vanishing_points[0], m)
	print '\n'
	print 'Equation:'
	equation = 'y = ' + str(m) + '*x + ' + str(c)
	print equation
	return vanishing_points

# Read image
# image_path = 'HW3/img1.jpg'
image_path = 'HW3/img2.jpg'
image = read_image(image_path)

# Pole points
plt.imshow(image)
points = plt.ginput(2)
pole_top = points[0]
pole_bottom = points[1]
print points
plt.plot([pole_top[0], pole_bottom[0]], [pole_top[1], pole_bottom[1]], marker = 'o')

# Tractor points
points = plt.ginput(2)
object_top = points[0]
object_bottom = points[1]
print points
plt.plot([object_top[0], object_bottom[0]], [object_top[1], object_bottom[1]], marker = 'o')

vanishing_points = plot_vanishing_line()
print vanishing_points

# Line equation - object bottom, pole bottom
A1 = pole_bottom
A2 = object_bottom
B1 = vanishing_points[0]
B2 = vanishing_points[1]
point_on_vl = intersection_point(A1, A2, B1, B2)
plt.plot([object_bottom[0], point_on_vl[0]], [object_bottom[1], point_on_vl[1]], marker = 'o')
plt.plot([object_top[0], point_on_vl[0]], [object_top[1], point_on_vl[1]], marker = 'o')
print point_on_vl

A1 = pole_top
A2 = pole_bottom
B1 = object_top
B2 = point_on_vl
height_object_point = intersection_point(A1, A2, B1, B2)

object_height = 1.65
height_of_object = object_height * (np.linalg.norm(np.array(height_object_point)-np.array(pole_bottom))*1.0/np.linalg.norm(np.array(pole_top)-np.array(pole_bottom)))
print height_of_object

# Height of camera
A1 = pole_bottom
A2 = pole_top
B1 = vanishing_points[0]
B2 = vanishing_points[1]
point_on_vl = intersection_point(A1, A2, B1, B2)
height_of_camera = object_height * (np.linalg.norm(np.array(point_on_vl)-np.array(pole_bottom))*1.0/np.linalg.norm(np.array(pole_top)-np.array(pole_bottom)))
print height_of_camera
plt.show()
