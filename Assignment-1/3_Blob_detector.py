from matplotlib import pyplot as plt
from scipy import ndimage,misc
from skimage import feature
import numpy as np
import math, cv2
np.set_printoptions(threshold = np.nan)

def detect_blobs(image):
	images_sigma = []
	sigma = [1, 2.2, 3.4, 4.0, 4.6, 5.2, 5.8, 6.4, 8.5, 11, 15]
	for count in range(11):
		filtered_image = image * laplacian_of_gaussian(image, sigma[count])
		#filtered_image = filtered_image * (scale**2)
		filtered_image = filtered_image
		images_sigma.append(filtered_image)
		#print "count"
	stacked_images = np.dstack(images_sigma)
	print stacked_images.shape
	# print stacked_images
	# print sigma
	lm = feature.peak_local_max(stacked_images, threshold_abs=10, footprint=np.ones((3, 3, 3)), threshold_rel=10, exclude_border=True)
	lm = lm.astype(np.float64)
	lm = np.array(lm)
	lm[:, 2] = (lm[:, 2]).astype(int)
	count = 0
	for x in lm[:,2]:
		lm[count][2] = sigma[int(x)%11]*math.sqrt(2)
		count+=1
	#print lm
	for points in lm:
		cv2.circle(image,(int(points[1]),int(points[0])), int(points[2]), (255,255,255), 1)
	plt.imshow(image)
	plt.title("Blobs"), plt.xticks([]), plt.yticks([])
	plt.show()
	misc.imsave("Blobs_Detection.png",image)
	
def laplacian_of_gaussian(image,sigma):
    f_image = ndimage.filters.gaussian_laplace(image, sigma, output=None, mode='reflect')
    return f_image

def read_image(image_path):
    image = cv2.imread(image_path)
    return image

def display_image(image, title):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Read Image
image = read_image('HW1_Data/HW1_Q3/fishes.jpg')
# display_image(image, "Blob")
detect_blobs(image)
