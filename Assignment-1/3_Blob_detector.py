import numpy as np
import cv2

def detect_blobs(image):

    def blobsAt(image, scale):
            filtered_image = image * laplacian_of_gaussian(scale)
            filtered_image = filtered_image * (scale**2)
        return 0

    scale = 1
    factor = 0.5
    for count in range(10):
        blobsAt(image, scale)
        filtered_image = image * laplacian_of_gaussian(scale)
        filtered_image = filtered_image * (scale**2)
        scale = scale + factor + 0.1
    return 0

def laplacian_of_gaussian(sigma):
    
    return 0

def read_image(image_path):
    image = cv2.imread(image_path, 0)
    return image

def display_image(image, title):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Read Image
image = read_image('HW1_Data/HW1_Q3/butterfly.jpg')

image_with_blobs = detect_blobs(image)
dispay_image(image_with_blobs)
