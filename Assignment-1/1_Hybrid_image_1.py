import cv2
import numpy as np
from matplotlib import pyplot as plt

def extract_high_frequency(image, alpha, beta):
    high_frequency_image = image - extract_low_frequency(image, alpha)
    return high_frequency_image

def extract_low_frequency(image, beta):
    low_frequency_image = cv2.GaussianBlur(image, (7, 7), beta)
    return low_frequency_image

def hybrid_image_combine(high_frequency_image, low_frequency_image):
    hybridImage = high_frequency_image + low_frequency_image
    return hybridImage

def read_image(image_path):
    image = cv2.imread(image_path, 0)
    return image

def display_image(image, title):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

### Image 1 ###

alpha = 10
beta = 10

# Read Image 1
image_path_1 = 'HW1_Data/HW1_Q1/einstein.bmp'
image = read_image(image_path_1)
display_image(image, 'Image1')
# resized_image = cv2.resize(image, (249, 369))
# display_image(resized_image, 'Resized Image 1')

# High Frequency Image
high_frequency_image = extract_high_frequency(image, alpha, beta)
display_image(high_frequency_image, 'HighPass')

# Read Image 2
image_path_2 = 'HW1_Data/HW1_Q1/marilyn.bmp'
image2 = read_image(image_path_2)
display_image(image2, 'Image2')

# Low Frequency Image
low_frequency_image = extract_low_frequency(image2, alpha)
display_image(low_frequency_image, 'LowPass')

hybrid_image =  hybrid_image_combine(high_frequency_image, low_frequency_image)
display_image(hybrid_image, 'Hybrid Image')
