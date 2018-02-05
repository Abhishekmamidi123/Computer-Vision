import numpy as np
import cv2
import math
from matplotlib import pyplot as plt

def gaussianFilter(numRows, numCols, sigma, highPass=True):
    centerI = int(numRows/2)
    centerJ = int(numCols/2)

    def gaussian(i,j):
        coefficient = math.exp(-1.0 * ((i - centerI)**2 + (j - centerJ)**2) / (2 * sigma**2))
        return 1 - coefficient if highPass else coefficient

    return np.array([[gaussian(i,j) for j in range(numCols)] for i in range(numRows)])

def extractHighFrequency(image, fshift, alpha):
    numRows, numCols = image.shape
    c_row = numRows/2
    c_col = numCols/2
    
    Filter = gaussianFilter(numRows, numCols, alpha, True)
    filteredShift = fshift * Filter
    f_ishift = np.fft.ifftshift(filteredShift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return img_back

def extractLowFrequency(image, dft_shift, beta):
    numRows, numCols = image.shape
    c_row = numRows/2
    c_col = numCols/2
        
    Filter = gaussianFilter(numRows, numCols, beta, False)
    filteredShift = fshift * Filter
    f_ishift = np.fft.ifftshift(filteredShift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return img_back

def hybridImage(highFrequencyImage, lowFrequencyImage):
    return highFrequencyImage + lowFrequencyImage

def magnitudeSpectrum(image):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*(np.log(np.abs(fshift)))
    return fshift, magnitude_spectrum

def readImage(imagePath):
    image = cv2.imread(imagePath, 0)
    return image

def displayImage(image, title):
    plt.imshow(image, cmap = 'gray')
    plt.title(title), plt.xticks([]), plt.yticks([])
    plt.show()

### Image 1 ###

# Read Image
imagePath_1 = 'HW1_Data/HW1_Q1/einstein.bmp'
image = readImage(imagePath_1)
print image

# Extract Magnitude Spectrum of the image
(fshift, magnitude_spectrum) = magnitudeSpectrum(image)
print magnitude_spectrum

# Extract High Frequncies from the image
alpha = 10
highFrequencyImage = extractHighFrequency(image, fshift, alpha)
displayImage(highFrequencyImage, 'Image after HPF')

### Image 2 ###
imagePath_2 = 'HW1_Data/HW1_Q1/marilyn.bmp'
image = readImage(imagePath_2)
print image

# Extract Magnitude Spectrum of the image
(fshift, magnitude_spectrum) = magnitudeSpectrum(image)
print magnitude_spectrum

# Extract Low Frequncies from the image
beta = 10
lowFrequencyImage = extractLowFrequency(image, fshift, beta)
displayImage(lowFrequencyImage, 'Image after LPF')

# Hybrid Image
hybridImageCombined = hybridImage(highFrequencyImage, lowFrequencyImage)
displayImage(hybridImageCombined, 'Hybrid Image')
