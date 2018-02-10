from matplotlib import pyplot as plt
import numpy as np
import cv2
import math

def extractHighFrequency(image, fshift, alpha):
    numRows, numCols = image.shape
    c_row = numRows/2
    c_col = numCols/2
    
    Filter = gaussianFilter(numRows, numCols, alpha, True)
    Filter = np.fft.fft2(Filter)
    filteredShift = fshift * Filter
    f_ishift = np.fft.ifftshift(filteredShift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return img_back

def extractLowFrequency(image, fshift, beta):
    numRows, numCols = image.shape
    c_row = numRows/2
    c_col = numCols/2
        
    Filter = gaussianFilter(numRows, numCols, beta, False)
    Filter = np.fft.fft2(Filter)
    filteredShift = fshift * Filter
    filteredShift = fshift * np.fft.fft2(Filter)
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

def gaussianFilter(numRows, numCols, sigma, highPass=True):
    centerI = int(numRows/2)
    centerJ = int(numCols/2)

    def gaussian(i,j):
        coefficient = math.exp(-1.0 * ((i - centerI)**2 + (j - centerJ)**2) / (2 * sigma**2))
        if highPass:
        	return 1 - coefficient
        else:
        	return coefficient

    return np.array([[gaussian(i,j) for j in range(numCols)] for i in range(numRows)])

def readImage(imagePath):
    image = cv2.imread(imagePath, 0)
    return image

def displayImage(image, title):
    plt.imshow(image, cmap = 'gray')
    plt.title(title), plt.xticks([]), plt.yticks([])
    plt.show()

def hybrid_image(imagePath_1, imagePath_2, alpha, beta):
	image = readImage(imagePath_1)
	numRows, numCols = image.shape
	(fshift, magnitude_spectrum) = magnitudeSpectrum(image)
	highFrequencyImage = extractHighFrequency(image, fshift, alpha)
	# displayImage(highFrequencyImage, 'Image after HPF')
	image = readImage(imagePath_2)
	image = cv2.resize(image, (numCols, numRows), interpolation = cv2.INTER_AREA)
	(fshift, magnitude_spectrum) = magnitudeSpectrum(image)
	lowFrequencyImage = extractLowFrequency(image, fshift, beta)
	# displayImage(lowFrequencyImage, 'Image after LPF')
	hybridImageCombined = hybridImage(highFrequencyImage, lowFrequencyImage)
	displayImage(hybridImageCombined, 'Hybrid Image')

def hybrid_image_high(imagePath_1, imagePath_2, imagePath_3, gamma, delta, beta):
	image = readImage(imagePath_1)
	numRows, numCols = image.shape
	(fshift, magnitude_spectrum) = magnitudeSpectrum(image)
	highFrequencyImage_1 = extractHighFrequency(image, fshift, gamma)
	
	image = readImage(imagePath_2)
	image = cv2.resize(image, (numCols, numRows), interpolation = cv2.INTER_AREA)
	(fshift, magnitude_spectrum) = magnitudeSpectrum(image)
	highFrequencyImage_2 = extractHighFrequency(image, fshift, delta)
	
	image = readImage(imagePath_3)
	image = cv2.resize(image, (numCols, numRows), interpolation = cv2.INTER_AREA)
	(fshift, magnitude_spectrum) = magnitudeSpectrum(image)
	lowFrequencyImage = extractLowFrequency(image, fshift, beta)
	
	hybridImageCombined = hybridImage(highFrequencyImage_1, highFrequencyImage_2)
	hybridImageCombined = hybridImage(lowFrequencyImage, hybridImageCombined)
	displayImage(hybridImageCombined, 'Hybrid Image_H_H_L')

def hybrid_image_low(imagePath_1, imagePath_2, imagePath_3, gamma, delta, beta):
	image = readImage(imagePath_1)
	numRows, numCols = image.shape
	(fshift, magnitude_spectrum) = magnitudeSpectrum(image)
	lowFrequencyImage_1 = extractLowFrequency(image, fshift, gamma)
	
	image = readImage(imagePath_2)
	image = cv2.resize(image, (numCols, numRows), interpolation = cv2.INTER_AREA)
	(fshift, magnitude_spectrum) = magnitudeSpectrum(image)
	lowFrequencyImage_2 = extractLowFrequency(image, fshift, delta)
	
	image = readImage(imagePath_3)
	image = cv2.resize(image, (numCols, numRows), interpolation = cv2.INTER_AREA)
	(fshift, magnitude_spectrum) = magnitudeSpectrum(image)
	highFrequencyImage = extractHighFrequency(image, fshift, beta)
	
	hybridImageCombined = hybridImage(lowFrequencyImage_1, lowFrequencyImage_2)
	hybridImageCombined = hybridImage(highFrequencyImage, hybridImageCombined)
	displayImage(hybridImageCombined, 'Hybrid Image_L_L_H')

imagePath_1 = 'HW1_Data/HW1_Q1/einstein.bmp'
imagePath_2 = 'HW1_Data/HW1_Q1/marilyn.bmp'
alpha = 10
beta = 10
hybrid_image(imagePath_1, imagePath_2, alpha, beta)

imagePath_1 = 'HW1_Data/HW1_Q1/cat.bmp'
imagePath_2 = 'HW1_Data/HW1_Q1/dog.bmp'
imagePath_3 = 'HW1_Data/HW1_Q1/bird.bmp'
gamma = 10
delta = 10
beta = 10
hybrid_image_high(imagePath_1, imagePath_2, imagePath_3, gamma, delta, beta)

gamma = 10
delta = 10
beta = 10
hybrid_image_low(imagePath_1, imagePath_2, imagePath_3, gamma, delta, beta)
