######## Resources ###########
# 1 : https://docs.opencv.org/3.1.0/dc/dc3/tutorial_py_matcher.html
# 2 : https://stackoverflow.com/questions/42538914/why-is-ransac-not-working-for-my-code
# 3 : https://docs.opencv.org/3.1.0/d6/d00/tutorial_py_root.html


import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import randrange


img = cv2.imread('/home/vubuntu/Desktop/uttower_left.JPG')
img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

img_ = cv2.imread('/home/vubuntu/Desktop/uttower_right.JPG')
img1 = cv2.cvtColor(img_,cv2.COLOR_BGR2GRAY)


sift = cv2.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2) 



#print matches
# Apply ratio test
good = []
for m in matches:
     if m[0].distance < 0.5*m[1].distance:         
     	good.append(m)
matches = np.asarray(good)
 	 

'''print matches[2,0].queryIdx
print matches[2,0].trainIdx
print matches[2,0].distance'''


if len(matches[:,0]) >= 4:
    src = np.float32([ kp1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
    dst = np.float32([ kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)

    H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    print H
else:
    raise AssertionError("Can't find enough keypoints.")  	
   
dst = cv2.warpPerspective(img_,H,(img.shape[1] + img_.shape[1], img.shape[0]))     	
plt.subplot(122),plt.imshow(dst),plt.title('Warped Image')
plt.show()
plt.figure()
dst[0:img.shape[0], 0:img.shape[1]] = img
cv2.imwrite('resultant_stitched_panorama.jpg',dst)
plt.imshow(dst)
plt.show()
cv2.imwrite('resultant_stitched_panorama.jpg',dst)


'''for i in range(1000):
	pts = []
	ind = []
	H_matrix = []
	j=[]
	while(len(ind)!=4):
		random_index = randrange(0,len(matches))
		if random_index not in ind:
			pts.append(matches[random_index])
			ind.append(random_index)

	src = np.float32([ kp1[m[0].queryIdx].pt for m in pts ]).reshape(-1,1,2)
	dst = np.float32([ kp2[m[0].trainIdx].pt for m in pts ]).reshape(-1,1,2)
	H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
	H_matrix.append(H)
	count=0
	inliers = []
	
	for i in matches:
		c1 = i[0].trainIdx
		c2 = i[0].queryIdx
		count+=1
		print count
		
		p=[]
		p.append(kp1[c2].pt[0])
		p.append(kp1[c2].pt[1])
		p.append(1)
		p = np.array(p)
		p1 = np.matmul(H,p)
		p1 = np.array([p1[0]/float(p1[2]),p1[1]/float(p1[2])])
		
		p2=[]
		p2.append(kp2[c1].pt[0])
		p2.append(kp2[c1].pt[1])
		p2 = np.array(p2)
		print p2
		print "#####"
		d = np.linalg.norm(p1-p2)
		print d
		
		if d<4:
			j.append(i)
	inliers.append(j)		
			
				
max_inlier = max(inliers, key=len)
print "@@@ "+str(len(max_inlier))
max_inlier_index = inliers.index(max_inlier)
best_H = H_matrix[max_inlier_index]
(a,b)=img1.shape
transformed_image = cv2.warpPerspective(img_, best_H,(b,a) )			
plt.subplot(122),plt.imshow(transformed_image),plt.title('Output')
plt.show()'''


