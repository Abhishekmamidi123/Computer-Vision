#####################
#Resources:
# 1: http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_sift_intro/py_sift_intro.html
# 2: https://kushalvyas.github.io/stitching.html


#####################
import cv2
import numpy as np
import time



def read_image(image):
	img = cv2.imread(image)
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)	
	return gray
	
def sift_keypoints(gray):
	sift = cv2.xfeatures2d.SIFT_create()
	kp = sift.detect(gray,None)
	return kp




###### Step1: Compute Sift key-points and descriptors for both images.	
start_time = time.time()
#read the images
left_gray = read_image('/home/vubuntu/Documents/Sem-6/CV/End/Assignment3/Assignment-3/uttower_left.JPG')
right_gray = read_image('/home/vubuntu/Documents/Sem-6/CV/End/Assignment3/Assignment-3/uttower_right.JPG')

#extract sift key points of the images
left_kp = sift_keypoints(left_gray)
right_kp = sift_keypoints(right_gray)
#plot keypoints
left_img=cv2.drawKeypoints(left_gray,left_kp,None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
right_img=cv2.drawKeypoints(right_gray,right_kp,None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('left_sift_kps.jpg',left_img)
cv2.imwrite('right_sift_kps.jpg',right_img)

#compute descriptors of the images
sift = cv2.xfeatures2d.SIFT_create()
left_kp1,left_des = sift.compute(left_gray,left_kp)
right_kp1,right_des = sift.compute(right_gray,right_kp)

left_key_pts=[]
for i in left_kp1:
	temp=[]
	temp.append(i.pt[0])
	temp.append(i.pt[1])
	left_key_pts.append(temp)
left_key_pts = np.array(left_key_pts)

right_key_pts=[]
for i in right_kp1:
	temp=[]
	temp.append(i.pt[0])
	temp.append(i.pt[1])
	right_key_pts.append(temp)
right_key_pts = np.array(right_key_pts)


###### Step 2 and 3: Compute distances between every descriptor in one image and every descriptor in the other image.
# and Select putative matches based on the matrix of pairwise descriptor distances obtained above.
f1=[]
f2=[]
count1=0
count2=0
for i in range(len(left_des)):
	count1+=1
	#print "count1 "+str(count1)+"----------------------------"0
	count2=0
	dis_list = []
	for j in range(len(right_des)):
		count2+=1
		#print "count2 "+str(count1)+"-"+str(count2)
		dis = np.linalg.norm(right_des[j]-left_des[i])
		dis_list.append((dis,right_key_pts[j]))
	dis_list.sort(key=lambda x: x[0])
	d1 = dis_list[0][0]
	d2 = dis_list[1][0]
	p1 = dis_list[0][1]
	ratio=round(d1/float(d2),4)
	print str(count1)+" -- "+str(count2)+" : "+str(ratio)
	if ratio < 0.5:
		print "===================="
		f2.append(right_key_pts[j])
		f1.append(left_key_pts[i])
		print "yes"
		
#######	Step 4: Run RANSAC to estimate a homography mapping one image onto the other.

#src_pts = np.float32(f1).reshape(-1,1,2)
#des_pts = np.float32(f2).reshape(-1,1,2)
#f1,f2=np.float32((left_kp,right_kp))

H, __ = cv2.findHomography( left_key_pts, right_key_pts,cv2.RANSAC, 4)
print H
	 
print("--- %s seconds ---" % (time.time() - start_time))
