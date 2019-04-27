#ECE 5470 - Final Project - Gordon Keil & Ben CHeng

import numpy as np
import cv2

imgname = "SingleMcDonaldsFries.jpg"          # query image (small object)
imgname2 = "MultipleMcDonaldsFries.jpg" # train image (large scene)

## Create ORB object
orb = cv2.ORB_create()
img1 = cv2.imread(imgname)
img2 = cv2.imread(imgname2)

#Change Colorspace from BGR to Grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

## Find the keypoints and descriptors with ORB
kpts1, descs1 = orb.detectAndCompute(gray1,None)
kpts2, descs2 = orb.detectAndCompute(gray2,None)

## match descriptors and sort them in the order of their distance
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) #bf = created Brute-Force object. BFMatcher = Brute-Force Matcher
matches = bf.match(descs1, descs2) #finds matches in the descrripters for set [arg1] and set [arg2]
dmatches = sorted(matches, key = lambda x:x.distance) #sorts the matches based on their distances

# Lets display the first 15 "best" matches
imgTestMatches = cv2.drawMatches(img1, kpts1, img2, kpts2, dmatches[:20], None, flags=2)
imgTestH, imgTestW = imgTestMatches.shape[:2]
imgTestMatchesResized = cv2.resize(imgTestMatches, (int(imgTestW / 2.7), int(imgTestH / 2.7))) #outputs the resized image
cv2.imshow("Top Matches", imgTestMatchesResized)

'''
## extract the matched keypoints
src_pts  = np.float32([kpts1[m.queryIdx].pt for m in dmatches]).reshape(-1,1,2)
dst_pts  = np.float32([kpts2[m.trainIdx].pt for m in dmatches]).reshape(-1,1,2)

## find homography matrix and do perspective transform
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0) #Note the homography Matrix M needs at least 4 matches
print("Matrix M", M)
h,w = img1.shape[:2]
pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
dst = cv2.perspectiveTransform(pts,M)

h2, w2 = img2.shape[:2]

## draw found regions
img2 = cv2.polylines(img2, [np.int32(dst)], True, (0,0,255), 1, cv2.LINE_AA)
resizedimg2 = cv2.resize(img2, (int(w2 / 1.5), int(h2 / 1.5)))
cv2.imshow("found", img2)

## draw match lines
res = cv2.drawMatches(img1, kpts1, img2, kpts2, dmatches[:20],None,flags=2)
resH, resW = res.shape[:2]

#Resize Image for Easy Viewing
resized = cv2.resize(res, (int(resW / 2.7), int(resH / 2.7))) #outputs the resized image

cv2.imshow("ORB Matches", resized)
'''
cv2.waitKey()
cv2.destroyAllWindows()