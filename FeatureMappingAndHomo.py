#ECE 5470 - Final Project - Gordon Keil & Ben CHeng

import numpy as np
import cv2

## Create ORB object
orb = cv2.ORB_create()
#img1 = cv2.imread(r'AmtrakRef.jpg') # query image (small object)
#img1 = cv2.imread(r'AmtrakRef2.png') # query image (small object)
#img2 = cv2.imread(r'AmtrakAngled.jpg') # train image (large scene)
img1 = cv2.imread(r'mount1.png') # query image (small object)
img2 = cv2.imread(r'mount2.png') # train image (large scene)


#Change Colorspace from BGR to Grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

## Find the keypoints and descriptors with ORB
kpts1, descs1 = orb.detectAndCompute(gray1,None)
kpts2, descs2 = orb.detectAndCompute(gray2,None)

## match descriptors and sort them in the order of their distance
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) #bf = created Brute-Force object. BFMatcher = Brute-Force Matcher
matches = bf.match(descs1, descs2) #finds matches in the descripters for set [arg1] and set [arg2]
dmatches = sorted(matches, key = lambda x:x.distance) #sorts the matches based on their distances

## extract the matched keypoints
src_pts  = np.float32([kpts1[m.queryIdx].pt for m in dmatches]).reshape(-1,1,2)
dst_pts  = np.float32([kpts2[m.trainIdx].pt for m in dmatches]).reshape(-1,1,2)

## find homography matrix and do perspective transform
M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0) #Note the homography Matrix M needs at least 4 matches
print("Matrix M", M)
h,w = img1.shape[:2]
pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
dst = cv2.perspectiveTransform(pts,M)
print("Dst: ", dst)

## #Draw the lines of where Image 1 "should be" on Image 2.
img2 = cv2.polylines(img2, [np.int32(dst)], True, (180,105,255), 10, cv2.LINE_AA)
img2H, img2W = img2.shape[:2]

'''
cv2.namedWindow('found', cv2.WINDOW_NORMAL)
cv2.resizeWindow('found', int(img2W/8) , int(img2H/8))
cv2.imshow("found", img2)
'''

## Create an single image with both Img1 and Img2 side-by-side, with the matches drawn in between them.
res = cv2.drawMatches(img1, kpts1, img2, kpts2, dmatches[:20],None,flags=2)
cv2.namedWindow('ORB Matches', cv2.WINDOW_KEEPRATIO)
cv2.imshow("ORB Matches", res)

finalImage = cv2.warpPerspective(img1, M, (int(img2W*1.5), int(img2H*1.5)) )
finalImage2 = cv2.warpPerspective(img1, M, (int(img2W*1.5), int(img2H*1.5)) )
finalImage2[0:img2.shape[0], 0:img2.shape[1]] = img2

cv2.namedWindow('Final Image', cv2.WINDOW_NORMAL)
cv2.imshow("Final Image", finalImage2)



cv2.waitKey()
cv2.destroyAllWindows()