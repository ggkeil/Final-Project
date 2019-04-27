# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 10:58:02 2019

@author: eagle
"""

import numpy as np
import cv2

imgname = "SingleMcDonaldsFries.jpg"          # query image (small object)
imgname2 = "MultipleMcDonaldsFries.jpg"       # train image (large scene)

MIN_MATCH_COUNT = 4                           # condition for number of matches to find single object in image

## Create ORB object and BF object(using HAMMING)
orb = cv2.ORB_create()
img1 = cv2.imread(imgname)
img2 = cv2.imread(imgname2)

# Convert both to grayscale first
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

## Find the keypoints and descriptors with ORB
kpts1, descs1 = orb.detectAndCompute(gray1,None)
kpts2, descs2 = orb.detectAndCompute(gray2,None)

## match descriptors and sort them in the order of their distance
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) # Brute Force Matcher
                                                      # NORM_HAMMING is a distance measurement
                                                      
matches = bf.match(descs1, descs2) # get all matches based on descriptors obtained earlier
dmatches = sorted(matches, key = lambda x:x.distance) # sorting happens here

## extract the matched keypoints
# the matched keypoints are in float 32 format
src_pts  = np.float32([kpts1[m.queryIdx].pt for m in dmatches]).reshape(-1,1,2) # matched keypoints from query image
dst_pts  = np.float32([kpts2[m.trainIdx].pt for m in dmatches]).reshape(-1,1,2) # matched keypoints from training image

## find homography matrix and do perspective transform
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
h,w = img1.shape[:2]
pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
dst = cv2.perspectiveTransform(pts,M)

# Grabbing the dimensions of 2nd image
h2, w2 = img2.shape[:2]

## draw found regions
img2 = cv2.polylines(img2, [np.int32(dst)], True, (0,0,255), 1, cv2.LINE_AA)

# for resizing the "found" window
resizedimg2 = cv2.resize(img2, (int(w2 / 2.7), int(h2 / 2.7)))
cv2.imshow("found", img2)

## draw match lines
res = cv2.drawMatches(img1, kpts1, img2, kpts2, dmatches[:20],None,flags=2)

# Grab dimensions of result image
resH, resW = res.shape[:2]

# resize the resulting image
resized = cv2.resize(res, (int(resW / 2.7), int(resH / 2.7)))

# show the resized resulting image
cv2.imshow("orb_match", resized)

cv2.waitKey()
cv2.destroyAllWindows()