# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 10:58:02 2019

@author: eagle
"""

import numpy as np
import cv2
<<<<<<< HEAD
<<<<<<< HEAD

imgname = "AmtrakAngled.jpg"          # query image (small object)
imgname2 = "AmtrakRef.jpg"       # train image (large scene)


imgname = "SingleMcDonaldsFries.jpg"          # query image (small object)
imgname2 = "MultipleMcDonaldsFries.jpg"       # train image (large scene)


=======
imgname = "SingleMcDonaldsFries.jpg"          # query image (small object)
imgname2 = "MultipleMcDonaldsFries.jpg"       # train image (large scene)

>>>>>>> parent of b463f3d... Added More Comments
=======
imgname = "SingleMcDonaldsFries.jpg"          # query image (small object)
imgname2 = "MultipleMcDonaldsFries.jpg"       # train image (large scene)

>>>>>>> parent of b463f3d... Added More Comments
MIN_MATCH_COUNT = 4

## Create ORB object and BF object(using HAMMING)
orb = cv2.ORB_create()
img1 = cv2.imread(imgname)
img2 = cv2.imread(imgname2)

gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

## Find the keypoints and descriptors with ORB
kpts1, descs1 = orb.detectAndCompute(gray1,None)
kpts2, descs2 = orb.detectAndCompute(gray2,None)

## match descriptors and sort them in the order of their distance
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descs1, descs2)
dmatches = sorted(matches, key = lambda x:x.distance)

## extract the matched keypoints
src_pts  = np.float32([kpts1[m.queryIdx].pt for m in dmatches]).reshape(-1,1,2)
dst_pts  = np.float32([kpts2[m.trainIdx].pt for m in dmatches]).reshape(-1,1,2)

## find homography matrix and do perspective transform
M = cv2.getPerspectiveTransform(src_pts, dst_pts)

dst = cv2.warpPerspective(img1, M, (300,300))

h2, w2 = img2.shape[:2]

## draw found regions
img2 = cv2.polylines(img2, [np.int32(dst)], True, (0,0,255), 1, cv2.LINE_AA)
resizedimg2 = cv2.resize(img2, (int(w2 / 1.5), int(h2 / 1.5)))
cv2.imshow("found", img2)

## draw match lines
res = cv2.drawMatches(img1, kpts1, img2, kpts2, dmatches[:20],None,flags=2)

resH, resW = res.shape[:2]

<<<<<<< HEAD
<<<<<<< HEAD

# resize the resulting image
resized = cv2.resize(res, (int(resW / 3.3), int(resH / 3.3)))

=======
>>>>>>> parent of b463f3d... Added More Comments
=======
>>>>>>> parent of b463f3d... Added More Comments
resized = cv2.resize(res, (int(resW / 2.7), int(resH / 2.7)))

cv2.imshow("orb_match", resized);

cv2.waitKey()
cv2.destroyAllWindows()