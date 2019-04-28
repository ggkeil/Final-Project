# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 10:58:02 2019

@author: eagle
"""

import numpy as np
import cv2

imgname = "WhiteBoard1.jpg"             # query image (small object)
imgname2 = "WhiteBoard2.jpg"            # train image (large scene)

img1 = cv2.imread(imgname)
img2 = cv2.imread(imgname2)

MAX_FEATURES = 500

def alignImages(img1, img2, numGoodMatches):
    ## Create ORB object and BF object(using HAMMING)
    # BF: Brute Force
    # Alternative to BF is FLANN
    orb = cv2.ORB_create(MAX_FEATURES)
    # Convert images to grayscale
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    
    ## Find the keypoints and descriptors with ORB
    kpts1, descs1 = orb.detectAndCompute(gray1,None)
    kpts2, descs2 = orb.detectAndCompute(gray2,None)
    
    ## match descriptors and sort them in the order of their distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) # Brute Force Matching
    matches = bf.match(descs1, descs2) # brute force match with the two descriptors found earlier
    dmatches = sorted(matches, key = lambda x:x.distance) # where the sorting happens
    
    ## extract the matched keypoints
    # the matched keypoints are in numpy float32 arrays
    src_pts  = np.float32([kpts1[m.queryIdx].pt for m in dmatches]).reshape(-1,1,2)
    dst_pts  = np.float32([kpts2[m.trainIdx].pt for m in dmatches]).reshape(-1,1,2)
    
    ## find homography matrix and do perspective transform
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0) # M is the homography matrix
                                                                   # not doing anything with mask at the moment
    h,w = img1.shape[:2]
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts, M)
    
    # Draw found regions
    img2 = cv2.polylines(img2, [np.int32(dst)], True, (0,0,255), 1, cv2.LINE_AA)
    cv2.imshow("found", img2)
    
    ## draw match lines
    res = cv2.drawMatches(img1, kpts1, img2, kpts2, dmatches[:20],None,flags=2)
    
    resH, resW = res.shape[:2]
    
    # resize the resulting image to fit on laptop screen
    resized = cv2.resize(res, (int(resW / 3.3), int(resH / 3.3)))
    
    # show the resulting image
    cv2.imshow("orb_match", resized)
    
    # Use homography
    height, width = img2.shape[:2]
    height = height * 2
    width = width * 2
    final = cv2.warpPerspective(img1, M, (width, height))
    final[0:img2.shape[0], 0:img2.shape[1]] = img2
    
    cv2.namedWindow('Final Image', cv2.WINDOW_NORMAL)
    cv2.imshow("Final Image", final)
    
    cv2.waitKey()
    cv2.destroyAllWindows()
    
alignImages(img1, img2, 4)