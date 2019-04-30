#ECE 5470 - Final Project - Gordon Keil & Ben CHeng

import numpy as np
import cv2

import imutils

def readImages():
    filenames = ["mount1.jpg",
                 "mount2.jpg"
                ] # 2 images, change filenames when you want different set of images
    
    images = [] # array setup to store the images with the above filenames

    for filename in filenames: # looping through the filenames provided
        im = cv2.imread(filename) # point to the image with the current filename
        images.append(im) # add this image to the images array
    
    return images # return the images array

# attempt at getting rid of the black around the stitched image
# displays the final stitched image after cropping
def crop(stitched):
    # First, create a 10 pixel border surrounding the stitched image
    stitched = cv2.copyMakeBorder(stitched, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0))
    
    # convert the stitched image to grayscale and threshold it
    # such that all pixels greater than zero are set to 255 (foreground)
    # others remain 0 (background)
    gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
    retVal, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    
    # find all external contours in the threshold image then find
    # the largest contour which will be the contour/outline of
    # the stitched image
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # first argument: image
                                                                                           # second argument: Contour Retrieval mode
                                                                                           # RETR_EXTERNAL: retrieves only the extreme outer contours
                                                                                           # Third argument: Contour Approximation mode
                                                                                           # CHAIN_APPROX_SIMPLE: compresses horizontal, vertical, and diagonal segments
                                                                                           # and leaves only their end points
                                                                                           # Example: An upright rectangular contour is encoded with 4 points
    contours = imutils.grab_contours(contours)
    c = max(contours, key = cv2.contourArea) # look for the contour that has the maximum area
    
    # create a numpy array for the mask which will contain the
    # rectangular bounding box of the stitched image region
    mask = np.zeros(thresh.shape, dtype='uint8') # allocate memory for the new rectangular mask
    (x, y, w, h) = cv2.boundingRect(c) # calculates the bounding box of the largest contour
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1) # draw a solid white rectangle on the mask
    
    # create two copies of the mask: one to serve as the actual
    # minimum rectangle region and another to serve as a counter
    # for how many pixels need to be removed to form the minimum
    # rectangular region
    minRect = mask.copy() # will be slowly reduced in size until it can fit inside
                          # the inner part of the panorama
    sub = mask.copy() # will be used to determine if we need to keep reducing
                      # the size of minRect
    
    # keep looping until there are no non-zero pixels (no more foreground pixels) left in the
    # subtracted image
    # the size of minRect is progressively reduced until there are no more foreground pixels
    # left in sub
    while cv2.countNonZero(sub) > 0:
        # erode the minimum rectangular mask and then subtract
        # the thresholded image from the minimum rectangular mask
        # so we can count if there are any non-zero pixels left
        minRect = cv2.erode(minRect, None)
        sub = cv2.subtract(minRect, thresh)
    
    # loop is broken, you now have found the smallest rectangular mask that can fit
    # into the largest rectangular region of the panorama
    
    # find contours in the minimum rectangular mask and then
    # extract the bounding box (x, y) coordinates
    contours = cv2.findContours(minRect.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    c = max(contours, key = cv2.contourArea)
    (x, y, w, h) = cv2.boundingRect(c)
    
    # Use the bounding box coordinates to extract the final stitched image
    stitched = stitched[y: y + h, x: x + w]
    
    # return the final stitched image
    return stitched

def HSV_Equalize(image):
    H, S, V = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV)) # grab the Hue, Saturation, and Value channels
                                                                # of the image
    eq_V = cv2.equalizeHist(V) # Perform histogram equalization on the Value channel only
                               # basically, only the brightness is equalized
    eq_image = cv2.cvtColor(cv2.merge([H, S, eq_V]), cv2.COLOR_HSV2BGR) # put all channels back together
                                                                        # and convert back to BGR space
    # return the HSV Equalized Image
    return eq_image

# Stitching part  
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


# End of stitching part

# Show the cropped and sitched image
cropped = crop(finalImage2)
cv2.imshow("After Cropping", cropped)

# Should the image be HSV Equalized?
user_input = input("Would you like to equalize the image? ")
ans = str(user_input)
if ans == 'y':
    equalized = HSV_Equalize(cropped)
    cv2.imshow("After Equalization", equalized)

cv2.waitKey()
cv2.destroyAllWindows()