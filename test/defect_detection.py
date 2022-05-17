# Import the necessary packages
import cv2 as cv
from cv2 import GaussianBlur
import numpy as np
from imutils import contours
from skimage import measure
import imutils

img = cv.imread("images/solar_thermal.JPG")
orig = img.copy()
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Apply a Gaussain blur to find better brightness region
blurred = cv.GaussianBlur(gray, (41, 41), 0)
thresh = cv.threshold(blurred, 200, 255, cv.THRESH_BINARY)[1]

# perform a series of erosions and dilations to remove  any small blobs of noise from the thresholded image
thresh = cv.erode(thresh, None, iterations=1)
thresh = cv.dilate(thresh, None, iterations=2)

# perform a connected component analysis on the thresholded image, then initialize a mask to store only the "large"components
labels = measure.label(thresh, background=0)
mask = np.zeros(thresh.shape, dtype="uint8")
# loop over the unique components
for label in np.unique(labels):
    # if this is the background label, ignore it
    if label == 0:
        continue
    # otherwise, construct the label mask and count the
    # number of pixels
    labelMask = np.zeros(thresh.shape, dtype="uint8")
    labelMask[labels == label] = 255
    numPixels = cv.countNonZero(labelMask)
    # if the number of pixels in the component is sufficiently
    # large, then add it to our mask of "large blobs"
    if numPixels > 100:
        mask = cv.add(mask, labelMask)


# find the contours in the mask, then sort them from left to right
cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = contours.sort_contours(cnts)[0]

# loop over the contours
for (i, c) in enumerate(cnts):
    # draw the bright spot on the image
    (x, y, w, h) = cv.boundingRect(c)
    ((cX, cY), radius) = cv.minEnclosingCircle(c)
    cv.circle(img, (int(cX), int(cY)), int(radius),
              (0, 0, 255), 3)
    cv.putText(img, "Panel{}".format(i + 1), (x, y - 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

cv.putText(img=img, text=f'Total Defect Panels :{i+1}', org=(30, 40), fontFace=cv.FONT_HERSHEY_PLAIN,
           fontScale=1.5, color=(0, 255, 0), thickness=2, lineType=cv.LINE_AA)

#cv.imshow("test", thresh)
cv.imshow("Detection", img)
cv.waitKey(0)
cv.destroyAllWindows()
