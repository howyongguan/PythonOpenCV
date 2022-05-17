import cv2 as cv
from cv2 import GaussianBlur
import numpy as np
import argparse

# Load the image and convert to grayscale
img = cv.imread("images/solar_thermal.JPG")
orig = img.copy()
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Apply a Gaussain blur to find better brightness region
blurred = cv.GaussianBlur(gray, (21, 21), 0)
(minVal, maxVal, minLoc, maxLoc) = cv.minMaxLoc(blurred)
cv.circle(img, maxLoc, 40, (255, 0, 0), 2)

cv.putText(img=img, text=f'PV Panels A:{maxVal}', org=(30, 40), fontFace=cv.FONT_HERSHEY_PLAIN,
           fontScale=1.5, color=(0, 255, 0), thickness=2, lineType=cv.LINE_AA)
cv.imshow("grayscale", blurred)
cv.imshow("Detection", img)

#cv.imshow("img", gray)
cv.waitKey(0)
cv.destroyAllWindows()
