# import the necessary packages
import numpy as np
import argparse
import imutils
import cv2


# load the image
image = cv2.imread("shapes.png")

image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Convert BGR to HSV
# hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#  define range of GREEN color in HSV
lower_green = np.array([46, 0, 0])
upper_green = np.array([70, 255, 255])

shapeMask = cv2.inRange(image_hsv, lower_green, upper_green)

# find the contours in the mask
cnts = cv2.findContours(shapeMask.copy(), cv2.RETR_TREE,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
print("I found {} green shapes".format(len(cnts)))
cv2.imshow("Mask", shapeMask)

image2 = image.copy()

cv2.drawContours(image2, cnts, -1, (0, 255, 0), 3)
cv2.imshow("Image2", image2)

# print(cnts)
# loop over the contours
for c in cnts:
    # draw the contour and show it
    cv2.drawContours(image, [c], -1, (0, 0, 0), 3)
    cv2.imshow("Image", image)

while True:
    k = cv2.waitKey(0)
    if k == 27:
        break
