import numpy as np
import argparse
import imutils
import cv2


def is_contour_bad(c):
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    # the contour is 'bad' if it is not a rectangle
    return not len(approx) == 4


def remove_background(image):
    r = 150.0 / image.shape[1]
    dim = (150, int(image.shape[0] * r))
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    lower_white = np.array([80, 1, 1], np.uint8)  # lower hsv value
    upper_white = np.array([130, 255, 255], np.uint8)  # upper hsv value
    # rgb to hsv color space
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # filter the background pixels
    frame_threshed = cv2.inRange(hsv_img, lower_white, upper_white)

    kernel = np.ones((5, 5), np.uint8)
    # dilate the resultant image to remove noises in the background
    # Number of iterations and kernal size will depend on the backgound noises size
    dilation = cv2.dilate(frame_threshed, kernel, iterations=5)
    # convert background pixels to black color
    image[dilation == 255] = (0, 0, 0)
    return image


# load the shapes image, convert it to grayscale, and edge edges in
# the image
image = cv2.imread("door4.jpeg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edged = cv2.Canny(gray, 50, 100)
cv2.imshow("Original", image)

filteredImage = remove_background(image)
cv2.imshow("BG", filteredImage)


# l_h = cv2.getTrackbarPos("L-H", "Trackbars")
# l_s = cv2.getTrackbarPos("L-S", "Trackbars")
# l_v = cv2.getTrackbarPos("L-V", "Trackbars")
# u_h = cv2.getTrackbarPos("U-H", "Trackbars")
# u_s = cv2.getTrackbarPos("U-S", "Trackbars")
# u_v = cv2.getTrackbarPos("U-V", "Trackbars")

# lower_red = np.array([l_h, l_s, l_v])
# upper_red = np.array([u_h, u_s, u_v])

# find contours in the image and initialize the mask that will be
# used to remove the bad contours
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
mask = np.ones(image.shape[:2], dtype="uint8") * 255

# loop over the contours
for c in cnts:
    # if the contour is bad, draw it on the mask
    if is_contour_bad(c):
        cv2.drawContours(mask, [c], -1, 0, -1)

# remove the contours from the image and show the resulting images
image = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("Mask", mask)
cv2.imshow("After", image)

while True:
    k = cv2.waitKey(0)
    if k == 27:
        break
