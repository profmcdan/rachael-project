import cv2
import numpy as np

image = cv2.imread("images/door4.jpeg")
copy = image.copy()
cv2.imshow("Original", image)
# cv2.waitKey(0)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray', gray)
# cv2.waitKey(0)

edged = cv2.Canny(gray, 10, 250)
cv2.imshow('Edged', edged)
# cv2.waitKey(0)

kernel = np.ones((5, 5), np.uint8)

dilation = cv2.dilate(edged, kernel, iterations=1)
cv2.imshow('Dilation', dilation)
# cv2.waitKey(0)

closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
cv2.imshow('Closing', closing)
# cv2.waitKey(0)

# if using OpenCV 4, remove image variable from below
cnts, hiers = cv2.findContours(
    closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cont = cv2.drawContours(copy, cnts, -1, (0, 0, 0), 1, cv2.LINE_AA)
cv2.imshow('Contours', cont)
# cv2.waitKey(0)

mask = np.zeros(cont.shape[:2], dtype="uint8") * 255

# Draw the contours on the mask
cv2.drawContours(mask, cnts, -1, (255, 255, 255), -1)

# remove the contours from the image and show the resulting images
img = cv2.bitwise_and(cont, cont, mask=mask)
cv2.imshow("Mask", img)
# cv2.waitKey(0)

for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    if w > 50 and h > 130:
        new_img = img[y:y + h, x:x + w]
        cv2.imwrite('Cropped.png', new_img)
        cv2.imshow("Cropped", new_img)


while True:
    k = cv2.waitKey(0)
    if k == 27:
        break
