import cv2
import numpy as np

img1 = cv2.imread("image/kinect_sport_2.jpg", 0)
height, width = img1.shape
height = int(0.3*height)
width = int(0.3*width)
img1 = cv2.resize(img1, (height, width), interpolation = cv2.INTER_AREA)
img2 = cv2.imread("image_train/kinect_sport_2.jpg", 0)

orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# img_kp1 = cv2.drawKeypoints(img1, kp1, None)
# img_kp2 = cv2.drawKeypoints(img2, kp2, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2,k=2)

good = []
for m, n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

# cv2.imshow("Image", img_kp1)
# cv2.imshow("Image Test", img_kp2)
cv2.imshow("", img3)
cv2.waitKey(0)