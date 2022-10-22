import cv2
import numpy as np

# Camera settings
cap = cv2.VideoCapture(0)

# Importing the image
imgTarget = cv2.imread('TargetImage.jpg')

# Importing the video
myVid = cv2.VideoCapture('video.mp4')

success, imgVideo = myVid.read()

cv2.imshow('ImgTarget', imgTarget)
cv2.imshow('myVideo', imgVideo)
cv2.waitKey(0)
