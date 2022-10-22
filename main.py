import cv2
import numpy as np

# Camera settings
cap = cv2.VideoCapture(0)

# Importing the image
imgTarget = cv2.imread('TargetImage.jpg')

# Importing the video
myVid = cv2.VideoCapture('video.mp4')

success, imgVideo = myVid.read()

# Resize video with image size
hT, wT, cT = imgTarget.shape
imgVideo = cv2.resize(imgVideo, (wT, hT))

# Key points
orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(imgTarget, None)
# imgTarget = cv2.drawKeypoints(imgTarget, kp1, None)

while True:
    successful, imgWebcam = cap.read()
    kp2, des2 = orb.detectAndCompute(imgWebcam, None)
    # imgWebcam = cv2.drawKeypoints(imgWebcam, kp2, None)

    # Brute-Force matcher to compare key points
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    print(len(good))
    imgFeatures = cv2.drawMatches(imgTarget, kp1, imgWebcam, kp2, good, None, flags=2)

    cv2.imshow('ImgFeatures', imgFeatures)
    cv2.imshow('ImgTarget', imgTarget)
    cv2.imshow('myVideo', imgVideo)
    cv2.imshow('Webcam', imgWebcam)
    cv2.waitKey(0)
