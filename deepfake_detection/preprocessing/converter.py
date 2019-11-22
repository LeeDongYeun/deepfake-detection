import cv2
import numpy as np
import random
from matplotlib import pyplot as plt


def GenerateFakeImages(img):
    face_cascade = cv2.CascadeClassifier(
        os.getcwd()+'/haarcascade_frontface.xml')  # opencv에서 제공하는 얼굴인식 data

    # img = cv2.imread('ironman.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # 받은 img에서 얼굴 부분 인식

    for (x, y, w, h) in faces:
        cropimg = img[y:y+h, x:x+w]
        # cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        # roi_gray = gray[y:y+h, x:x+w]
        # roi_color = img[y:y+h, x:x+w]

    alsize = random.randrange(1, 10)
    fx = 0.6 + 0.1 * alsize
    fy = 0.6 + 0.1 * alsize

    alimg = cv2.resize(cropimg, dsize=(0, 0), fx=fx, fy=fy,
                       interpolation=cv2.INTER_LINEAR)
    temp = cv2.GaussianBlur(alimg, (5, 5), 0)  # 얼굴부분 추출해서 Gaussian Blur
    blur = cv2.resize(temp, dsize=(0, 0), fx=1/fx, fy=1 /
                      fy, interpolation=cv2.INTER_LINEAR)

    for i in range(y, y+h):
        for j in range(x, x+w):
            for k in range(3):
                img[i, j, k] = blur[i-y, j-x, k]

    # img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    # mask_inv = cv2.bitwise_not(mask) #마스크 만들기
    # final = cv2.bitwise_and(cropimg, blur, mask=mask) #두 이미지 다시 합치기(아직 미완성)

    cv2.imshow('result3', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img = cv2.imread('ironman.jpg')
GenerateFakeImages(img)
