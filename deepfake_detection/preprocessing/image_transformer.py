import cv2

def rotate(img, angle):
    height, width, channel = img.shape
    matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
    dst = cv2.warpAffine(img, matrix, (width, height))

    return dst

def symmetry(img, dir):
    dst = cv2.flip(img, dir)    #dir<0:up-down, dir>1 : left-right

    return dst

def reverse(img):
    dst = cv2.bitwise_not(img)

    return dst

def blur(img, size):
    dst = cv2.blur(img, (size, size), anchor=(-1, -1), borderType=cv2.BORDER_DEFAULT)

    return dst