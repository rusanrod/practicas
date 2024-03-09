#
# COMPUTER VISION FOR ROBOTICS - UNAM, 2024-2
# NOISE TYPES
#

import numpy as np
import cv2

def trackbar_sp_callback(val):
    global img1
    sp = float(val)/100.0
    mask1 = np.random.choice([0,255], size=(512,512), p=[1.0-sp/2, sp/2]).astype(np.uint8)
    mask2 = np.random.choice([255,0], size=(512,512), p=[1.0-sp/2, sp/2]).astype(np.uint8)
    img1 = cv2.bitwise_or(img, mask1)
    img1 = cv2.bitwise_and(img1, mask2)

def trackbar_impulse_callback(val):
    global img2
    impulse = float(val)/100.0
    mask = np.random.choice([0,255], size=(512,512), p=[1.0-impulse, impulse]).astype(np.uint8)
    img2 = cv2.bitwise_or(img, mask)

def trackbar_gaussian_callback(val):
    global img3
    noise = np.random.normal(scale=val, size=(512,512)).astype(np.int8)
    img3 = (img + noise).astype(np.uint8)
    
def main():
    global img1, img2, img3, img
    img  = cv2.cvtColor(cv2.imread('baboon.jpg'), cv2.COLOR_BGR2GRAY)
    img1 = np.copy(img)
    img2 = np.copy(img)
    img3 = np.copy(img)
    cv2.namedWindow('Salt&Pepper')
    cv2.namedWindow('Impulse')
    cv2.namedWindow('Gaussian')
    cv2.createTrackbar('p','Salt&Pepper', 0, 30, trackbar_sp_callback)
    cv2.createTrackbar('p','Impulse',     0, 30, trackbar_impulse_callback)
    cv2.createTrackbar('s','Gaussian',    0, 30, trackbar_gaussian_callback)
        
    while True:
        cv2.imshow('Salt&Pepper', img1)
        cv2.imshow('Impulse'    , img2)
        cv2.imshow('Gaussian'   , img3)
        if cv2.waitKey(100) & 0xFF == 27:
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
