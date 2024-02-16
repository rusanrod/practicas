#
# COMPUTER VISION
# FI UNAM 2024-2
# BASIC SIGNAL OPERATIONS
#

import numpy as np
import cv2

def trackbar_shift_x_callback(val):
    global shift_x
    shift_x = val

def trackbar_shift_y_callback(val):
    global shift_y
    shift_y = val

def trackbar_scale_callback(val):
    global scale
    scale = float(val)/100.0

def trackbar_merge_callback(val):
    global merge
    merge = float(val)/100.0
    
def main():
    global shift_x, shift_y, scale, merge 
    shift_x, shift_y = 0,0
    scale = 1.0
    merge = 0.5

    img1 = cv2.cvtColor(cv2.imread('baboon.jpg'), cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(cv2.imread('lena.png')  , cv2.COLOR_BGR2GRAY)

    cv2.namedWindow('Shift')
    cv2.createTrackbar('x','Shift', 0, 100, trackbar_shift_x_callback)
    cv2.createTrackbar('y','Shift', 0, 100, trackbar_shift_y_callback)
    cv2.setTrackbarMin('x', 'Shift', -100)
    cv2.setTrackbarMin('y', 'Shift', -100)

    cv2.namedWindow('Scale')
    cv2.createTrackbar('%', 'Scale', 100, 200, trackbar_scale_callback)

    cv2.namedWindow('Merge')
    cv2.createTrackbar('Bal', 'Merge', 50, 100, trackbar_merge_callback)
        
    while True:
        img_shifted = np.zeros((712, 712), np.uint8)
        img_shifted[100+shift_y:612+shift_y, 100+shift_x:612+shift_x] = img1
        cv2.imshow("Shift", img_shifted[100:612, 100:612])
        cv2.imshow("Scale", img1*scale/255.0)
        cv2.imshow("Invert XY", np.flip(img1))
        cv2.imshow("Merge", (img1*merge/255.0 + img2*(1.0 - merge)/255.0))
        if cv2.waitKey(100) & 0xFF == 27:
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

