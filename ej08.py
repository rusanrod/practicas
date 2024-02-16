import cv2
import numpy
import math

def get_gaussian_kernel(k,sigma):
    k = k//2
    H = numpy.zeros((2*k+1, 2*k+1))
    r,c = H.shape
    for i in range(r):
        for j in range(c):
            H[i,j] = 1.0/(2*math.pi*sigma*sigma)*math.exp(-((i-k)**2 + (j-k)**2)/(2*sigma*sigma))
    H /= numpy.sum(H)
    return H

def main():
    img  = cv2.imread("baboon.jpg")
    img  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
    img /= 255.0
    Sx = numpy.asarray([[1,0,-1], [2,0,-2], [1,0,-1]])
    Sy = Sx.T

    Gx = cv2.filter2D(img, cv2.CV_64F, Sx)
    Gy = cv2.filter2D(img, cv2.CV_64F, Sy)
    cv2.imshow("Original", img)
    cv2.imshow("Gx", Gx)
    cv2.imshow("Gy", Gy)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()
