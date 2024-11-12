import cv2 as cv#EPF, 边缘保留滤波    （高斯双边、均值迁移）   ps必备
import numpy as np

def bi_demo(image):#高斯双边
    dst =cv.bilateralFilter(image, 0, 100, 15) #15-100
    cv.imshow("bi", dst)

def shift_demo(image):#均值迁移
    dst =cv.pyrMeanShiftFiltering(image, 10, 50)
    cv.imshow("shift", dst)

src = cv.imread("image/2.png")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)

bi_demo(src)
shift_demo(src)

cv.waitKey(0)
cv.destroyAllWindows()