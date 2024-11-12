import cv2 as cv
import numpy as np
#圆检测（霍夫圆变换） 对噪声敏感，做中值滤波
def detect_circle_demo(image):
    dst = cv.pyrMeanShiftFiltering(image, 10, 100)
    cimage = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
    circles = cv.HoughCircles(cimage, cv.HOUGH_GRADIENT, 1, 20, param1=50, param2=25,
                              minRadius=0, maxRadius=0)
    #灰度图像，使用霍夫梯度的方法，累加器图像分辨率（1分辨率相同），
    #min_dist区分的两个不同圆之间的最小距离
    #param1用于Canny的边缘阀值上限，下限被置为上限的一半，param2累加器阈值
    #minRadius最小圆半径，maxRadius最大圆半径
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv.circle(image, (i[0], i[1]), i[2], (0, 0, 255), 2)
        cv.circle(image, (i[0], i[1]), 2, (0, 255, 255), 2)
    cv.imshow("circles", image)


src = cv.imread("image1/004.png")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)

detect_circle_demo(src)

cv.waitKey(0)
cv.destroyAllWindows()
