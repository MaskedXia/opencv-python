import cv2 as cv
import numpy as np

#轮廓发现 (利用梯度来避免阈值烦恼)
def contours_demo(image):
    dst = cv.GaussianBlur(image, (3,3), 0)
    gray = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow("binary image", binary)#也可以用canny

    cloneImage, contours, heriachy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    #第一个是输入图像，第二个是轮廓检索模式，第三个是轮廓近似方法
    #轮廓近似方法：CHAIN_APPROX_NONE：存储所有边界点 CHAIN_APPROX_SIMPLE：去除冗余点
    #cv.imshow("cloneImage", cloneImage) 和原图像一样
    #返回值有三个，第一个是图像，第二个是轮廓，第三个是（轮廓的）层析结构
    #第二个轮廓是个列表，存储所有轮廓

    for i, contour in enumerate(contours):
        cv.drawContours(image, contours, i, (0,0,255), 2)
        #绘制轮廓第一个参数是原始图像，第二个参数是轮廓，一个 Python 列表。
        #第三个参数是轮廓的索引，设为-1就会绘制所有轮廓
        print(i)
    cv.imshow("detect contours", image)

src = cv.imread("image/1.png")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)

contours_demo(src)

cv.waitKey(0)
cv.destroyAllWindows()
