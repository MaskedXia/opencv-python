import cv2 as cv
import numpy as np
'''
开操作 = 腐蚀 + 膨胀（去除小的干扰快）    去除小白点
闭操作 = 膨胀 + 腐蚀 (填充闭合区间)       去除小黑点
水平线和垂直线的提取
'''

def open_demo(image):
    print(image.shape)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow("binary", binary)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))#(15, 1)来提取水平线和垂直线
    dst = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)#形态学操作
    cv.imshow("open-result", dst)

def close_demo(image):
    print(image.shape)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow("binary", binary)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    dst = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)
    cv.imshow("close-result", dst)


src = cv.imread("image/1.png")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)

open_demo(src)
close_demo(src)

cv.waitKey(0)
cv.destroyAllWindows()