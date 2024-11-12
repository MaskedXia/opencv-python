import cv2 as cv
import numpy as np

# BRIEF 是一种对特征点描述符计算和匹配的快速方法
def brief(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    star = cv.xfeatures2d.StarDetector_create()#初始化STAR探测器
    brief = cv.xfeatures2d.BriefDescriptorExtractor_create()#初始化BRIEF探测器
    kp = star.detect(gray, None)
    kp, des = brief.compute(gray, kp)
    dst = cv.drawKeypoints(image, kp, None, color=(255, 0, 0))
    cv.imshow("brief", dst)

#2011
#ORB 基本是 FAST 关键点检测和 BRIEF 关键点描述器的结合体，并通
#过很多修改增强了性能
def orb(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    orb = cv.ORB_create()
    kp = orb.detect(image, None)
    kp, des = orb.compute(gray, kp)
    dst = cv.drawKeypoints(image, kp, None, color=(255, 0, 0))
    cv.imshow("orb", dst)

src = cv.imread("image1/011.png")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)

brief(src)
orb(src)

cv.waitKey(0)
cv.destroyAllWindows()