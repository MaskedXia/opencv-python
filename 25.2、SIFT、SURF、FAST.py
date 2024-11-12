import cv2 as cv
import numpy as np

#Harris对缩放敏感，可能检测不到角点
#Scale-Invariant Feature Transform 尺度不变特征变换  SIFT 2004
#高斯拉普拉斯算子（LoG）改进为高斯差分算子（DoG）
def sift(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    sift = cv.xfeatures2d.SIFT_create()#整合到xfeatures2d
    kp = sift.detect(gray, None)#在图像中找到关键点
    dst = cv.drawKeypoints(gray, kp, None, cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #在关键点绘制小圆圈，DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS绘制出方向
    cv.imshow("SIFT", dst)

#SURF(Speeded-Up Robust Features) 加速稳健特征  2006
# LoG改进为盒子滤波器（box_filter）
#使用小波变换
def surf(image):#白斑检测器
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    surf = cv.xfeatures2d.SURF_create(50000)
    #设置海森阈值到400
    kp, des = surf.detectAndCompute(gray, None)
    #print(len(kp)) 关键点的个数
    dst = cv.drawKeypoints(image, kp, None, (255,0,0), 4)
    cv.imshow("SURF", dst)

#速度更快  2006
#对像素点周围的像素点测试，连续几个点都高于或者都低于，就认为是角点
#构建向量，ID3决策树分类器，非极大值抑制
def fast(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    fast = cv.FastFeatureDetector_create()
    fast.setNonmaxSuppression(False)#不使用非极大值抑制
    kp = fast.detect(gray, None)
    dst = cv.drawKeypoints(image, kp, None, color=(255,0,0))
    cv.imshow("FAST", dst)


src = cv.imread("image1/011.png")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)

sift(src)
surf(src)
fast(src)

cv.waitKey(0)
cv.destroyAllWindows()