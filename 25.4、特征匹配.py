import cv2 as cv
import numpy as np
#蛮力（Brute-Force）匹配和 FLANN 匹配

#Brute-Force 匹配
#第一个是 normType默认值为 cv2.Norm_L2,这很适合 SIFT 和 SURF
#使用 cv2.NORM_HAMMING,二进制描述符的 ORB，BRIEF，BRISK算法等
#第二个参数是布尔变量 crossCheck，默认值为 False。如果设置为True，
# 匹配条件就会更加严格
def Brute_Force():
    img1 = cv.imread("image/3.png")
    img2 = cv.imread("image/5.png")
    orb = cv.ORB_create()
    kp1, des1 =  orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    img3 = cv.drawMatches(img1, kp1, img2, kp2, matches, None)
    cv.imshow("Brute_Force",img3)

def Brute_Force_Another():
    img1 = cv.imread("image/1.png")
    img2 = cv.imread("image/7.png")
    sift = cv.sift = cv.xfeatures2d.SIFT_create()
    kp1, des1 =  sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
    img3 = cv.drawMatches(img1, kp1, img2, kp2, good[:10], None, flags=2)
    cv.imshow("Brute_Force_Another",img3)

#FLANN 是快速最近邻搜索包（Fast_Library_for_Approximate_Nearest_Neighbors）

src = cv.imread("image/1.png")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)

Brute_Force()
Brute_Force_Another()


cv.waitKey(0)
cv.destroyAllWindows()