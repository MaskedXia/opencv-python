import cv2 as cv
import numpy as np

def harris_detect(image): #1988
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = np.float32(gray)#输入图像必须是float32
    dst = cv.cornerHarris(gray, 2, 3, 0.04)
    #img - 数据类型为 float32 的输入图像,  blockSize - 角点检测中要考虑的领域大小
    #ksize - Sobel 求导中使用的窗口大小,k - Harris 角点检测方程中的自由参数，取值参数为 [0,04，0.06]
    dst = cv.dilate(dst, None)
    image[dst > 0.01*dst.max()] = [0,0,255]
    ##这里的打分值以大于0.01×dst中最大值为边界
    cv.imshow("harris_detect",image)

def Good_Features_to_Track(image):#改进Harris  1994
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    corners = cv.goodFeaturesToTrack(gray, 25, 0.01, 10)
    #灰度图像， 角点数目， 角点的质量水平（0-1之间）， 角点间最短欧氏距离
    corners = np.int0(corners)
    for i in corners:
        x,y = i.ravel()
        cv.circle(image, (x,y), 3, (0,0,255), -1)
    cv.imshow("Good_Features_to_Track",image)

src = cv.imread("image/4.png")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)

harris_detect(src)
Good_Features_to_Track(src)

cv.waitKey(0)
cv.destroyAllWindows()