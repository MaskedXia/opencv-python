import cv2 as cv
import numpy as np

# flags = [i for i in dir(cv) if i.startswith('COLOR_')]
# print(flags)
'''
色彩空间:BGR、HSV、HIS、YCrCb、YUV
HSV 格式中，H（色调）的取值范围是 [0，179]，
S（饱和度）的取值范围 [0，255]，V（亮度）的取值范围 [0，255]
因为HSV颜色空间比BGR空间更容易表示特定颜色，可以通过设置HSV的阈值圈定颜色范围
'''
def bgr2hsv():
    red = np.uint8([[[0,0,255]]])#uint8是无符号八位整型，表示范围是[0, 255]的整数
    hsv_red = cv.cvtColor(red, cv.COLOR_BGR2HSV)
    print(hsv_red)


def extract_object():
    capture = cv.VideoCapture("video/01.mp4")
    while(True):
        ret, frame = capture.read()
        if ret == False:
            break
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        lower_hsv = np.array([37, 43, 46])
        upper_hsv = np.array([77, 255, 255])
        mask = cv.inRange(hsv, lowerb=lower_hsv, upperb=upper_hsv)
        #有常见几种颜色范围图表
        dst = cv.bitwise_and(frame, frame, mask=mask)
        cv.imshow("video", frame)
        cv.imshow("mask", dst)#追踪绿色的对象
        c = cv.waitKey(40)
        if c == 27:
            break

def color_space(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imshow("gray", gray)
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    cv.imshow("hsv", hsv)#常用，h:0-180 s:0-255 v:0-255
    yuv = cv.cvtColor(image, cv.COLOR_BGR2YUV)
    cv.imshow("yuv", yuv)#常用
    Ycrcb = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
    cv.imshow("Ycrcb", Ycrcb)

src = cv.imread("image/1.png")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)

cv.imshow("input image", src)

# b, g, r =cv.split(src)
# cv.imshow("blue", b)
# cv.imshow("green", g)
# cv.imshow("red", r)

# src[:, :, 2] = 0
# src = cv.merge([b, g, r])
# cv.imshow("changed image", src)

#color_space(src)
#extract_object()
bgr2hsv()

cv.waitKey(0)
cv.destroyAllWindows()
