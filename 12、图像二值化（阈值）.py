import cv2 as cv
import numpy as np
#图像二值化（全局阈值，局部阈值） 只有0和1  #OTSU Triangle 自动和手动
def threshold_demo(image):#全局阈值
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    print("threshold value", ret)
    cv.imshow("binary", binary)

def local_threshold(image):#局部二值化(更好)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    binary = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv.THRESH_BINARY, 25, 10)
    #输入图像，向上最大值，自适应方法（平均或者高斯），阈值化类型，块大小25*25（必须是奇数），减去常量C
    cv.imshow("local_binary", binary)

def custom_threshold(image):#自定义
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    #m = np.reshape(gray, [1, w*h])
    mean = gray.sum() / (w+h)
    print("mean:", mean)
    ret, binary = cv.threshold(gray, mean, 255, cv.THRESH_BINARY)
    print("threshold value", ret)
    cv.imshow("binary", binary)


src = cv.imread("insulator/001.jpg")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)

threshold_demo(src)
local_threshold(src)
#custom_threshold(src)

cv.waitKey(0)
cv.destroyAllWindows()