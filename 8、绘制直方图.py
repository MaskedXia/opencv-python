import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
#图像直方图（histogram）：图像的灰度分布
def plot_demo(image):#简单方法
    plt.hist(image.ravel(), 256, [0, 256])
    plt.show()

def image_hist_demo(image):#绘制多通道BGR的直方图
    color = ("blue", "green", "red")
    for i, color in enumerate(color):
        hist = cv.calcHist([image], [i], None, [256], [0, 256])
        #源图像组, 图像信道BGR, 可选的掩码，BIN 的数目（分组数目）, 像素值范围
        #返回BIN * 1的数组，每个值代表灰度值对应的像素点个数
        plt.plot(hist, color = color)
        plt.xlim([0, 256])#x轴上下限
    plt.show()

src = cv.imread("image/1.png")
#cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)

plot_demo(src)
image_hist_demo(src)

cv.waitKey(0)
cv.destroyAllWindows()