import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
#直方图反向投影
def back_projection_demo():
    sample = cv.imread("image1/010.png")
    target = cv.imread("image/7.png")
    roi_hsv = cv.cvtColor(sample, cv.COLOR_BGR2HSV)
    target_hsv = cv.cvtColor(target, cv.COLOR_BGR2HSV)
    #显示图像
    cv.imshow("sample", sample)
    cv.imshow("target", target)

    roi_hist = cv.calcHist([roi_hsv], [0, 1], None, [36, 48], [0, 180, 0, 256])
    cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)
    # 归一化：原始图像，结果图像，映射到结果图像中的最小值，最大值，归一化类型
    # cv2.NORM_MINMAX 对数组的所有值进行转化，使它们线性映射到最小值和最大值之间
    # 归一化之后的直方图便于显示，归一化之后就成了 0 到 255 之间的数了。
    dst = cv.calcBackProject([target_hsv], [0,1], roi_hist, [0, 180, 0, 256], 1)
    cv.imshow("backprojection", dst)


def hist2d_demo(image):
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    hist = cv.calcHist([image], [0, 1], None, [180, 256], [0, 180, 0, 256])
    # 2D直方图，转换到HSV，[0,1]处理H、S通道，[0, 180, 0, 256] H是0-180，S是0-256
    #cv.imshow("hist2d", hist)
    plt.imshow(hist, interpolation="nearest")
    plt.title(" 2D Hist")
    plt.show()



src = cv.imread("image/7.png")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)

#hist2d_demo(src)
back_projection_demo()

cv.waitKey(0)
cv.destroyAllWindows()