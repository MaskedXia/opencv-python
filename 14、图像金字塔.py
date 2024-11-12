import cv2 as cv
import numpy as np

#图像金字塔（高斯金字塔、拉普拉斯金字塔）
#    reduce = 高斯模糊 + 降采样       （尺寸变小，分辨率降低）
#    expand = 扩大 + 卷积             （尺寸变大，但分辨率不会增加）
def pyramid_demo(image):
    level = 3
    temp = image.copy()
    pyramid_images = []
    for i in range(level):
        dst = cv.pyrDown(temp)
        #print(dst.shape)
        pyramid_images.append(dst)
        cv.imshow("pyramid_down"+str(i), dst)
        temp = dst.copy()
    return  pyramid_images

def laplace_demo(image):#看起来像边界图,必须宽高相等
    pyramid_images = pyramid_demo(image)
    level = len(pyramid_images)
    for i in range(level-1, -1, -1):
        if (i-1) < 0:
            expand = cv.pyrUp(pyramid_images[i], dstsize=image.shape[:2])
            lpls = cv.subtract(image, expand)
            cv.imshow("laplace_down" + str(i), lpls)
        else:
            expand = cv.pyrUp(pyramid_images[i], dstsize=pyramid_images[i-1].shape[:2])
            lpls = cv.subtract(pyramid_images[i-1], expand)
            cv.imshow("laplace_down"+str(i), lpls)




src = cv.imread("image1/012.png")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)

#pyramid_demo(src)
laplace_demo(src)


cv.waitKey(0)
cv.destroyAllWindows()