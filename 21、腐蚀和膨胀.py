import cv2 as cv
import numpy as np
#腐蚀（最小值滤波）和膨胀（最大值滤波）              (灰度和二值图像）
def erode_demo(image):#腐蚀
    print(image.shape)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow("binary", binary)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (10, 10))#矩形
    #等效kernel = np.ones((10,10), np.float32)
    #内核形状：MORPH_RECT：矩形  MORPH_CROSS：交叉形  MORPH_ELLIPSE：椭圆形
    #尺寸， 锚点的位置，有默认值Point（-1,-1），表示锚点位于中心点
    dst = cv.erode(binary, kernel)#腐蚀操作
    cv.imshow("erode_demo", dst)

def dilate_demo(image):#膨胀
    print(image.shape)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow("binary", binary)
    kernel1 = cv.getStructuringElement(cv.MORPH_RECT, (10, 10))#矩形
    kernel2 = np.ones((10, 10), np.uint8)
    dst1 = cv.dilate(binary, kernel1)#膨胀操作
    dst2 = cv.dilate(binary, kernel2)  # 膨胀操作
    cv.imshow("dilate_demo1", dst1)
    cv.imshow("dilate_demo2", dst2)

src = cv.imread("image/1.png")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)

#erode_demo(src)
dilate_demo(src)

cv.waitKey(0)
cv.destroyAllWindows()
