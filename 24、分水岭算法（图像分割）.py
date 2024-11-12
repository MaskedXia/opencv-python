import cv2 as cv
import numpy as np
#分水岭算法  （基于距离变换）
'''
输入图像 - 灰度(去噪) - 二值 - 距离变换 - 寻找种子 - 生成Marker - 分水岭变换 - 输出图像
'''
def watershed_demo(image):
    print(image.shape)
    blurred = cv.pyrMeanShiftFiltering(image, 10, 100)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow("binary", binary)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    nb = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv.dilate(nb, kernel, iterations=3)
    cv.imshow("mcr", sure_bg)

    dist = cv.distanceTransform(nb, cv.DIST_L2, 3)
    dist_output = cv.normalize(dist, 0, 1.0, cv.NORM_MINMAX)
    cv.imshow("distance", dist_output*50)

    ret, surface = cv.threshold(dist, dist.max()*0.6, 255, cv.THRESH_BINARY)
    cv.imshow("surface", surface)

    surface_fg = np.uint8(surface)
    unknown = cv.subtract(sure_bg, surface_fg)
    ret, markers = cv.connectedComponents(surface_fg)
    print(ret)

    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv.watershed(image, markers)
    image[markers == -1] = [0, 0, 255]
    cv.imshow("result", image)

src = cv.imread("image/2.png")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)

watershed_demo(src)

cv.waitKey(0)
cv.destroyAllWindows()