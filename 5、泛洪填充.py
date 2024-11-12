import cv2 as cv
import numpy as np
#1.ROI操作，region of interst 2.泛洪填充
def fill_color_demo(image):
    copyimg = image.copy()
    h, w = image.shape[:2]
    mask = np.zeros([h+2, w+2], np.uint8)#掩膜
    cv.floodFill(copyimg, mask, (30,30), (0, 255, 255), (100, 100, 100), (50, 50, 50),
                 cv.FLOODFILL_FIXED_RANGE)#(0, 255, 255)是黄色
    #(30,30)里BGR值减去(100, 100, 100)最低，(30,30)里BGR值加上(50, 50, 50)最高
    cv.imshow("fill_color", copyimg)

def fill_binary_demo():
    image = np.zeros([400, 400, 3], np.uint8)
    image[100:300, 100:300, :] =255
    mask = np.ones([402, 402, 1], np.uint8)#必须为1
    mask[101:301, 101:301] = 0
    cv.floodFill(image, mask, (200,200), (0,0,255), cv.FLOODFILL_MASK_ONLY)
    cv.imshow("fill", image)#(0,0,255)是红色cv.FLOODFILL_MASK_ONLY只填充数值为0的区域

src = cv.imread("image/1.png")
#cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)

# print(src.shape)
# center = src[:80, :150]
# cv.imshow("center", center)
# gray = cv.cvtColor(center, cv.COLOR_BGR2GRAY) #BGR↔Gray 的转换
# back = cv.cvtColor(gray, cv.COLOR_BAYER_BG2BGR) #Gray↔BGR的转换
# src[:80, :150] = back
# cv.imshow("changed", src)

fill_color_demo(src)

#fill_binary_demo()

cv.waitKey(0)
cv.destroyAllWindows()