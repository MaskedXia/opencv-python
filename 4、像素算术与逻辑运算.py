import cv2 as cv
import numpy as np
#像素运算
def add_demo(m1, m2):
    dst = cv.add(m1, m2)
    cv.imshow("add_demo", dst)

def addweight_demo(m1, m2):#图像混合, dst = α · img1 + β · img2 + γ
    dst = cv.addWeighted(m1, 0.7, m2, 0.3, 0)
    cv.imshow("addweight_demo", dst)


def subtract_demo(m1, m2):
    dst = cv.subtract(m1, m2)
    cv.imshow("subtract_demo", dst)

def divide_demo(m1, m2):
    dst = cv.divide(m1, m2)
    cv.imshow("divide_demo", dst)

def multiply_demo(m1, m2):
    dst = cv.multiply(m1, m2)
    cv.imshow("multiply_demo", dst)

def others(m1, m2):
    M1 = cv.mean(m1)
    M2 = cv.mean(m2)
    #print(M1,M2)
    M1, dev1 = cv.meanStdDev(m1)
    M2, dev2 = cv.meanStdDev(m2)
    print(dev1)
    print(dev2)

def logic_demo(m1, m2):#逻辑运算 AND，OR，NOT，XOR
    # dst = cv.bitwise_and(m1, m2)#bitwise：按位
    # cv.imshow("and", dst)
    # dst = cv.bitwise_or(m1, m2)
    # cv.imshow("or", dst)
    # dst = cv.bitwise_not(m1)#取反
    # cv.imshow("not", dst)
    # dst = cv.bitwise_xor(m1, m2)
    # cv.imshow("xor", dst)

    gray = cv.cvtColor(m1, cv.COLOR_BGR2GRAY)
    ret, mask = cv.threshold(gray, 177, 255, cv.THRESH_BINARY)
    #源图像, 阈值， 大于阈值的赋值， 阈值类型 （ps：只能是灰度图像）
    dst = cv.bitwise_and(m1, m2, mask=mask)  # bitwise：按位
    cv.imshow("and_mask", dst)
    dst = cv.bitwise_and(m1, m2)
    cv.imshow("and", dst)

def contrast_brightness_demo(image, c, b):#对比度和亮度
    h, w, ch = image.shape
    blank = np.zeros([h, w, ch], image.dtype)
    dst = cv.addWeighted(image, c, blank, 1-c, b)
    cv.imshow("con_bri_demo", dst)
    cv.imwrite("insulator1/0071.jpg",dst)

src1 = cv.imread("image/2.png")
src2 = cv.imread("insulator/0071.jpg")
#cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
# cv.imshow("2.png", src1)
cv.imshow("3.jpg", src2)
print(src1.shape, src2.shape)

# add_demo(src1, src2)
# addweight_demo(src1, src2)
# subtract_demo(src1, src2)
# divide_demo(src1, src2)
# multiply_demo(src1, src2)
# others(src1, src2)

# logic_demo(src1, src2)

contrast_brightness_demo(src2, 1.4, 15)#对比度，亮度

cv.waitKey(0)
cv.destroyAllWindows()