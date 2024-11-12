import cv2 as cv
import numpy as np
'''
其他形态学操作：
顶帽：原图像和开操作之间的差值图像
黑帽：闭操作和原图像之间的差值图像
形态学梯度：基本梯度：膨胀-腐蚀的差值图像
            内部梯度：原图像-腐蚀
            外部梯度：膨胀-原图像
'''

def hat_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    #ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    dst = cv.morphologyEx(gray, cv.MORPH_BLACKHAT, kernel)
    cimage = np.array(gray.shape, np.uint8)
    cimage = 100
    dst = cv.add(dst, cimage)
    cv.imshow("tophat-result", dst)

def hat_binary_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    cv.imshow("binary", binary)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))
    dst = cv.morphologyEx(binary, cv.MORPH_BLACKHAT, kernel)#BLACKHAT
    cv.imshow("hat_binary-result", dst)

def gradient_demo(image):#膨胀与腐蚀的差别，看上去像前景的轮廓
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    dst = cv.morphologyEx(binary, cv.MORPH_GRADIENT, kernel)#BLACKHAT
    cv.imshow("gradient-result", dst)

def interal_exteral_demo(image):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    dm = cv.dilate(image, kernel)
    em = cv.erode(image, kernel)
    dst1 = cv.subtract(image, em)
    dst2 = cv.subtract(dm, image)
    cv.imshow("interal-result", dst1)
    cv.imshow("exteral-result", dst2)


src = cv.imread("image/4.png")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)

# hat_binary_demo(src)
# gradient_demo(src)
interal_exteral_demo(src)

cv.waitKey(0)
cv.destroyAllWindows()