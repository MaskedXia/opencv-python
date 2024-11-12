import cv2 as cv
import numpy as np

img1 = cv.imread("insulator/1581.jpg")
img2 = cv.imread("image/2.png")
cv.namedWindow("image1", cv.WINDOW_NORMAL)
#后面参数WINDOW_NORMAL可以调整窗口大小，默认cv.WINDOW_AUTOSIZE

dst = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
cv.imshow("image1", dst)
# cv.imshow("image2", img2)

print(img1.shape)

#cv.imwrite("image1/666.jpg", img1)
k = cv.waitKey(0) & 0xFF
if k == 27:#esc值
    cv.destroyAllWindows()
elif k == ord('s'):#它以一个字符（长度为1的字符串）作为参数，返回对应的 ASCII 数值
    cv.imwrite("insulator1/1581.jpg", dst)
    cv.destroyAllWindows()
#cv.destroyAllWindows()
#cv.destroyWindow("image1")