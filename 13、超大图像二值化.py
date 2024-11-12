import cv2 as cv
import numpy as np
#超大图像二值化 （分块）
def big_image_binary(image):
    print(image.shape)
    cw = 256
    ch = 256
    h, w = image.shape[:2]
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    for row in range(0, h, ch):
        for col in range(0, w, cw):
            roi = gray[row:row+ch, col:col+cw]
            dst = cv.adaptiveThreshold(roi, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv.THRESH_BINARY,127, 15)#更好
            #ret, dst = cv.threshold(roi, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
            gray[row:row + ch, col:col + cw] = dst
            print(np.std(dst), np.mean(dst))
    cv.imwrite("image1/006.jpg", gray)




src = cv.imread("image1/001.jpg")
# cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
# cv.imshow("input image", src)

big_image_binary(src)

cv.waitKey(0)
cv.destroyAllWindows()