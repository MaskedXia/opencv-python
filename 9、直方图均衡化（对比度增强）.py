import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
#直方图均衡化（对比度增强，必须是灰度图像）  直方图做一个横向拉伸
def equalHist_demo(image):#直方图均衡化
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    dst = cv.equalizeHist(gray)
    cv.imshow("equalHist", dst)

def equalHist_bgr_demo(image):#彩色直方图均衡化
    yuv = cv.cvtColor(image, cv.COLOR_BGR2YUV)
    yuv[:,:,0] = cv.equalizeHist(yuv[:,:,0])
    dst = cv.cvtColor(yuv, cv.COLOR_YUV2BGR)
    cv.imwrite("insulator1/1072.jpg",dst)
    cv.imshow("equalHist_bgr", dst)

#CLAHE：对比度受限的自适应直方图均衡化
def self_equalHist_demo(image):#局部自适应均衡化,效果更好
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    #分成小块tiles默认 8x8
    dst = clahe.apply(gray)
    cv.imshow("self_equalHist", dst)

def self_equalHist_bgr_demo(image):#彩色局部自适应均衡化
    b,g,r = cv.split(image)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # 分成小块tiles默认 8x8
    br = clahe.apply(b)
    gr = clahe.apply(g)
    rr = clahe.apply(r)
    dst = cv.merge([br,gr,rr])
    cv.imwrite("insulator1/3546.jpg", dst)
    cv.imshow("self_equalHist_bgr", dst)


def create_rgb_hist(image):
    h, w, c = image.shape
    rgbHist = np.zeros([16*16*16, 1], np.float32)
    bsize = 256/16
    for row in range(h):
        for col in range(w):
            b = image[row, col, 0]  # blue
            g = image[row, col, 1]  # green
            r = image[row, col, 2]  # red
            index = (b//bsize)*16*16 + (g//bsize)*16 + (r//bsize)
            rgbHist[np.int(index), 0] = rgbHist[np.int(index), 0] + 1
    return  rgbHist

def hist_compare(im1, im2):
    hist1 = create_rgb_hist(im1)
    hist2 = create_rgb_hist(im2)
    match1 = cv.compareHist(hist1, hist2, cv.HISTCMP_BHATTACHARYYA)
    match2 = cv.compareHist(hist1, hist2, cv.HISTCMP_CORREL)
    match3 = cv.compareHist(hist1, hist2, cv.HISTCMP_CHISQR)
    print("巴氏距离 ：%s， 相关性 ： %s， 卡方 ： %s "%(match1, match2, match3))






src = cv.imread("insulator/3546.jpg")
#cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)

# equalHist_demo(src)
# self_equalHist_demo(src)
#equalHist_bgr_demo(src)
self_equalHist_bgr_demo(src)

# src1 = cv.imread("image/1.png")
# src2 = cv.imread("image/2.png")
# hist_compare(src1, src2)

cv.waitKey(0)
cv.destroyAllWindows()